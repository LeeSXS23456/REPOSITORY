import pandas as pd
import numpy as np
from collections import defaultdict
from helpfunc import *
import os
srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/excess_exposure"
desdir = "E:/SJTU/实习/国泰海通/业绩回测/result/中证500指增产品各收益回测_超额收益贡献"
style_factors = [
    "账面市值比因子累计收益",
    "非线性市值因子累计收益",
    "流动性因子累计收益",
    "盈利率因子累计收益",
    "贝塔因子累计收益",
    "规模因子累计收益",
    "动量因子累计收益",
    "杠杆率因子累计收益",
    "残余波动率因子累计收益",
    "成长因子累计收益"
]
id_df = pd.read_excel(f"{srcdir}/脱敏barra暴露偏离数据2020-2026.xlsx",index_col=0)
codes = id_df[id_df['基准'] == 905]['编码'].unique().tolist()
codes = sorted(codes)
print(f"   共有 {len(codes)} 个编码",codes)


if not os.path.exists(desdir):
    os.makedirs(desdir)

zero_ids = defaultdict(list)
start_dates = {}

print("1. 读取数据文件...")
barra_df = pd.read_excel(f"{srcdir}/脱敏barra累计超额收益2020-2026.xlsx", index_col=0)
barra_df.index = pd.to_datetime(barra_df.index)
barra_df.sort_index(inplace=True)

decompose_df = pd.read_excel(f"{srcdir}/脱敏累计超额收益分解2020-2026.xlsx", index_col=0)
decompose_df.index = pd.to_datetime(decompose_df.index)
decompose_df.sort_index(inplace=True)

print(f"   barra累计超额收益数据形状: {barra_df.shape}")
print(f"   累计超额收益分解数据形状: {decompose_df.shape}")

print("\n2. 按编码分组处理...")
grouped = barra_df.groupby("编码")


all_results = {}

#DEBUG 
#codes = [7]
for code in codes:
    print(f"\n   处理编码: {code}")
    
    if code not in grouped.groups:
        print(f"   - 编码 {code} 不存在于分组中，跳过")
        continue
    
    barra_group = grouped.get_group(code)
    print(barra_group)
    zero_mask = (barra_group[style_factors] == 0).all(axis=1) 
    zero_dates = barra_group[zero_mask].index
    
    if not zero_dates.empty:
        zero_ids[code].append(barra_group.index[0])
        zero_ids[code].append(len(zero_dates))
        zero_ids[code].append(zero_dates)
        latest_zero_date = zero_dates.max()
        barra_group = barra_group[barra_group.index > latest_zero_date]
        print(f"   - 发现{code}全部为0的行，从 {latest_zero_date.strftime('%Y-%m-%d')} 之后开始")
        
        if len(barra_group) == 0:
            continue
            
        
    
    barra_nav = 1 + barra_group
    caculate_date = barra_nav.index[0]
    start_dates[code] = caculate_date
    
    
    if "编码" in decompose_df.columns:
        decompose_group = decompose_df[decompose_df["编码"] == code]
    else:
        print(f"缺少{code}的累计超额收益")
        continue
    
    decompose_nav = 1 + decompose_group
    
    common_dates = barra_nav.index.intersection(decompose_nav.index)
    barra_nav = barra_nav.loc[common_dates]
    decompose_nav = decompose_nav.loc[common_dates]

    barra_nav = barra_nav / barra_nav.iloc[0] #因为有了截取，所以要归一化到1
    decompse_nav = decompose_nav /decompose_nav.iloc[0] #因为有了截取，所以要归一化到1
    

    product_results = {}
    
    for col in style_factors:
        nav_series = barra_nav[col].dropna()
        if len(nav_series) >= 2:
            result = calculate_performance(nav_series, freq='W')
            product_results[col] = result
    
    for col in decompose_nav.columns:
        if col == "编码":
            continue
        nav_series = decompose_nav[col].dropna()
        if len(nav_series) >= 2:
            result = calculate_performance(nav_series, freq='W')
            product_results[col] = result
    
    all_results[code] = product_results
    print(f"   - 已完成 {len(product_results)} 个指标的回测")

print("\n3. 按指标列输出...")
all_cols = set()
for code, results in all_results.items():
    for col in results.keys():
        all_cols.add(col)

for col in all_cols:
    col_results = {}
    all_periods = set()
    
    for code, results in all_results.items():
        if col in results:
            col_results[code] = results[col]
            for period in results[col]['时间区间']:
                all_periods.add(period)
    
    if not col_results:
        continue
    
    safe_col_name = col.replace('/', '_').replace('\\', '_')
    output_path = f"{desdir}/{safe_col_name}_回测结果.xlsx"
    
    periods = sorted(all_periods)
    
    with pd.ExcelWriter(output_path) as writer:
        for period in sorted(periods):
            period_data = []
            codes = []
            
            for code, result in col_results.items():
                row = result[result['时间区间'] == period]
                if not row.empty:
                    period_data.append(row.iloc[0].drop('时间区间').values)
                    codes.append(code)
            
            if period_data:
                period_df = pd.DataFrame(period_data, index=codes, columns=list(col_results.values())[0].columns.drop('时间区间'))
                period_df.to_excel(writer, sheet_name=str(period)[:31])
    
    print(f"   - {col} 回测结果已保存")
    pd.DataFrame(zero_ids).T.to_excel(f"{desdir}/基金中断区间.xlsx")
    pd.DataFrame([start_dates]).T.to_excel(f"{desdir}/基金开始计算日期.xlsx")

print("\n所有回测结果已保存完成！")