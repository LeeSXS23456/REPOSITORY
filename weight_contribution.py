import pandas as pd
import numpy as np
from collections import defaultdict
from helpfunc import *
import os

srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/excess_exposure"
desdir = "E:/SJTU/实习/国泰海通/业绩回测/result/中证500指增产品各收益回测_超额收益贡献"

FREQ = 'W'

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


def run_contribution_analysis(bmk,start_date=None, end_date=None):
    id_df = pd.read_excel(f"{srcdir}/脱敏barra暴露偏离数据2020-2026.xlsx", index_col=0)
    codes = id_df[id_df['基准']==bmk]['编码'].unique().tolist()
    codes = sorted(codes)
    print(f"   共有 {len(codes)} 个编码", codes)

    zero_ids = defaultdict(list)
    start_dates = {}
    all_results = {}
    non_exist_codes = []
    error_codes = {}

    print("1. 读取数据文件...")
    barra_df = read_df(f"{srcdir}/脱敏barra累计超额收益2020-2026.xlsx")
    decompose_df = read_df(f"{srcdir}/脱敏累计超额收益分解2020-2026.xlsx")
    print(f"   barra累计超额收益数据形状: {barra_df.shape}")
    print(f"   累计超额收益分解数据形状: {decompose_df.shape}")

    print("\n2. 按编码分组处理...")
    grouped = barra_df.groupby("编码")
    
    for code in codes:
        print(f"\n   处理编码: {code}")
        if code not in grouped.groups:
            non_exist_codes.append(code)
            continue

        barra_group = grouped.get_group(code)
        barra_group, zero_dates = filter_df_zero(barra_group, style_factors)
        zero_ids[code] = zero_dates.tolist()
        if len(barra_group) == 0:
            error_codes[code] = "提前终止"
            continue

        if start_date and end_date:
            if barra_group.index.min() > end_date or barra_group.index.max() < start_date:
                error_codes[code] = "时间区间超出数据范围"
                print(f"     警告：{code} 时间区间超出数据范围，跳过该编码")
                continue

        barra_nav = 1 + barra_group
        std = barra_nav.index[0]
        start_dates[code] = std

        decompose_group = decompose_df[decompose_df["编码"] == code]
        decompose_nav = 1 + decompose_group

        barra_nav, decompose_nav = align_dates(barra_nav, decompose_nav)
        barra_nav = barra_nav / barra_nav.iloc[0]
        decompose_nav = decompose_nav / decompose_nav.iloc[0]

        product_results = {}

        for col in style_factors:
            nav_series_style = barra_nav[col].dropna()
            if len(nav_series_style) < 2:
                error_codes[code] = f"{code}_{col}净值数据不足2个"
                continue

            result = calculate_performance(nav_series_style, freq=FREQ, start_date=start_date, end_date=end_date)
            product_results[col] = result

        for col in decompose_nav.columns:
            if col == "编码":
                continue
            nav_series_decompose = decompose_nav[col].dropna()
            if len(nav_series_decompose) < 2:
                error_codes[code] = f"{code}_{col}净值数据不足2个"
                continue

            result = calculate_performance(nav_series_decompose, freq=FREQ, start_date=start_date, end_date=end_date)
            product_results[col] = result

        all_results[code] = product_results
        print(f"   - 已完成 {len(product_results)} 个指标的回测")

    print("\n4. 区间收益分析...")
    #if start_date and end_date:
    interval_results = []
    if not all_results:
        print("     警告：没有有效数据，跳过该分析")
        return None
    
    for code, results in all_results.items():
        row = {'code': code}
        for col in results.keys():
            result = results[col]
            mask = (result['时间区间'] == (f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}' if start_date and end_date else "成立以来"))
            # print(mask)
            # print(result)
            if mask.any():
                row[col] = result.loc[mask, '区间收益'].iloc[0]

        interval_results.append(row)

    interval_df = pd.DataFrame(interval_results)
    interval_df['PA强度'] = (interval_df['累计残差贡献'] - interval_df['累计风格因子贡献']) / abs(interval_df['累计风格因子贡献'])

    p_cols = ['code'] + ['PA强度'] + ['累计超额收益', '累计风格因子贡献', '累计行业因子贡献', '累计残差贡献']
    cols = p_cols + [c for c in interval_df.columns if c not in p_cols]
    interval_df = interval_df[cols]

    cols_to_rank = [c for c in interval_df.columns if c not in ['code', 'PA强度']]
    for col in cols_to_rank:
        interval_df[f'{col}_排名'] = interval_df[col].rank(ascending=False, method='min')

    detailed_results = []
    for code, results in all_results.items():
        for col, result in results.items():
            for _, row in result.iterrows():
                detailed_row = {'code': code, '指标': col}
                detailed_row.update(row.to_dict())
                detailed_results.append(detailed_row)

    detailed_df = pd.DataFrame(detailed_results)

    suffix = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}" if start_date and end_date else "成立以来"
    output_path = f"{desdir}/区间收益分析_{suffix}_{bmk}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        interval_df.to_excel(writer, sheet_name='区间收益分析', index=False)
        detailed_df.to_excel(writer, sheet_name='详细信息', index=False)

    print(f"   - 区间收益分析已保存（含详细信息sheet）")

    pd.DataFrame([zero_ids]).T.to_excel(f"{desdir}/基金中断区间.xlsx")
    pd.DataFrame([start_dates]).T.to_excel(f"{desdir}/基金开始计算日期.xlsx")
    pd.DataFrame([error_codes]).T.to_excel(f"{desdir}/错误代码.xlsx")
    print("\n所有回测结果已保存完成！")

    return output_path


if __name__ == "__main__":
    start_dt = pd.to_datetime('2025-06-01')
    end_dt = pd.to_datetime('2026-08-31')
    bmk = 905 #905,852,932000
    run_contribution_analysis(bmk,start_dt, end_dt)
