# 计算任意时间段内的因子累计收益率（至少三个月）
import pandas as pd
import numpy as np
import os

facdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/fac_ret/中证500"
desdir = "E:/SJTU/实习/国泰海通/业绩回测/result/三月风格因子净值"
factors_905 = ["non_linear_size", "size", "liquidity", "beta", "momentum"]

df_fac = pd.read_pickle(f"{facdir}/factor_returns_07_2604.pkl")

# 只取前11个因子，排除comovement列
if 'comovement' in df_fac.columns:
    df_fac = df_fac.drop('comovement', axis=1)
df_fac = df_fac.iloc[:, :10]  # 确保只取前10列

# 设置起始年份
start_year = 2020

# 获取所有月份
df_fac['year_month'] = df_fac.index.to_period('M')
all_months = df_fac['year_month'].unique()
all_months = sorted([m for m in all_months if m.year >= start_year])

# 存储满足条件的区间结果
selected_results = []
selected_navs = {}

for month in all_months:
    # 计算三个月区间：当前月 ~ 当前月+2月
    end_month = month + 2
    if end_month > all_months[-1]:
        break
    
    # 获取区间内的数据
    mask = (df_fac['year_month'] >= month) & (df_fac['year_month'] <= end_month)
    period_data = df_fac[mask].drop('year_month', axis=1)
    
    if len(period_data) < 50:  # 至少需要50个交易日
        continue
    
    # 计算累计收益率
    cum_returns = (1 + period_data).prod() - 1
    
    # 按累计收益率排序（收益越大，rank越小）
    ranking = cum_returns.sort_values(ascending=False).reset_index()
    ranking.columns = ['factor', 'cum_return']
    ranking['rank'] = ranking.index + 1
    
    # 检查 factors_905 中有多少个因子进入前三名
    top3_factors = set(ranking[ranking['rank'] <= 3]['factor'])
    matched_factors = [f for f in factors_905 if f in top3_factors]
    
    if len(matched_factors) >= 2:
        # 打印结果
        print(f"\n{'='*60}")
        print(f"时间区间: {month.strftime('%Y-%m')} ~ {end_month.strftime('%Y-%m')}")
        print(f"进入前三名的目标因子: {', '.join(matched_factors)}")
        print(f"\n{factors_905} 因子表现:")
        
        for factor in factors_905:
            if factor in cum_returns:
                rank = ranking[ranking['factor'] == factor]['rank'].iloc[0]
                print(f"  {factor}: 累计收益 = {cum_returns[factor]:.4f}, 排名 = {rank}")
        
        # 计算每日因子净值
        nav_data = (1 + period_data).cumprod()
        nav_data = nav_data.reset_index()
        nav_data['period'] = f"{month.strftime('%Y%m')}_{end_month.strftime('%Y%m')}"
        
        selected_navs[f"{month.strftime('%Y%m')}_{end_month.strftime('%Y%m')}"] = nav_data
        
        # 存储结果
        result_info = {
            'start_month': month.strftime('%Y-%m'),
            'end_month': end_month.strftime('%Y-%m'),
            'matched_factors': ', '.join(matched_factors),
            'num_matched': len(matched_factors)
        }
        for factor in factors_905:
            if factor in cum_returns:
                result_info[f'{factor}_return'] = cum_returns[factor]
                result_info[f'{factor}_rank'] = ranking[ranking['factor'] == factor]['rank'].iloc[0]
        selected_results.append(result_info)

# 保存结果到Excel
output_path = f"{desdir}/因子累计收益分析结果.xlsx"
with pd.ExcelWriter(output_path) as writer:
    # 保存满足条件的区间汇总
    if selected_results:
        summary_df = pd.DataFrame(selected_results)
        summary_df.to_excel(writer, sheet_name='满足条件的区间汇总', index=False)
    
    # 保存每个满足条件区间的每日净值
    for period, nav_df in selected_navs.items():
        nav_df.to_excel(writer, sheet_name=f'净值_{period}', index=False)

print(f"\n分析结果已保存至: {output_path}")