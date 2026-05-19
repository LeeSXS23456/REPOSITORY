import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from helpfunc import *
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/excess_exposure"
desdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_result/风格因子超额暴露分析"
FAC_RET_PATH = 'E:/SJTU/实习/国泰海通/barra因子/data_base/fac_ret/中证500/factor_returns_07_2604.pkl'


def read_factor_returns(start_date, end_date, weekly_dates):
    with open(FAC_RET_PATH, 'rb') as f:
        factor_df = pickle.load(f)
    factor_df = factor_df.loc[(factor_df.index >= start_date) & (factor_df.index <= end_date)]
    
    weekly_factor_returns = []
    weekly_indices = []
    for i in range(1, len(weekly_dates)):
        week_start = weekly_dates[i-1]
        week_end = weekly_dates[i]
        week_factor = factor_df.loc[(factor_df.index > week_start) & (factor_df.index <= week_end)]
        if not week_factor.empty:
            cum_return = (1 + week_factor).prod() - 1
            weekly_factor_returns.append(cum_return)
            weekly_indices.append(week_end)
    
    if weekly_factor_returns:
        factor_df = pd.DataFrame(weekly_factor_returns, index=weekly_indices)
    return factor_df


def run_factor_exposure_analysis(bmk, start_date=None, end_date=None):
    error = {}
    
    weight_df = read_df(f"{srcdir}/脱敏barra暴露偏离数据2020-2026.xlsx")
    weight_df.columns = weight_df.columns.str.replace(' ', '')
    print(f"   原始数据形状: {weight_df.shape}")
    print(f"   日期范围: {weight_df.index.min().strftime('%Y-%m-%d')} 到 {weight_df.index.max().strftime('%Y-%m-%d')}")
    
    weight_df = weight_df[weight_df['基准'] == bmk]
    print(f"   筛选后数据形状: {weight_df.shape}")
    print(f"   编码数量: {weight_df['编码'].nunique()}")
    print(f"   编码列表: {weight_df['编码'].unique().tolist()}")
    
    style_factors = ["贝塔暴露","账面市值比暴露","盈利率暴露","成长暴露","杠杆率暴露","流动性暴露","动量暴露","非线性市值暴露","残余波动率暴露","规模暴露"]

    print("\n按编码分组计算因子暴露...")
    grouped = weight_df.groupby('编码')
    all_factor_contributions = {}
    all_exposures = {}
    
    zero_ids = defaultdict(list)
    for code, group in grouped:
        print(f"   处理编码: {code}")
        
        exposure_df, nan_dates = filter_df_nan(group.copy(), style_factors)
        if exposure_df.empty:
            print(nan_dates)
            error[code] = f"最新日期为{nan_dates.max().strftime('%Y-%m-%d')}，全部为NaN的行"
            continue
        if not nan_dates.empty:
            print(f"   - 发现年底全为NaN，从 {nan_dates.max().strftime('%Y-%m-%d')} 之后开始")  
        
        exposure_df, zero_dates = filter_df_zero(exposure_df.copy(), style_factors)
        if exposure_df.empty:
            print(zero_dates)
            error[code] = f"最新日期为{zero_dates.max().strftime('%Y-%m-%d')}，全部为0的行"
            continue
        if not zero_dates.empty:
            zero_ids[code].append(exposure_df.index[0]) 
            zero_ids[code].append(zero_dates) 
            print(f"   - 发现全部为0的行，从 {zero_dates.max().strftime('%Y-%m-%d')} 之后开始") 
        
        mask = (exposure_df != 0).any(axis=1)
        exposure_df = exposure_df[mask]
        
        if exposure_df.empty:
            print(f"   - 编码 {code} 处理后数据为空，跳过")
            continue
        
        all_exposures[code] = exposure_df
    
    print("\n3. 风格因子超额暴露分析...")
    import scipy.stats as stats
    
    analysis_results = {}
    
    for code, exposure_df in all_exposures.items():
        if start_date and end_date:
            time_periods = [f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"]
        else:
            time_periods = ['成立以来']
            years = exposure_df.index.year.unique()
            for year in sorted(years):
                time_periods.append(f'{year}年')
                quarters = exposure_df[exposure_df.index.year == year].index.quarter.unique()
                for q in sorted(quarters):
                    time_periods.append(f'{year}年Q{q}')
        
        code_results = {}
        
        for period in time_periods:
            if period == '成立以来':
                period_data = exposure_df
            elif 'Q' in period:
                year, q = int(period[:4]), int(period[-1])
                period_data = exposure_df[(exposure_df.index.year == year) & (exposure_df.index.quarter == q)]
            elif period.endswith('年'):
                period_data = exposure_df[exposure_df.index.year == int(period[:4])]
            else:
                start, end = period.split('_')
                period_data = exposure_df[(exposure_df.index >= pd.to_datetime(start)) & (exposure_df.index <= pd.to_datetime(end))]
            
            if len(period_data) < 2:
                continue
            
            result = {'code': code, '时间段': period}
            
            t_stats = {}
            for factor in style_factors:
                if factor in period_data.columns:
                    data = period_data[factor].dropna()
                    if len(data) >= 2:
                        t_stat, p_value = stats.ttest_1samp(data, 0)
                        t_stats[factor] = t_stat
            
            sorted_t = sorted(t_stats.items(), key=lambda x: abs(x[1]), reverse=True)
            top3 = sorted_t[:3]
            
            for i, (factor, t_val) in enumerate(top3, 1):
                result[f'top{i}_因子'] = factor
                result[f'top{i}_t值'] = t_val
                data = period_data[factor]
                result[f'top{i}_平均值'] = data.mean()
            
            sig_count = sum(1 for t_val in t_stats.values() if abs(t_val) > 1.645)
            result['t绝对值>1.645的因子数'] = sig_count
            
            total_abs_sum = period_data[style_factors].abs().mean().sum()
            result['总风格超额暴露强度'] = total_abs_sum
            
            reversal_factors = {}
            for factor in style_factors:
                if factor in period_data.columns:
                    data = period_data[factor].dropna()
                    if len(data) >= 2:
                        for i in range(len(data) - 1):
                            x_t = data.iloc[i]
                            x_t1 = data.iloc[i+1]
                            if x_t != 0 and x_t * x_t1 < 0:
                                ratio = -x_t1 / x_t
                                if ratio > 0.3:
                                    date = data.index[i+1].strftime('%Y-%m-%d')
                                    if factor not in reversal_factors or ratio > reversal_factors[factor]['ratio']:
                                        reversal_factors[factor] = {'ratio': ratio, 'date': date}
            
            sorted_reversal = {k: f"{v['ratio']:.2%}@{v['date']}" for k, v in sorted(reversal_factors.items(), key=lambda x: x[1]['ratio'], reverse=True)}
            result['反转因子'] = str(sorted_reversal) if sorted_reversal else ''
            
            code_results[period] = result
        
        analysis_results[code] = code_results
    
    print("\n4. 输出分析结果...")
    time_periods = set()
    for code_results in analysis_results.values():
        for period in code_results.keys():
            time_periods.add(period)
    
    if start_date and end_date:
        suffix = f"_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    else:
        suffix = ""
    output_path = f"{desdir}/风格因子超额暴露分析结果{suffix}_{bmk}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        for period in sorted(time_periods):
            period_data = []
            for code, code_results in analysis_results.items():
                if period in code_results:
                    period_data.append(code_results[period])
            
            if period_data:
                df = pd.DataFrame(period_data)
                
                if '时间段' in df.columns:
                    df = df.drop('时间段', axis=1)
                
                cols = df.columns.tolist()
                if 't绝对值>1.645的因子数' in cols:
                    cols.remove('t绝对值>1.645的因子数')
                    cols.insert(0, 't绝对值>1.645的因子数')
                if '总风格超额暴露强度' in cols:
                    cols.remove('总风格超额暴露强度')
                    cols.insert(1, '总风格超额暴露强度')
                
                df = df[cols].set_index('code')
                df.to_excel(writer, sheet_name=period[:31])
    
    print(f"   - 分析结果已保存至: {output_path}")
    return output_path


if __name__ == "__main__":
    start_dt = None
    end_dt = None
    bmk = 905 #905,852,932000
    run_factor_exposure_analysis(bmk, start_dt, end_dt)
