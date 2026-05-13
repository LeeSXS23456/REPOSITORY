import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict



plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

srcdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_exposure"
desdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_result"
FUND_NAV_PATH = 'E:/SJTU/实习/国泰海通/业绩回测/nav/纯中证500净值数据.xlsx'
FAC_RET_PATH = 'E:/SJTU/实习/国泰海通/barra因子/data_base/fac_ret/中证500/factor_returns_07_2604.pkl'

if not os.path.exists(desdir):
    os.makedirs(desdir)

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

def main():
    weight_df = pd.read_excel(f"{srcdir}/脱敏barra暴露偏离数据2020-2026.xlsx", index_col=0)
    weight_df.index = pd.to_datetime(weight_df.index)
    weight_df.sort_index(inplace=True)
    weight_df.columns = weight_df.columns.str.replace(' ', '')
    print(f"   原始数据形状: {weight_df.shape}")
    print(f"   日期范围: {weight_df.index.min().strftime('%Y-%m-%d')} 到 {weight_df.index.max().strftime('%Y-%m-%d')}")
    
    print("\n2. 筛选基准==905的产品...")
    weight_df = weight_df[weight_df['基准'] == 905]
    print(f"   筛选后数据形状: {weight_df.shape}")
    print(f"   编码数量: {weight_df['编码'].nunique()}")
    print(f"   编码列表: {weight_df['编码'].unique().tolist()}")
    
    style_factors = ["贝塔暴露","账面市值比暴露","盈利率暴露","成长性暴露","杠杆率暴露","流动性暴露","动量暴露","非线性市值暴露","残余波动率暴露","规模暴露"]
    factor_name_map = {
        '流动性暴露': 'liquidity',
        '杠杆率暴露': 'leverage',
        '残余波动率暴露': 'residual_volatility',
        '盈利率暴露': 'earnings_yield',
        '非线性市值暴露': 'non_linear_size',
        '贝塔暴露': 'beta',
        '账面市值比暴露': 'book_to_price',
        '成长性暴露': 'growth',
        '动量暴露': 'momentum',
        '规模暴露': 'size'
    }

    print("\n3. 读取因子收益率数据...")
    weekly_dates = weight_df.index
    factor_returns = read_factor_returns(weight_df.index.min(), weight_df.index.max(), weekly_dates)
    print(f"   周频因子数据形状: {factor_returns.shape}")
    print(f"   因子数量: {len(factor_returns.columns)}")
    
    print("\n4. 按编码分组计算因子贡献...")
    grouped = weight_df.groupby('编码')
    all_factor_contributions = {}
    all_exposures = {}
    
    year_end_dates = pd.to_datetime([
        '2025-12-31', '2024-12-31', '2023-12-31', 
        '2022-12-31', '2021-12-31', '2020-12-31'
    ])
    
    zero_ids = defaultdict(list)
    for code, group in grouped:
        print(f"   处理编码: {code}")
        
        exposure_df = group[style_factors].copy()
        
        na_mask = exposure_df.isna().all(axis=1)
        na_dates = exposure_df[na_mask].index
        valid_year_end_dates = na_dates.intersection(year_end_dates)
        
        if not valid_year_end_dates.empty:
            latest_na_date = valid_year_end_dates.max()
            exposure_df = exposure_df[exposure_df.index > latest_na_date]
            print(f"   - 发现年底全为NaN，从 {latest_na_date.strftime('%Y-%m-%d')} 之后开始")
        
        zero_mask = (exposure_df == 0).all(axis=1)
        zero_dates = exposure_df[zero_mask].index
        
        if not zero_dates.empty:
            zero_ids[code].append(exposure_df.index[0]) #起始日期+
            zero_ids[code].append(zero_dates) #空缺日期
            latest_zero_date = zero_dates.max()
            exposure_df = exposure_df[exposure_df.index > latest_zero_date]
            print(f"   - 发现全部为0的行，从 {latest_zero_date.strftime('%Y-%m-%d')} 之后开始")
        
        exposure_df = exposure_df.dropna()
        
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
        years = exposure_df.index.year.unique()
        time_periods = ['成立以来'] + [f'{year}年' for year in sorted(years)]
        
        code_results = {}
        
        for period in time_periods:
            if period == '成立以来':
                period_data = exposure_df
            else:
                year = int(period[:4])
                period_data = exposure_df[exposure_df.index.year == year]
            
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
                data = period_data[factor]#.dropna()
                result[f'top{i}_平均绝对值'] = data.abs().mean()
            
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
    
    output_path = f"{desdir}/风格因子超额暴露分析结果.xlsx"
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


    ###根据因子超额暴露计算因子贡献
    #     exposure_df_shifted = exposure_df.shift(1).dropna()
    #     common_dates = exposure_df_shifted.index.intersection(factor_returns.index)
        
    #     if len(common_dates) == 0:
    #         print(f"   - 编码 {code} 没有匹配的日期，跳过")
    #         continue
        
    #     exposure_aligned = exposure_df_shifted.loc[common_dates]
    #     factor_returns_aligned = factor_returns.loc[common_dates]
        
    #     factor_contributions = pd.DataFrame(index=common_dates)
    #     for style_factor in style_factors:
    #         factor_name = factor_name_map[style_factor]
    #         if factor_name in factor_returns_aligned.columns:
    #             factor_contributions[style_factor] = exposure_aligned[style_factor] * factor_returns_aligned[factor_name]
        
    #     all_factor_contributions[code] = factor_contributions
    #     all_exposures[code] = exposure_df.loc[common_dates]
    #     print(f"   - 编码 {code} 处理完成，有效日期数: {len(common_dates)}")
    
    # print(f"\n   处理完成的编码数量: {len(all_factor_contributions)}")
    
    # print("\n5. 计算超额收益率并保存...")
    # fund_nav = pd.read_excel(FUND_NAV_PATH, index_col=0)
    # fund_nav.index = pd.to_datetime(fund_nav.index)
    # fund_nav.columns = fund_nav.columns.astype(str)
    
    # index_returns = fund_nav.iloc[:, 0].pct_change().dropna()
    
    # exreturns_dir = f"{desdir}/EXreturns"
    # exnav_dir = f"{desdir}/EXnav"
    # if not os.path.exists(exreturns_dir):
    #     os.makedirs(exreturns_dir)
    # if not os.path.exists(exnav_dir):
    #     os.makedirs(exnav_dir)
    
    # for code in all_factor_contributions.keys():
    #     if str(code) in fund_nav.columns:
    #         fund_returns = fund_nav[str(code)].pct_change().dropna()
            
    #         excess_returns = fund_returns - index_returns.reindex(fund_returns.index)
            
    #         contributions = all_factor_contributions[code]
            
    #         common_dates = excess_returns.index.intersection(contributions.index)
            
    #         result_df = pd.DataFrame({
    #             'excess_return': excess_returns.loc[common_dates]
    #         })
            
    #         for factor in style_factors:
    #             if factor in contributions.columns:
    #                 english_factor_name = factor_name_map[factor]
    #                 result_df[english_factor_name] = contributions.loc[common_dates, factor]
            
    #         result_df.to_excel(f"{exreturns_dir}/{code}_超额收益与因子贡献.xlsx")
            
    #         nav_df = (1 + result_df).cumprod()
    #         nav_df = nav_df / nav_df.iloc[0]
    #         nav_df.to_excel(f"{exnav_dir}/{code}_超额收益与因子贡献[净值].xlsx")
            
    #         print(f"   - 编码 {code} 超额收益与因子贡献及净值已保存")
    #     else:
    #         print(f"   - 编码 {code} 在净值数据中不存在，跳过")

if __name__ == "__main__":
    main()