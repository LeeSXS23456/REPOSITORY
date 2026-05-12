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
        
        exposure_df_shifted = exposure_df.shift(1).dropna()
        common_dates = exposure_df_shifted.index.intersection(factor_returns.index)
        
        if len(common_dates) == 0:
            print(f"   - 编码 {code} 没有匹配的日期，跳过")
            continue
        
        exposure_aligned = exposure_df_shifted.loc[common_dates]
        factor_returns_aligned = factor_returns.loc[common_dates]
        
        factor_contributions = pd.DataFrame(index=common_dates)
        for style_factor in style_factors:
            factor_name = factor_name_map[style_factor]
            if factor_name in factor_returns_aligned.columns:
                factor_contributions[style_factor] = exposure_aligned[style_factor] * factor_returns_aligned[factor_name]
        
        all_factor_contributions[code] = factor_contributions
        all_exposures[code] = exposure_df.loc[common_dates]
        print(f"   - 编码 {code} 处理完成，有效日期数: {len(common_dates)}")
    
    print(f"\n   处理完成的编码数量: {len(all_factor_contributions)}")
    
    # style_factor_dir = f"{desdir}/风格因子收益贡献净值"
    # if not os.path.exists(style_factor_dir):
    #     os.makedirs(style_factor_dir)
    
    # for code, contributions in all_factor_contributions.items():
    #     plt.figure(figsize=(15, 10))
    #     for factor in style_factors:
    #         if factor in contributions.columns:
    #             cum_return = (1 + contributions[factor]).cumprod()
    #             if not cum_return.empty:
    #                 cum_return = cum_return / cum_return.iloc[0]
    #                 plt.plot(cum_return.index, cum_return.values, label=factor_name_map[factor], alpha=0.7)
    #     plt.title(f'Code {code} - Style Factor Net Value')
    #     plt.xlabel('Date')
    #     plt.ylabel('Net Value')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(f"{style_factor_dir}/{code}_因子净值曲线.png")
    #     plt.close()
    #     print(f"   - 编码 {code} 因子净值曲线已保存")
    
    # for code, contributions in all_factor_contributions.items():
    #     code_dir = f"{desdir}/EXexposure_with_EXfactor_return/{code}"
    #     if not os.path.exists(code_dir):
    #         os.makedirs(code_dir)
        
    #     exposure_df = all_exposures[code]
        
    #     for factor in style_factors:
    #         if factor in contributions.columns and factor in exposure_df.columns:
    #             fig, ax1 = plt.subplots(figsize=(15, 8))
                
    #             common_dates = contributions.index.intersection(exposure_df.index)
    #             if len(common_dates) == 0:
    #                 continue
                
    #             exposure_data = exposure_df.loc[common_dates, factor]
    #             contribution_data = contributions.loc[common_dates, factor]
                
    #             factor_name = factor_name_map[factor]
    #             if factor_name in factor_returns.columns:
    #                 factor_return_data = factor_returns.loc[common_dates, factor_name]
    #             else:
    #                 factor_return_data = pd.Series(index=common_dates)
                
    #             ax1.bar(exposure_data.index, exposure_data.values, alpha=0.7, label=f'{factor_name} Exposure', color='blue')
    #             ax1.set_xlabel('Date')
    #             ax1.set_ylabel(f'{factor_name} Exposure', color='blue')
    #             ax1.tick_params(axis='y', labelcolor='blue')
    #             ax1.set_xticks(exposure_data.index[::max(1, len(exposure_data)//10)])
    #             ax1.set_xticklabels(exposure_data.index[::max(1, len(exposure_data)//10)].strftime('%Y-%m-%d'), rotation=45)
                
    #             ax2 = ax1.twinx()
                
    #             cum_return_contribution = (1 + contribution_data).cumprod()
    #             if not cum_return_contribution.empty:
    #                 cum_return_contribution = cum_return_contribution / cum_return_contribution.iloc[0]
    #                 ax2.plot(cum_return_contribution.index, cum_return_contribution.values, 'r-', linewidth=2, label=f'{factor_name} Contribution')
                
    #             if not factor_return_data.empty:
    #                 cum_return_factor = (1 + factor_return_data).cumprod()
    #                 cum_return_factor = cum_return_factor / cum_return_factor.iloc[0]
    #                 ax2.plot(cum_return_factor.index, cum_return_factor.values, 'g--', linewidth=2, label=f'{factor_name} Factor Return')
                
    #             ax2.set_ylabel('Net Value', color='red')
    #             ax2.tick_params(axis='y', labelcolor='red')
                
    #             lines1, labels1 = ax1.get_legend_handles_labels()
    #             lines2, labels2 = ax2.get_legend_handles_labels()
    #             ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
    #             plt.title(f'Code {code} - {factor_name} Exposure & Net Value')
    #             plt.grid(True)
    #             plt.tight_layout()
    #             plt.savefig(f"{code_dir}/{factor_name}_exposure_net_value.png")
    #             plt.close()

    #     print(f"   - 编码 {code} 暴露净值图片已保存")
    
    print("\n5. 计算超额收益率并保存...")
    fund_nav = pd.read_excel(FUND_NAV_PATH, index_col=0)
    fund_nav.index = pd.to_datetime(fund_nav.index)
    fund_nav.columns = fund_nav.columns.astype(str)
    
    index_returns = fund_nav.iloc[:, 0].pct_change().dropna()
    
    exreturns_dir = f"{desdir}/EXreturns"
    exnav_dir = f"{desdir}/EXnav"
    if not os.path.exists(exreturns_dir):
        os.makedirs(exreturns_dir)
    if not os.path.exists(exnav_dir):
        os.makedirs(exnav_dir)
    
    for code in all_factor_contributions.keys():
        if str(code) in fund_nav.columns:
            fund_returns = fund_nav[str(code)].pct_change().dropna()
            
            excess_returns = fund_returns - index_returns.reindex(fund_returns.index)
            
            contributions = all_factor_contributions[code]
            
            common_dates = excess_returns.index.intersection(contributions.index)
            
            result_df = pd.DataFrame({
                'excess_return': excess_returns.loc[common_dates]
            })
            
            for factor in style_factors:
                if factor in contributions.columns:
                    english_factor_name = factor_name_map[factor]
                    result_df[english_factor_name] = contributions.loc[common_dates, factor]
            
            result_df.to_excel(f"{exreturns_dir}/{code}_超额收益与因子贡献.xlsx")
            
            nav_df = (1 + result_df).cumprod()
            nav_df = nav_df / nav_df.iloc[0]
            nav_df.to_excel(f"{exnav_dir}/{code}_超额收益与因子贡献[净值].xlsx")
            
            print(f"   - 编码 {code} 超额收益与因子贡献及净值已保存")
        else:
            print(f"   - 编码 {code} 在净值数据中不存在，跳过")

if __name__ == "__main__":
    main()