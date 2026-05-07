import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

srcdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_exposure"
desdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_result"
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
    
    for code, group in grouped:
        print(f"   处理编码: {code}")
        
        exposure_df = group[style_factors].copy().dropna()
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
    
    style_factor_dir = f"{desdir}/风格因子收益贡献净值"
    if not os.path.exists(style_factor_dir):
        os.makedirs(style_factor_dir)
    
    for code, contributions in all_factor_contributions.items():
        plt.figure(figsize=(15, 10))
        for factor in style_factors:
            if factor in contributions.columns:
                cum_return = (1 + contributions[factor]).cumprod()
                if not cum_return.empty:
                    cum_return = cum_return / cum_return.iloc[0]
                    plt.plot(cum_return.index, cum_return.values, label=factor_name_map[factor], alpha=0.7)
        plt.title(f'Code {code} - Style Factor Net Value')
        plt.xlabel('Date')
        plt.ylabel('Net Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{style_factor_dir}/{code}_因子净值曲线.png")
        plt.close()
        print(f"   - 编码 {code} 因子净值曲线已保存")
    
    for code, contributions in all_factor_contributions.items():
        code_dir = f"{desdir}/{code}"
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        
        exposure_df = all_exposures[code]
        
        for factor in style_factors:
            if factor in contributions.columns and factor in exposure_df.columns:
                fig, ax1 = plt.subplots(figsize=(15, 8))
                
                common_dates = contributions.index.intersection(exposure_df.index)
                if len(common_dates) == 0:
                    continue
                
                exposure_data = exposure_df.loc[common_dates, factor]
                contribution_data = contributions.loc[common_dates, factor]
                
                factor_name = factor_name_map[factor]
                if factor_name in factor_returns.columns:
                    factor_return_data = factor_returns.loc[common_dates, factor_name]
                else:
                    factor_return_data = pd.Series(index=common_dates)
                
                ax1.bar(exposure_data.index, exposure_data.values, alpha=0.7, label=f'{factor_name} Exposure', color='blue')
                ax1.set_xlabel('Date')
                ax1.set_ylabel(f'{factor_name} Exposure', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_xticks(exposure_data.index[::max(1, len(exposure_data)//10)])
                ax1.set_xticklabels(exposure_data.index[::max(1, len(exposure_data)//10)].strftime('%Y-%m-%d'), rotation=45)
                
                ax2 = ax1.twinx()
                
                cum_return_contribution = (1 + contribution_data).cumprod()
                if not cum_return_contribution.empty:
                    cum_return_contribution = cum_return_contribution / cum_return_contribution.iloc[0]
                    ax2.plot(cum_return_contribution.index, cum_return_contribution.values, 'r-', linewidth=2, label=f'{factor_name} Contribution')
                
                if not factor_return_data.empty:
                    cum_return_factor = (1 + factor_return_data).cumprod()
                    cum_return_factor = cum_return_factor / cum_return_factor.iloc[0]
                    ax2.plot(cum_return_factor.index, cum_return_factor.values, 'g--', linewidth=2, label=f'{factor_name} Factor Return')
                
                ax2.set_ylabel('Net Value', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.title(f'Code {code} - {factor_name} Exposure & Net Value')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{code_dir}/{factor_name}_exposure_net_value.png")
                plt.close()

        print(f"   - 编码 {code} 暴露净值图片已保存")

if __name__ == "__main__":
    main()