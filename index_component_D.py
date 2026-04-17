import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
basdir = "E:/SJTU/实习/国泰海通/barra因子/data_base"
srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/index_component_日频"
bardir = "E:/SJTU/实习/国泰海通/barra因子/data_base/barra_data/全A_M/米筐全A_20_26D"
retdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/fac_ret/whole_mkt"
desdir = "E:/SJTU/实习/国泰海通/barra因子/result/指数构成/日频"
tds = pd.read_pickle(f"{basdir}/trading_dates.pkl")
target_lst = ['000300.XSHG','000905.XSHG', '000510.XSHG', '000852.XSHG','932000.INDX','000922.XSHG']
a = "866011.RI"

# 风格因子列表
style_factors = ['size', 'non_linear_size', 'momentum', 'liquidity',
       'book_to_price', 'leverage', 'growth', 'earnings_yield', 'beta',
       'residual_volatility']

# 创建目标目录
os.makedirs(desdir, exist_ok=True)

# 读取全A指数权重数据
a_file = f"{srcdir}/{a}_20_26D_dict.pkl"
with open(a_file, 'rb') as f:
    a_dict = pickle.load(f)
print(f"全A指数数据加载完成，包含 {len(a_dict)} 个交易日")

# 读取因子收益率数据
df_facret = pd.read_pickle(f"{retdir}/factor_returns_20_2603.pkl")
print(f"因子收益率数据加载完成，时间范围: {df_facret.index.min()} 到 {df_facret.index.max()}")

# 处理所有目标指数
for target in target_lst:
    print(f"\n=== 处理指数: {target} ===")
    
    # 为每个目标指数创建单独的文件夹
    target_desdir = f"{desdir}/{target}"
    os.makedirs(target_desdir, exist_ok=True)
    
    # 读取目标指数权重数据
    target_file = f"{srcdir}/{target}_20_26D_dict.pkl"
    with open(target_file, 'rb') as f:
        target_dict = pickle.load(f)
    print(f"目标指数数据加载完成，包含 {len(target_dict)} 个交易日")
    
    # 获取所有日期
    all_dates = sorted(target_dict.keys())
    print(f"总天数: {len(all_dates)}")
    
    # 存储每天的因子暴露
    factor_exposure_dict = {}
    relative_factor_exposure_dict = {}
    
    # 存储每天的因子超额暴露贡献率
    factor_contribution_dict = {}
    
    # 处理每一天
    for i, current_date in enumerate(all_dates):
        print(f"\n处理日期: {current_date} ({i+1}/{len(all_dates)})")
        
        # 获取该日期的成分股权重
        target_weights = target_dict[current_date]
        a_weights = a_dict.get(current_date, pd.Series())
        
        # 读取当天的因子暴露数据
        exp_file = f"{bardir}/{current_date.strftime('%Y-%m-%d')}.pkl"
        if not os.path.exists(exp_file):
            print(f"警告: {exp_file} 不存在")
            continue
        
        with open(exp_file, 'rb') as f:
            exposure = pickle.load(f)
        
        # 计算目标指数的因子暴露
        if not target_weights.empty:
            df_target = target_weights.to_frame('weight')
            df_target_exp = df_target.merge(exposure, left_index=True, right_index=True, how='left')
            if df_target_exp.isnull().any(axis=1).sum():
                print(f"警告: {target} {current_date} 有缺失值，个数为：{df_target_exp.isnull().any(axis=1).sum()}")
                
            
            if 'weight' in df_target_exp.columns and len(df_target_exp) > 0:
                daily_exp = df_target_exp[style_factors].mul(df_target_exp['weight'], axis=0).sum()
                factor_exposure_dict[current_date] = daily_exp
        
        # 计算全A的因子暴露
        daily_a_exp = None
        if not a_weights.empty:
            df_a = a_weights.to_frame('weight')
            df_a_exp = df_a.merge(exposure, left_index=True, right_index=True, how='left')
            
            if 'weight' in df_a_exp.columns and len(df_a_exp) > 0:
                df_a_exp["weight"] = df_a_exp["weight"] / df_a_exp["weight"].sum() #全A中的北交所股票暂时没有因子暴露
                daily_a_exp = df_a_exp[style_factors].mul(df_a_exp['weight'], axis=0).sum()
        
        # 计算相对于全A的暴露
        if daily_exp is not None and daily_a_exp is not None:
            relative_exp = daily_exp - daily_a_exp
            relative_factor_exposure_dict[current_date] = relative_exp
        
        # 计算因子超额暴露贡献率（乘以第二天的因子收益率）
        next_date_str = tds[tds > current_date.strftime('%Y-%m-%d')].iloc[0]
        if next_date_str in df_facret.index and current_date in relative_factor_exposure_dict:
            relative_exp = relative_factor_exposure_dict[current_date]
            daily_facret = df_facret.loc[next_date_str]
            
            # 确保因子顺序对齐
            common_factors = [f for f in style_factors if f in daily_facret.index]
            if common_factors:
                factor_contribution = relative_exp[common_factors] * daily_facret[common_factors]
                factor_contribution_dict[current_date] = factor_contribution
    
    # 保存因子暴露数据
    if factor_exposure_dict:
        absolute_exposure_df = pd.DataFrame(factor_exposure_dict).T
        relative_exposure_df = pd.DataFrame(relative_factor_exposure_dict).T
        
        # 保存到Excel文件
        exposure_file = f"{target_desdir}/{target}_factor_exposure.xlsx"
        with pd.ExcelWriter(exposure_file) as writer:
            absolute_exposure_df.to_excel(writer, sheet_name='绝对暴露')
            relative_exposure_df.to_excel(writer, sheet_name='相对全A暴露')
        print(f"\n因子暴露数据已保存到: {exposure_file}")
    
    # 保存因子超额暴露贡献率数据
    if factor_contribution_dict:
        factor_contribution_df = pd.DataFrame(factor_contribution_dict).T
        
        # 计算因子贡献率的净值，确保从1开始
        factor_equity_df = (1 + factor_contribution_df).cumprod()
        # 确保净值从1开始
        if not factor_equity_df.empty:
            # 直接将整个序列除以第一个元素
            factor_equity_df = factor_equity_df / factor_equity_df.iloc[0]
        
        # 保存到Excel文件
        contribution_file = f"{target_desdir}/{target}_factor_contribution.xlsx"
        with pd.ExcelWriter(contribution_file) as writer:
            factor_contribution_df.to_excel(writer, sheet_name='因子贡献率')
            factor_equity_df.to_excel(writer, sheet_name='因子贡献率净值')
        print(f"因子贡献率数据已保存到: {contribution_file}")
    
    # 生成数据可视化
    print("\n生成数据可视化...")
    
    # 绘制因子超额暴露和贡献率净值图表
    if relative_factor_exposure_dict and factor_contribution_dict:
        relative_exposure_df = pd.DataFrame(relative_factor_exposure_dict).T
        factor_contribution_df = pd.DataFrame(factor_contribution_dict).T
        factor_equity_df = (1 + factor_contribution_df).cumprod()
        # 确保净值从1开始
        if not factor_equity_df.empty:
            # 直接将整个序列除以第一个元素
            factor_equity_df = factor_equity_df / factor_equity_df.iloc[0]
        
        # 为每个因子创建图表
        common_factors = [f for f in style_factors if f in relative_exposure_df.columns and f in factor_equity_df.columns]
        for factor in common_factors:
            plt.figure(figsize=(15, 8))
            ax1 = plt.gca()
            
            # 绘制因子超额暴露（左轴，柱状图）
            bar_width = 1  # 以天为单位的宽度
            ax1.bar(relative_exposure_df.index, relative_exposure_df[factor], width=bar_width, label=f'{factor} 超额暴露', alpha=0.6)
            ax1.set_xlabel('日期')
            ax1.set_ylabel(f'{factor} 超额暴露', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_title(f'{target} {factor} 超额暴露与贡献率净值')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # 创建第二根轴（右轴）用于显示因子贡献率净值
            ax2 = ax1.twinx()
            ax2.plot(factor_equity_df.index, factor_equity_df[factor], label=f'{factor} 贡献率净值', color='tab:red', linewidth=2)
            ax2.set_ylabel(f'{factor} 贡献率净值', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # 合并图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"{target_desdir}/{target}_{factor}_exposure_contribution.png")
            plt.close()
    
    print(f"\n数据可视化已保存到对应文件")

print("\n所有指数处理完成！")