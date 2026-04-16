import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

basdir = "E:/SJTU/实习/国泰海通/barra因子/data_base"
srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/index"
bardir = "E:/SJTU/实习/国泰海通/barra因子/data_base/barra_data/全A_M"
retdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/fac_ret/whole_mkt"
desdir = "E:/SJTU/实习/国泰海通/barra因子/result/指数构成"
target_index = ['000300.XSHG','000905.XSHG', '000852.XSHG','932000.INDX','000922.XSHG'] #
ban_index = ["399006.XSHE","000680.XSHG"]
style_factors = ['size', 'non_linear_size', 'momentum', 'liquidity',
       'book_to_price', 'leverage', 'growth', 'earnings_yield', 'beta',
       'residual_volatility']
tds = pd.read_pickle(f"{basdir}/trading_dates.pkl")
# 创建目标目录
os.makedirs(desdir, exist_ok=True)

# 读取板指数据
ban_data = {}
for ban in ban_index:
    ban_file = f"{srcdir}/component/{ban}_20_26M_dict.pkl"
    with open(ban_file, 'rb') as f:
        ban_data[ban] = pickle.load(f)

# 读取行业信息数据
industry_file = f"{srcdir}/component_industry_20_26M_dict.pkl"
with open(industry_file, 'rb') as f:
    industry_data = pickle.load(f)

# 读取全A数据
a_file = f"{srcdir}/component/866011.RI_20_26M_dict.pkl"
with open(a_file, 'rb') as f:
    a_dict = pickle.load(f)

# 读取因子暴露数据
exp_file = f"{bardir}/米筐全A_20_26M_dict.pkl"
with open(exp_file, 'rb') as f:
    ex_dict = pickle.load(f)

# 紧接着计算全A的因子暴露
A_dict ={}
for dt,s in a_dict.items():
    exposure = ex_dict[dt]
    df_exposure = s.to_frame().merge(exposure, left_index=True, right_index=True)
    if len(s) != len(df_exposure):
        print(f"{dt}天的因子暴露数据缺失:",len(s)-len(df_exposure))
    daily_exp = df_exposure.iloc[:, 1:].mul(df_exposure['weight'], axis=0).sum()
    A_dict[dt] = daily_exp

# 读取因子收益率数据
df_facret = pd.read_pickle(f"{retdir}/factor_returns_20_2603.pkl")

# 处理所有目标指数
for target in target_index:
    print(f"\n=== 处理指数: {target} ===")
    
    # 为每个目标指数创建单独的文件夹
    target_desdir = f"{desdir}/{target}"
    os.makedirs(target_desdir, exist_ok=True)
    
    target_file = f"{srcdir}/component/{target}_20_26M.pkl"
    target_df = pd.read_pickle(target_file)
    
    print(f"原始数据形状: {target_df.shape}")
    print(f"索引: {target_df.index.names}")
    
    # 获取所有日期
    all_dates = target_df.index.get_level_values(0).unique()
    print(f"总天数: {len(all_dates)}")
    
    # 存储所有处理后的数据
    all_data_dict = {}
    
    # 存储分析结果
    top_industries_daily = []
    industry_changes_daily = []
    ban_weights_daily = []
    ban_weight_changes_daily = []
    
    # 存储每天的行业权重
    industry_weights_dict = {}
    
    # 存储每天的因子暴露
    factor_exposure_dict = {}
    relative_factor_exposure_dict = {}
    
    # 存储每天的因子收益率
    factor_returns_dict = {}
    
    # 处理每一天
    for i, current_date in enumerate(all_dates):
        print(f"\n处理日期: {current_date} ({i+1}/{len(all_dates)})")
        
        # 获取该日期的成分股数据
        date_df = target_df.loc[[current_date]].copy()
        
        # 添加板指列
        for ban in ban_index:
            if current_date in ban_data[ban]:
                ban_stocks = ban_data[ban][current_date]
                date_df[f'板指_{ban}'] = date_df.index.get_level_values(1).isin(ban_stocks)
            else:
                date_df[f'板指_{ban}'] = False
        
        # 合并行业信息
        str_date = current_date.strftime('%Y-%m-%d')
        if str_date in industry_data:
            industry_df = industry_data[str_date]
            # 重置索引以获取order_book_id列
            date_df_reset = date_df.reset_index()
            # 基于order_book_id合并行业信息
            date_df_reset = date_df_reset.merge(industry_df, left_on='order_book_id', right_index=True, how='left')
            # 恢复双重索引
            date_df = date_df_reset.set_index(['date', 'order_book_id'])
        
        # 存储处理后的数据
        all_data_dict[current_date] = date_df

        # 计算因子暴露
        if current_date in ex_dict:
            exposure = ex_dict[current_date]
            # 重置索引以获取order_book_id列
            date_df_reset = date_df.reset_index()
            # 基于order_book_id合并因子暴露数据
            df_exposure = date_df_reset.merge(exposure, left_on='order_book_id', right_index=True, how='left')
            
            if len(date_df_reset) != len(df_exposure):
                print(f"{current_date}天的因子暴露数据缺失: {len(date_df_reset) - len(df_exposure)}")
            
            # 计算加权因子暴露
            if 'weight' in df_exposure.columns and len(df_exposure) > 0:
                daily_exp = df_exposure[style_factors].mul(df_exposure['weight'], axis=0).sum()
                factor_exposure_dict[current_date] = daily_exp
                    
                # 计算相对于全A的暴露
                if current_date in A_dict:
                    relative_exp = daily_exp - A_dict[current_date][style_factors]
                    relative_factor_exposure_dict[current_date] = relative_exp
        
        # 计算因子暴露乘以因子收益率
        next_date = tds[tds > current_date.strftime('%Y-%m-%d')].iloc[0]
        if current_date in factor_exposure_dict and current_date in df_facret.index:
            # 获取当天的因子暴露
            daily_exp = factor_exposure_dict[current_date]
            # 获取当天的因子收益率
            daily_facret = df_facret.loc[next_date] #next_date是字符串！
            
            # 确保因子顺序对齐
            common_factors = [f for f in style_factors if f in daily_facret.index]
            if common_factors:
                # 计算因子暴露乘以因子收益率
                factor_returns = daily_exp[common_factors] * daily_facret[common_factors]
                factor_returns_dict[current_date] = factor_returns

        # 1. 计算每一天行业权重最大的三个行业以及他们的权重
        if 'first_industry_name' in date_df.columns:
            industry_weights = date_df.groupby('first_industry_name')['weight'].sum()
            industry_weights_dict[current_date] = industry_weights
            
            top_3_industries = industry_weights.nlargest(3)
            top_industries_daily.append({
                'date': current_date,
                'top1_industry': top_3_industries.index[0] if len(top_3_industries) > 0 else None,
                'top1_weight': top_3_industries.iloc[0] if len(top_3_industries) > 0 else 0,
                'top2_industry': top_3_industries.index[1] if len(top_3_industries) > 1 else None,
                'top2_weight': top_3_industries.iloc[1] if len(top_3_industries) > 1 else 0,
                'top3_industry': top_3_industries.index[2] if len(top_3_industries) > 2 else None,
                'top3_weight': top_3_industries.iloc[2] if len(top_3_industries) > 2 else 0
            })
        
        # 2. 计算每天板指的权重
        for ban in ban_index:
            ban_weight = date_df[date_df[f'板指_{ban}']]['weight'].sum()
            ban_weights_daily.append({
                'date': current_date,
                'ban': ban,
                'weight': ban_weight
            })
    
    # 3. 计算每天行业权重变化最大的三个行业及他们的变化权重
    if industry_weights_dict:
        # 构建行业权重时间序列
        industry_weights_over_time = pd.DataFrame(industry_weights_dict).T.fillna(0)
        industry_weight_changes = industry_weights_over_time.diff().fillna(0)
        
        for current_date in all_dates:
            if current_date in industry_weight_changes.index:
                daily_changes = industry_weight_changes.loc[current_date]
                top_3_changes = daily_changes.abs().nlargest(3)
                industry_changes_daily.append({
                    'date': current_date,
                    'top1_industry_change': top_3_changes.index[0] if len(top_3_changes) > 0 else None,
                    'top1_change_value': daily_changes.loc[top_3_changes.index[0]] if len(top_3_changes) > 0 else 0,
                    'top2_industry_change': top_3_changes.index[1] if len(top_3_changes) > 1 else None,
                    'top2_change_value': daily_changes.loc[top_3_changes.index[1]] if len(top_3_changes) > 1 else 0,
                    'top3_industry_change': top_3_changes.index[2] if len(top_3_changes) > 2 else None,
                    'top3_change_value': daily_changes.loc[top_3_changes.index[2]] if len(top_3_changes) > 2 else 0
                })
    
    # 4. 计算每天板指的权重变化量
    ban_weights_df = pd.DataFrame(ban_weights_daily)
    ban_weights_pivot = ban_weights_df.pivot(index='date', columns='ban', values='weight').fillna(0)
    ban_weight_changes = ban_weights_pivot.diff().fillna(0)
    
    for current_date in ban_weight_changes.index:
        for ban in ban_index:
            ban_weight_change = ban_weight_changes.loc[current_date, ban]
            ban_weight_changes_daily.append({
                'date': current_date,
                'ban': ban,
                'weight_change': ban_weight_change
            })
    
    # 保存处理后的数据（按日期保存）
    pd.to_pickle(all_data_dict, f"{target_desdir}/{target}_ind_ban_20_26M.pkl")
    print(f"\n字典数据已保存到: {target_desdir}/{target}_ind_ban_20_26M.pkl")
    
    # 保存分析结果
    top_industries_df = pd.DataFrame(top_industries_daily)
    top_industries_df.to_excel(f"{target_desdir}/{target}_top_industries.xlsx", index=False)
    
    industry_changes_df = pd.DataFrame(industry_changes_daily)
    industry_changes_df.to_excel(f"{target_desdir}/{target}_industry_changes.xlsx", index=False)
    
    ban_weights_df.to_excel(f"{target_desdir}/{target}_ban_weights.xlsx", index=False)
    
    ban_weight_changes_df = pd.DataFrame(ban_weight_changes_daily)
    ban_weight_changes_df.to_excel(f"{target_desdir}/{target}_ban_weight_changes.xlsx", index=False)
    
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
    
    # 保存因子收益率数据
    if factor_returns_dict:
        factor_returns_df = pd.DataFrame(factor_returns_dict).T
        
        # 保存到Excel文件
        factor_file = f"{target_desdir}/{target}_factor_returns.xlsx"
        with pd.ExcelWriter(factor_file) as writer:
            factor_returns_df.to_excel(writer, sheet_name='因子贡献收益率')
        print(f"\n因子收益率数据已保存到: {factor_file}")
    
    print(f"\n分析结果已保存到对应文件")
    
    # 数据可视化
    print("\n生成数据可视化...")
    
    # 1. 行业权重趋势图
    if industry_weights_dict:
        industry_weights_over_time = pd.DataFrame(industry_weights_dict).T.fillna(0)
        # 选择权重较大的前10个行业
        top_industries = industry_weights_over_time.sum().nlargest(10).index
        top_industry_weights = industry_weights_over_time[top_industries]
        
        plt.figure(figsize=(15, 8))
        for industry in top_industry_weights.columns:
            plt.plot(top_industry_weights.index, top_industry_weights[industry], label=industry)
        plt.title(f'{target} 行业权重趋势（前10大行业）')
        plt.xlabel('日期')
        plt.ylabel('权重')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{target_desdir}/{target}_industry_trend.png")
        plt.close()
    
    # 2. 行业权重变化图（每天前三大变化）
    if industry_changes_daily:
        # 从已计算的industry_changes_daily中提取数据
        daily_top_changes = []
        for item in industry_changes_daily:
            # 提取top1行业变化
            if item['top1_industry_change']:
                daily_top_changes.append({
                    'date': item['date'],
                    'industry': item['top1_industry_change'],
                    'change': item['top1_change_value'],
                    'rank': 1
                })
            # 提取top2行业变化
            if item['top2_industry_change']:
                daily_top_changes.append({
                    'date': item['date'],
                    'industry': item['top2_industry_change'],
                    'change': item['top2_change_value'],
                    'rank': 2
                })
            # 提取top3行业变化
            if item['top3_industry_change']:
                daily_top_changes.append({
                    'date': item['date'],
                    'industry': item['top3_industry_change'],
                    'change': item['top3_change_value'],
                    'rank': 3
                })
        
        # 转换为DataFrame
        top_changes_df = pd.DataFrame(daily_top_changes)
        
        # 为每个行业分配颜色
        unique_industries = top_changes_df['industry'].unique()
        color_map = plt.cm.get_cmap('tab20', len(unique_industries))
        industry_colors = {industry: color_map(i) for i, industry in enumerate(unique_industries)}
        
        # 创建图表，调整大小以适应更多数据
        plt.figure(figsize=(20, 10))
        
        # 计算每天的位置
        all_dates = sorted(top_changes_df['date'].unique())
        date_positions = {date: i for i, date in enumerate(all_dates)}
        
        # 绘制每天权重变化最大的三个行业（柱状图）
        bar_width = 0.25
        # 用于跟踪已经添加到图例的行业
        legend_added = set()
        
        for date in all_dates:
            date_changes = top_changes_df[top_changes_df['date'] == date]
            if not date_changes.empty:
                for i, (_, row) in enumerate(date_changes.iterrows()):
                    # 计算位置偏移，避免柱子重叠
                    pos = date_positions[date] + (i - 1) * bar_width
                    # 只有当行业还没有添加到图例时才添加标签
                    label = row['industry'] if row['industry'] not in legend_added else ""
                    if label:
                        legend_added.add(row['industry'])
                    
                    plt.bar(
                        pos,
                        row['change'],
                        width=bar_width,
                        color=industry_colors[row['industry']],
                        label=label
                    )
        
        # 设置x轴标签
        plt.xticks([i for i in range(len(all_dates))], [date.strftime('%Y-%m-%d') for date in all_dates], rotation=45, ha='right')
        plt.xlabel('日期')
        plt.ylabel('权重变化')
        plt.title(f'{target} 行业权重变化（每天前三大变化）')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{target_desdir}/{target}_industry_changes.png")
        plt.close()
    
    # 2. 板指权重趋势图
    plt.figure(figsize=(15, 6))
    for ban in ban_index:
        if ban in ban_weights_pivot.columns:
            ban_data_series = ban_weights_pivot[ban]
            plt.plot(ban_data_series.index, ban_data_series.values, label=ban)
    plt.title(f'{target} 板指权重趋势')
    plt.xlabel('日期')
    plt.ylabel('权重')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{target_desdir}/{target}_ban_trend.png")
    plt.close()
    
    # 3. 板指权重变化图
    plt.figure(figsize=(15, 6))
    for ban in ban_index:
        if ban in ban_weight_changes.columns:
            ban_change_data = ban_weight_changes[ban]
            plt.plot(ban_change_data.index, ban_change_data.values, label=ban)
    plt.title(f'{target} 板指权重变化趋势')
    plt.xlabel('日期')
    plt.ylabel('权重变化')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{target_desdir}/{target}_ban_change_trend.png")
    plt.close()
    
    # 4. 因子暴露和收益率图表
    if factor_exposure_dict and factor_returns_dict:
        factor_exposure_df = pd.DataFrame(factor_exposure_dict).T
        factor_returns_df = pd.DataFrame(factor_returns_dict).T
        
        # 为每个因子创建图表
        common_factors = [f for f in style_factors if f in factor_exposure_df.columns and f in factor_returns_df.columns]
        for factor in common_factors:
            plt.figure(figsize=(15, 8))
            ax1 = plt.gca()
            
            # 绘制因子暴露（左轴，柱状图）
            # 增加柱子宽度
            bar_width = 15  # 以天为单位的宽度
            ax1.bar(factor_exposure_df.index, factor_exposure_df[factor], width=bar_width, label=f'{factor} 暴露', alpha=0.6)
            ax1.set_xlabel('日期')
            ax1.set_ylabel(f'{factor} 暴露', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_title(f'{target} {factor} 暴露与收益率')
            ax1.grid(True, alpha=0.3)
            
            # 创建第二根轴（右轴）用于显示因子收益率
            ax2 = ax1.twinx()
            ax2.scatter(factor_returns_df.index, factor_returns_df[factor], label=f'{factor} 收益率', color='tab:red', s=30)
            # 添加y=0的水平线
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel(f'{factor} 收益率', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # 合并图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"{target_desdir}/{target}_{factor}_exposure_returns.png")
            plt.close()
    
    print(f"\n数据可视化已保存到对应文件")

print("\n所有指数处理完成！")
