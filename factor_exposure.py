import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据路径
FUND_NAV_PATH = 'E:/SJTU/实习/国泰海通/barra因子/data_base/fund_nav/中证500净值数据.xlsx'
FAC_RET_PATH = 'E:/SJTU/实习/国泰海通/barra因子/data_base/fac_ret/中证500/factor_returns_07_2604.pkl'
desdir = 'E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露'
FREQ = 'W'  # 可以设置为 'W' 或 'D'

# 创建结果目录
if not os.path.exists(desdir):
    os.makedirs(desdir)
    print(f"Created directory: {desdir}")

# 读取私募产品净值数据
def read_fund_nav():
    print("读取私募产品净值数据...")
    df = pd.read_excel(FUND_NAV_PATH, index_col=0)
    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df.index.min()} 到 {df.index.max()}")
    print(f"产品数量: {len(df.columns)}")
    return df

# 读取因子收益率数据
def read_factor_returns(start_date, end_date):
    print("读取因子收益率数据...")
    if os.path.exists(FAC_RET_PATH):
        with open(FAC_RET_PATH, 'rb') as f:
            factor_df = pickle.load(f)
        print(f"因子数据形状: {factor_df.shape}")
        print(f"因子数量: {len(factor_df.columns)}")
        
        # 筛选日期范围
        factor_df = factor_df.loc[(factor_df.index >= start_date) & (factor_df.index <= end_date)]
        print(f"筛选后因子数据形状: {factor_df.shape}")
        
        # 根据 FREQ 参数处理因子收益率
        if FREQ == 'W':
            # 读取基金净值数据以获取周频日期
            fund_nav = read_fund_nav()
            
            # 直接使用基金净值的日期作为周频日期
            weekly_dates = fund_nav.index
            
            # 计算因子每周的累计收益率
            weekly_factor_returns = []
            for i in range(1, len(weekly_dates)):
                week_start = weekly_dates[i-1]
                week_end = weekly_dates[i]
                
                # 获取该周的日因子收益率
                week_factor = factor_df.loc[(factor_df.index > week_start) & (factor_df.index <= week_end)]
                
                if not week_factor.empty:
                    # 计算累计收益率
                    cum_return = (1 + week_factor).prod() - 1
                    weekly_factor_returns.append(cum_return)
            
            # 构建每周因子收益率 DataFrame
            if weekly_factor_returns:
                factor_df = pd.DataFrame(weekly_factor_returns, index=weekly_dates[1:])
                print(f"转换为周频后因子数据形状: {factor_df.shape}")
                print(f"周频因子收益率日期范围: {factor_df.index.min()} 到 {factor_df.index.max()}")
        else:
            # FREQ == 'D'，保持不变
            print("使用日频因子收益率")
        
        return factor_df
    else:
        print(f"因子数据文件不存在: {FAC_RET_PATH}")
        return None

# 计算因子暴露（使用EWMA时间加权回归 + 主成分分析）
def calculate_factor_exposure(fund_returns, factor_returns, window=12, half_life=3, n_components=0.9, normalize_industry=False):
    print("计算因子暴露...")
    exposure_dict = {}
    stats_dict = {}
    
    # 确保日期对齐
    common_dates = fund_returns.index.intersection(factor_returns.index)
    fund_returns = fund_returns.loc[common_dates]
    factor_matrix = factor_returns.loc[common_dates]
    
    # 区分风格因子和行业因子
    style_factors = [col for col in factor_matrix.columns if all(ord(c) < 128 for c in col)]  # 英文列名
    industry_factors = [col for col in factor_matrix.columns if any(ord(c) >= 128 for c in col)]  # 中文列名
    
    print(f"风格因子数量: {len(style_factors)}")
    print(f"行业因子数量: {len(industry_factors)}")
    print(f"对齐后数据形状 - 基金收益: {fund_returns.shape}, 因子矩阵: {factor_matrix.shape}")
    print(f"当前频率: {'周频' if FREQ == 'W' else '日频'}")
    print(f"使用的窗口大小: {window}, 半衰期: {half_life}")
    print(f"主成分分析保留方差比例: {n_components}")
    print(f"是否对行业暴露归一化: {normalize_industry}")
    
    for fund in fund_returns.columns:
        fund_data = fund_returns[fund].dropna()
        exposures = []
        r_squared_list = []
        industry_sum_list = []
        num_negative_list = []
        
        for i in range(window, len(fund_data)):
            # 滚动窗口数据
            window_data = fund_data.iloc[i-window:i]
            window_factors = factor_matrix.loc[window_data.index]
            
            if len(window_factors) == window and not window_factors.isnull().any().any():
                # 计算EWMA权重
                weights = np.exp(-np.log(2) / half_life * np.arange(window-1, -1, -1))
                #weights = np.ones(window)
                weights = weights / weights.sum()
                
                # 对所有因子一起做分析
                X = window_factors.values
                y = window_data.values
                
                # 对所有因子进行主成分分析
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)
                print(f"所有因子PCA后生成了{X_pca.shape[1]}个主成分因子")

                # 回归分析
                X_with_intercept = np.hstack([np.ones((window, 1)), X_pca])
                weighted_X = X_with_intercept * weights[:, np.newaxis]
                weighted_y = y * weights
                
                r_squared = None
                industry_sum = None
                num_negative = None
                
                try:
                    # 对主成分回归
                    full_beta = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)[0]
                    intercept = full_beta[0]  # 截距项系数
                    beta_pca = full_beta[1:]  # 主成分系数
                    # 将主成分回归系数还原回原始因子空间
                    beta = beta_pca @ pca.components_
                    
                    # 计算PCA回归的R方
                    # 计算预测值
                    y_pred = weighted_X @ full_beta
                    # 计算总平方和 (TSS)
                    tss = np.sum((weighted_y - np.mean(weighted_y))**2)
                    # 计算残差平方和 (RSS)
                    rss = np.sum((weighted_y - y_pred)**2)
                    # 计算R方
                    r_squared = 1 - (rss / tss)
                    print(f"PCA回归R方: {r_squared:.4f}, 截距项: {intercept:.4f}")
                    
                    # # 为前5个基金的第一个窗口绘制拟合图
                    # if fund in fund_returns.columns[:5] and i == window:
                    #     import matplotlib.pyplot as plt
                    #     # 创建图表
                    #     plt.figure(figsize=(10, 6))
                    #     # 绘制加权收益率
                    #     plt.plot(range(len(weighted_y)), weighted_y, 'b-', label='Weighted Y')
                    #     # 绘制预测收益率
                    #     plt.plot(range(len(y_pred)), y_pred, 'r--', label='Predicted Y')
                    #     # 添加标题和标签
                    #     plt.title(f'{fund} - Weighted Y vs Predicted Y (R²={r_squared:.4f})')
                    #     plt.xlabel('Time')
                    #     plt.ylabel('Return')
                    #     plt.legend()
                    #     plt.grid(True)
                    #     # 保存图表
                    #     plot_dir = f"{desdir}/plots"
                    #     if not os.path.exists(plot_dir):
                    #         os.makedirs(plot_dir)
                    #     plot_path = f"{plot_dir}/{fund}_fit_plot.png"
                    #     plt.savefig(plot_path)
                    #     plt.close()
                    #     print(f"拟合图已保存到: {plot_path}")
                    
                    # 如果需要对行业暴露归一化
                    if normalize_industry and industry_factors:
                        # 分离行业因子暴露
                        industry_beta = beta[len(style_factors):]
                        # 确保非负
                        industry_beta = np.maximum(0, industry_beta)
                        # 归一化处理
                        beta_sum = industry_beta.sum()
                        if beta_sum > 0:
                            industry_beta = industry_beta / beta_sum  # 归一化，确保和为1
                        else:
                            industry_beta = np.zeros(len(industry_factors))  # 避免除以0
                        # 替换回原始beta
                        beta[len(style_factors):] = industry_beta
                    elif not normalize_industry and industry_factors:
                        # 计算行业暴露之和
                        industry_beta = beta[len(style_factors):]
                        industry_sum = industry_beta.sum()
                        # 检查是否含有负数
                        num_negative = sum(b < 0 for b in industry_beta)
                        print(f"行业暴露之和: {industry_sum:.4f}, 是否含有负数: {num_negative}")
                except Exception as e:
                    beta = np.zeros(factor_matrix.shape[1])
                    print(f"错误: {e}")
                
                exposures.append(beta)
                r_squared_list.append(r_squared)
                industry_sum_list.append(industry_sum)
                num_negative_list.append(num_negative)
            else:
                # 数据不足时填充0
                exposures.append(np.zeros(factor_matrix.shape[1]))
                r_squared_list.append(None)
                industry_sum_list.append(None)
                num_negative_list.append(None)
        
        # 构建暴露DataFrame
        exposure_df = pd.DataFrame(
            exposures, 
            index=fund_data.index[window:],
            columns=factor_matrix.columns
        )
        exposure_df.to_excel(f"{desdir}/exposure/{fund}_exposure.xlsx", index=True)
        exposure_dict[fund] = exposure_df
        
        # 构建统计信息DataFrame
        stats_df = pd.DataFrame({
            'r_squared': r_squared_list,
            'industry_sum': industry_sum_list,
            'num_negative': num_negative_list
        }, index=fund_data.index[window:])
        stats_dict[fund] = stats_df
    
    print(f"完成因子暴露计算，共 {len(exposure_dict)} 个产品")
    return exposure_dict, stats_dict

# 绘制每个基金每个因子的暴露和收益图
def plot_factor_exposure_returns(exposure_dict, factor_returns):
    print("绘制每个基金每个风格因子的暴露和收益图...")
    
    # 创建结果目录
    plot_dir = f"{desdir}/exposure_with_returns_plot"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # 遍历每个基金
    for fund, exposure_df in exposure_dict.items():
        # 区分风格因子（英文列名）
        style_factors = [col for col in exposure_df.columns if all(ord(c) < 128 for c in col)]
        
        # 只遍历风格因子
        for factor in style_factors:
            # 确保因子在因子收益率数据中存在
            if factor in factor_returns.columns:
                # 确保日期对齐
                common_dates = exposure_df.index.intersection(factor_returns.index)
                if len(common_dates) > 0:
                    # 获取对应日期的数据
                    fund_exposure = exposure_df.loc[common_dates, factor]
                    factor_ret = factor_returns.loc[common_dates, factor]
                    
                    # 计算因子暴露*因子收益率
                    factor_contribution = fund_exposure * factor_ret
                    
                    # 计算累计收益（净值曲线），从1开始
                    factor_ret_cum = (1 + factor_ret).cumprod()
                    if not factor_ret_cum.empty:
                        factor_ret_cum = factor_ret_cum / factor_ret_cum.iloc[0]
                    
                    factor_contribution_cum = (1 + factor_contribution).cumprod()
                    if not factor_contribution_cum.empty:
                        factor_contribution_cum = factor_contribution_cum / factor_contribution_cum.iloc[0]
                    
                    # 创建双轴图表
                    fig, ax1 = plt.subplots(figsize=(15, 10))
                    
                    # 绘制因子暴露（柱状图）
                    ax1.bar(fund_exposure.index, fund_exposure.values, alpha=0.7, label=f'{factor} Exposure')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel(f'{factor} Exposure', color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    ax1.set_xticks(fund_exposure.index[::max(1, len(fund_exposure)//10)])  # 避免x轴标签过于密集
                    ax1.set_xticklabels(fund_exposure.index[::max(1, len(fund_exposure)//10)].strftime('%Y-%m-%d'), rotation=45)
                    
                    # 创建第二个y轴用于累计收益
                    ax2 = ax1.twinx()
                    # 绘制因子收益率累计净值
                    ax2.plot(factor_ret_cum.index, factor_ret_cum.values, 'r-', linewidth=2, label=f'{factor} Return Cumulative')
                    # 绘制因子贡献累计净值
                    ax2.plot(factor_contribution_cum.index, factor_contribution_cum.values, 'g-', linewidth=2, label=f'{factor} Contribution Cumulative')
                    ax2.set_ylabel('Cumulative Return (Net Value)', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    
                    # 合并图例
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    # 添加标题
                    plt.title(f'{fund} - {factor} Exposure and Returns')
                    plt.grid(True)
                    plt.tight_layout()
                    
                                    # 保存图表
                    # 为每个基金创建单独的文件夹
                    fund_dir = f"{plot_dir}/{fund}"
                    if not os.path.exists(fund_dir):
                        os.makedirs(fund_dir)
                    # 确保文件名合法
                    safe_factor_name = factor.replace('/', '_').replace('\\', '_')
                    plot_path = f"{fund_dir}/{safe_factor_name}_exposure_with_returns.png"
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"风格因子暴露和收益图已保存到: {plot_path}")

# 计算主动暴露因子
def calculate_active_exposure(exposure_dict, fund_returns):
    print("计算主动暴露因子...")
    active_exposure_dict = {}
    
    # 获取所有基金名称
    fund_names = list(exposure_dict.keys())
    
    # 第一个fund是指数
    index_fund = fund_names[0]
    index_exposure = exposure_dict[index_fund]
    print(f"识别指数产品: {index_fund}")
    
    for fund, exposure_df in exposure_dict.items():
        # 区分风格因子和行业因子
        style_factors = [col for col in exposure_df.columns if all(ord(c) < 128 for c in col)]  # 英文列名
        industry_factors = [col for col in exposure_df.columns if any(ord(c) >= 128 for c in col)]  # 中文列名
        
        # 计算相对于指数的因子暴露
        if fund == index_fund:
            # 指数本身的暴露
            relative_exposure = exposure_df
        else:
            # 私募产品相对于指数的暴露
            common_dates = exposure_df.index.intersection(index_exposure.index)
            if len(common_dates) > 0:
                relative_exposure = exposure_df.loc[common_dates] - index_exposure.loc[common_dates]
            else:
                relative_exposure = pd.DataFrame()
        
        # 计算每日风格和行业暴露绝对值之和（基于相对暴露）
        daily_style_exposure = relative_exposure[style_factors].abs().sum(axis=1) if style_factors else None
        daily_industry_exposure = relative_exposure[industry_factors].abs().sum(axis=1) if industry_factors else None
        
        # 计算净值曲线（从1开始累计）
        fund_ret = fund_returns[fund]
        common_dates = relative_exposure.index.intersection(fund_ret.index)
        if len(common_dates) > 0:
            # 计算累计净值，确保第一天为1
            fund_ret_subset = fund_ret.loc[common_dates]
            # 创建一个长度为len(common_dates)的数组，第一天为1，后续为累计乘积
            net_value = pd.Series(index=common_dates)
            net_value.iloc[0] = 1.0  # 第一天净值为1
            if len(common_dates) > 1:
                # 从第二天开始计算累计净值
                net_value.iloc[1:] = (1 + fund_ret_subset.iloc[1:]).cumprod()
            
            # 绘制净值与暴露的关系图
            plot_net_value_exposure(net_value, daily_style_exposure, daily_industry_exposure, fund)
        else:
            net_value = pd.Series(dtype='float64')
        
        # 构建用于保存的DataFrame
        base_df = pd.DataFrame({
            'daily_style_exposure': daily_style_exposure,    # 一维
            'daily_industry_exposure': daily_industry_exposure,  # 一维
            'net_value': net_value,                          # 一维
        }, index=net_value.index)
        
        # 构建结果字典，包含原始的relative_exposure DataFrame
        active_data = {
            'exposure': relative_exposure,  # 原始的相对暴露DataFrame
            'daily_style_exposure': daily_style_exposure,    # 一维
            'daily_industry_exposure': daily_industry_exposure,  # 一维
            'net_value': net_value,                          # 一维
        }
        
        # 添加相对暴露的列到保存的DataFrame
        if not relative_exposure.empty:
            for col in relative_exposure.columns:
                base_df[col] = relative_exposure[col]
        
        # 保存到Excel
        base_df.to_excel(f"{desdir}/excess_exposure/{fund}_relative_exposure.xlsx", index=True)
        
        # 保存到字典
        active_exposure_dict[fund] = active_data
    
    print("完成主动暴露因子计算")
    return active_exposure_dict

# 绘制净值与暴露的关系图
def plot_net_value_exposure(net_value, daily_style_exposure, daily_industry_exposure, fund):
    # 确保net_value不为空
    if len(net_value) == 0:
        print(f"No data to plot for {fund}")
        return
    
    plt.figure(figsize=(15, 8))
    
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 绘制净值曲线
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Value', color='tab:blue')
    ax1.plot(net_value.index, net_value.values, label='Net Value', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # 创建第二个y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Exposure (Abs)', color='tab:red')
    
    # 绘制风格暴露
    if daily_style_exposure is not None:
        # 确保使用相同的日期索引
        style_exposure = daily_style_exposure.reindex(net_value.index)
        # 只绘制非空值
        valid_mask = style_exposure.notna()
        if valid_mask.any():
            ax2.plot(style_exposure.index[valid_mask], style_exposure[valid_mask], label='Style Exposure', color='tab:red', alpha=0.7)
    
    # 绘制行业暴露
    if daily_industry_exposure is not None:
        # 确保使用相同的日期索引
        industry_exposure = daily_industry_exposure.reindex(net_value.index)
        # 只绘制非空值
        valid_mask = industry_exposure.notna()
        if valid_mask.any():
            ax2.plot(industry_exposure.index[valid_mask], industry_exposure[valid_mask], label='Industry Exposure', color='tab:green', alpha=0.7)
    
    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(f'{fund} - Net Value vs Exposure')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{desdir}/exposure_nav/{fund}_net_value_exposure.png')
    plt.close()
    print(f"Net value vs exposure plot saved as {fund}_net_value_exposure.png")

# 计算滚动相关性并绘图
def calculate_rolling_correlation(exposure_dict, fund_returns, windows=[24, 52, 104]): #周频
    print("计算滚动相关性...")
    
    for fund, active_data in exposure_dict.items():
        exposure_df = active_data['exposure'] #relative_exposure
        # 区分风格因子和行业因子
        style_factors = [col for col in exposure_df.columns if all(ord(c) < 128 for c in col)]  # 英文列名
        
        if style_factors:
            # 获取对应时期的收益率
            common_dates = exposure_df.index.intersection(fund_returns[fund].index)
            fund_ret = fund_returns[fund].loc[common_dates]
            exposure_subset = exposure_df.loc[common_dates]
            
            # 计算每个风格因子与收益率的滚动相关性
            correlation_dict = {}
            for window in windows:
                window_corrs = {}
                for factor in style_factors:
                    # 使用原始暴露计算相关性（不取绝对值）
                    corr = fund_ret.rolling(window=window).corr(exposure_subset[factor])
                    window_corrs[factor] = corr
                correlation_dict[f'window_{window}'] = window_corrs
            
            # 绘制相关性图表
            plot_rolling_correlation(correlation_dict, fund)
    
    print("完成滚动相关性计算和绘图")

# 绘制滚动相关性图表
def plot_rolling_correlation(correlation_dict, fund):
    plt.figure(figsize=(15, 10))
    
    # 为每个窗口创建子图
    windows = list(correlation_dict.keys())
    n_windows = len(windows)
    
    for i, window in enumerate(windows, 1):
        plt.subplot(n_windows, 1, i)
        
        # 绘制每个风格因子的相关性
        for factor, corr_series in correlation_dict[window].items():
            plt.plot(corr_series.index, corr_series.values, label=factor)
        
        plt.title(f'{fund} - Rolling Correlation (Window: {window.split("_")[1]} {FREQ})')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend(loc='upper left')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{desdir}/rolling_corr/{fund}_rolling_correlation.png')
    plt.close()
    print(f"Rolling correlation plot saved as {fund}_rolling_correlation.png")

# 绘制风格因子收益和pure alpha贡献图
def plot_style_alpha_contribution(rank_df, style_cols, fund):
    import matplotlib.pyplot as plt
    
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(15, 10))
    
    # 计算风格因子累计收益（净值曲线）
    style_cum_data = (1 + rank_df[style_cols]).cumprod()
    # 确保从1开始
    if not style_cum_data.empty:
        style_cum_data = style_cum_data / style_cum_data.iloc[0]
    # 绘制风格因子累计收益（折线图）
    for col in style_cols:
        ax1.plot(style_cum_data.index, style_cum_data[col], label=col, alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Style Factor Cumulative Return (Net Value)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 创建第二个y轴用于pure alpha
    ax2 = ax1.twinx()
    # 计算pure alpha累计收益（净值曲线）
    pure_alpha_cum = (1 + rank_df['pure_alpha']).cumprod()
    # 确保从1开始
    if not pure_alpha_cum.empty:
        pure_alpha_cum = pure_alpha_cum / pure_alpha_cum.iloc[0]
    # 绘制pure alpha累计收益（线图）
    ax2.plot(rank_df.index, pure_alpha_cum, 'r--', linewidth=2, label='Pure Alpha Cumulative Return')
    ax2.set_ylabel('Pure Alpha Cumulative Return (Net Value)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 添加标题
    plt.title(f'{fund} - Style Factor and Pure Alpha Cumulative Returns')
    plt.grid(True)
    
    # 保存图表
    plot_dir = f"{desdir}/decompose_nav_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = f"{plot_dir}/{fund}_style_alpha.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"风格因子收益和pure alpha图已保存到: {plot_path}")

# 计算pure alpha并分析排位
def calculate_pure_alpha(fund_returns, exposure_dict, factor_returns):
    print("计算pure alpha并分析排位...")
    
    # 创建结果目录
    alpha_dir = f"{desdir}/pure_alpha"
    if not os.path.exists(alpha_dir):
        os.makedirs(alpha_dir)
    
    # 获取所有基金名称
    fund_names = list(exposure_dict.keys())
    
    # 计算每个基金的pure alpha
    alpha_dict = {}
    for fund in fund_names:
        exposure_df = exposure_dict[fund]
        # 确保日期对齐
        common_dates = exposure_df.index.intersection(fund_returns[fund].index).intersection(factor_returns.index)
        
        if len(common_dates) > 0:
            # 获取对应日期的数据
            fund_ret = fund_returns[fund].loc[common_dates]
            exposure = exposure_df.loc[common_dates]
            factor_ret = factor_returns.loc[common_dates]
            
            # 计算纯alpha
            # 确保因子暴露和因子收益率的列顺序一致
            factor_cols = [col for col in exposure.columns if col in factor_ret.columns]
            print(f"计算pure alpha用到的因子个数：{len(factor_cols)}")
            exposure = exposure[factor_cols]
            factor_ret = factor_ret[factor_cols]
            
            # 区分风格因子（剔除comovement列）
            style_factors = [col for col in factor_cols if all(ord(c) < 128 for c in col) ] # and col != 'comovement'
            
            # 计算风格因子收益
            style_factor_returns = {}
            for factor in style_factors:
                style_factor_returns[factor] = exposure[factor] * factor_ret[factor]
            
            # 计算因子贡献
            factor_contribution = (exposure * factor_ret).sum(axis=1)
            # 计算pure alpha
            pure_alpha = fund_ret - factor_contribution
            
            # 存储结果
            alpha_dict[fund] = pure_alpha
            # 存储风格因子收益
            alpha_dict[f'{fund}_style'] = style_factor_returns
    
    # 计算排位
    print("计算排位...")
    rank_dict = {}
    for fund in fund_names:
        if fund in alpha_dict:
            pure_alpha = alpha_dict[fund]
            ranks = []
            self_ranks = []
            
            for date in pure_alpha.index:
                # 计算该日期所有基金的pure alpha
                date_alphas = []
                for f in fund_names:
                    if f in alpha_dict and date in alpha_dict[f].index:
                        date_alphas.append(alpha_dict[f][date])
                
                # 计算在所有基金中的排位
                if date_alphas:
                    current_alpha = pure_alpha[date]
                    rank = sum(1 for a in date_alphas if a < current_alpha) / len(date_alphas) * 100
                    ranks.append(rank)
                else:
                    ranks.append(None)
                
                # 计算在自己历史中的排位
                history_alpha = pure_alpha.loc[:date]
                if len(history_alpha) > 0:
                    current_alpha = pure_alpha[date]
                    self_rank = sum(1 for a in history_alpha if a < current_alpha) / len(history_alpha) * 100
                    self_ranks.append(self_rank)
                else:
                    self_ranks.append(None)
            
            # 构建排位DataFrame
            rank_data = {
                'pure_alpha': pure_alpha,
                'rank_all': ranks,
                'rank_self': self_ranks
            }
            
            # 添加风格因子收益列
            style_returns = alpha_dict.get(f'{fund}_style', {})
            for factor, returns in style_returns.items():
                rank_data[factor] = returns
            
            rank_df = pd.DataFrame(rank_data, index=pure_alpha.index)
            rank_dict[fund] = rank_df
    
    # 保存结果
    print("保存结果...")
    # 创建汇总DataFrame
    all_alpha_data = []
    for fund, rank_df in rank_dict.items():
        fund_data = rank_df.copy()
        fund_data['fund'] = fund
        all_alpha_data.append(fund_data)
    
    if all_alpha_data:
        combined_alpha = pd.concat(all_alpha_data)
        # 重新排列列顺序，确保fund、pure_alpha、rank_all、rank_self在前
        cols = ['fund', 'pure_alpha', 'rank_all', 'rank_self']
        # 添加所有风格因子列
        style_cols = [col for col in combined_alpha.columns if col not in cols]
        combined_alpha = combined_alpha[cols + style_cols]
        # 输出到Excel
        alpha_file = f"{alpha_dir}/pure_alpha_ranking.xlsx"
        combined_alpha.to_excel(alpha_file, index=True)
        print(f"pure alpha及排位信息已输出到: {alpha_file}")
    
    print("完成pure alpha计算和排位分析")
    return alpha_dict, rank_dict


# 主函数
def main():
    # 1. 读取数据
    fund_nav = read_fund_nav()
    
    # 计算日收益率
    fund_returns = fund_nav.pct_change()#.dropna()
    print(f"收益率数据形状: {fund_returns.shape}")
    
    # 2. 读取因子数据
    start_date = fund_returns.index.min()
    end_date = fund_returns.index.max()
    factor_returns = read_factor_returns(start_date, end_date)
    
    if factor_returns is not None:
        # 3. 计算因子暴露
        exposure_dict, stats_dict = calculate_factor_exposure(fund_returns, factor_returns)
        
        # 3.5. 绘制每个基金每个因子的暴露和收益图
        #plot_factor_exposure_returns(exposure_dict, factor_returns)
        
        # # 4. 计算主动暴露因子
        # active_exposure_dict = calculate_active_exposure(exposure_dict, fund_returns)
        
        # # 5. 计算滚动相关性并绘图
        # calculate_rolling_correlation(active_exposure_dict, fund_returns)
        
        # 6. 汇总统计信息并输出到Excel
        print("汇总统计信息并输出到Excel...")
        # 创建汇总统计信息的目录
        stats_dir = f"{desdir}/stats"
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        
        # 汇总所有基金的统计信息
        all_stats = []
        for fund, stats_df in stats_dict.items():
            # 添加基金名称列
            fund_stats = stats_df.copy()
            fund_stats['fund'] = fund
            all_stats.append(fund_stats)
        
        # 合并所有统计信息
        if all_stats:
            combined_stats = pd.concat(all_stats)
            # 重新排列列顺序
            combined_stats = combined_stats[['fund', 'r_squared', 'industry_sum', 'num_negative']]
            # 输出到Excel
            stats_file = f"{stats_dir}/all_fund_stats.xlsx"
            combined_stats.to_excel(stats_file, index=True)
            print(f"统计信息已输出到: {stats_file}")
        
        # 7. 计算pure alpha并分析排位
        alpha_dict, rank_dict = calculate_pure_alpha(fund_returns, exposure_dict, factor_returns)
        
        # 8. 为所有基金绘制风格因子收益和pure alpha图
        print("为所有基金绘制风格因子收益和pure alpha图...")
        for fund in rank_dict:
            rank_df = rank_dict[fund]
            # 提取风格因子列
            style_cols = [col for col in rank_df.columns if col not in ['fund', 'pure_alpha', 'rank_all', 'rank_self']] #"comovement"
            if style_cols:
                plot_style_alpha_contribution(rank_df, style_cols, fund)
        
        

    else:
        print("无法进行分析，因子数据缺失")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
