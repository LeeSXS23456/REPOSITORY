import pandas as pd
import numpy as np
import re

def is_chinese(text):
    """判断字符串是否包含中文字符"""
    return bool(re.search('[\u4e00-\u9fff]', text))

def calculate_port_allocation(df, w_col, factor_cols):
    """计算组合在指定因子上的暴露"""
    return df[w_col] @ df[factor_cols]

def calculate_effective_stocks(weights):
    """计算有效持仓数 (Herfindahl-Hirschman Index的倒数)"""
    weights = weights.dropna()
    return 1 / (weights ** 2).sum()

def calculate_top_n_weight_sum(weights, n):
    """计算前N大权重之和"""
    return weights.abs().nlargest(n).sum()

def analyze_portfolio(df_copy, df, w_col, barra_cols, output_path=None):
    """
    综合分析组合持仓数据，返回分析结果并可选保存到文件
    
    参数:
        df: 持仓信息DataFrame
        w_col: 权重列名
        barra_cols: Barra因子列表
        output_path: 输出Excel文件路径（可选，为None时不保存）
    
    返回:
        dict: 包含所有分析结果的字典
    """
    df['weight'] = df["weight"].fillna(0)
    df_copy['weight'] = df_copy["weight"].fillna(0)
    df['combined_weight'] = df[w_col] + df["weight"]
    df_copy['combined_weight'] = df_copy[w_col] + df_copy["weight"]

    df_copy["ret_rank"] = df_copy["ret"].apply(lambda x: np.sum(x >= df_copy["ret"])) / len(df_copy)

    # 1. 分类处理barra_cols
    english_cols = [col for col in barra_cols if not is_chinese(col)]
    chinese_cols = [col for col in barra_cols if is_chinese(col)]
    
    # 计算port_allocation
    port_allocation_en = calculate_port_allocation(df, 'combined_weight', english_cols)
    #port_allocation_en = port_allocation_en.sort_values(ascending=False)
    
    port_allocation_cn = calculate_port_allocation(df, 'combined_weight', chinese_cols)
    #port_allocation_cn = port_allocation_cn.sort_values(ascending=False)
    
    # 2. 超额持仓股票分析 (select_ids)
    select_ids = df_copy[abs(df_copy[w_col]) > 1e-4].copy()
    active_weight_pos = select_ids[select_ids[w_col] > 0]
    active_weight_neg = select_ids[select_ids[w_col] < 0]
    
    select_active_stats = pd.DataFrame({
        '指标': ['主动权重总和', '主动权重（正）总和', '主动权重（负）总和',
                '主动股票数量', '主动股票（正）数量', '主动股票（负）数量'],
        '数值': [
            select_ids[w_col].sum(),
            active_weight_pos[w_col].sum(),
            active_weight_neg[w_col].sum(),
            len(select_ids),
            len(active_weight_pos),
            len(active_weight_neg)
        ]
    })
    
    # 3. 持仓股票分析 (hold_ids) - 先进行与select_ids相同的分析逻辑
    hold_ids = df_copy[abs(df_copy['combined_weight']) > 1e-4].copy()
    
    # 对hold_ids进行同样的主动权重分析
    hold_active_weight_pos = hold_ids[hold_ids['combined_weight'] > 0]
    hold_active_weight_neg = hold_ids[hold_ids['combined_weight'] < 0]
    
    hold_active_stats = pd.DataFrame({
        '指标': ['持仓权重总和', '持仓权重（正）总和', '持仓权重（负）总和',
                '持仓股票数量', '持仓股票（正）数量', '持仓股票（负）数量'],
        '数值': [
            hold_ids['combined_weight'].sum(),
            hold_active_weight_pos['combined_weight'].sum(),
            hold_active_weight_neg['combined_weight'].sum(),
            len(hold_ids),
            len(hold_active_weight_pos),
            len(hold_active_weight_neg)
        ]
    })
    
    # 4. 个性化分析 - 组合特征指标
    eff_n_stocks = calculate_effective_stocks(hold_ids['combined_weight'])
    
    top_n_weights = {
        '前5大权重和': calculate_top_n_weight_sum(hold_ids['combined_weight'], 5),
        '前10大权重和': calculate_top_n_weight_sum(hold_ids['combined_weight'], 10),
        '前15大权重和': calculate_top_n_weight_sum(hold_ids['combined_weight'], 15),
        '前20大权重和': calculate_top_n_weight_sum(hold_ids['combined_weight'], 20)
    }
    
    port_weighted_mcap = (hold_ids['combined_weight'] * hold_ids['free_circulation']).sum()
    benchmark_weighted_mcap = (hold_ids['weight'] * hold_ids['free_circulation']).sum()
    
    portfolio_stats = pd.DataFrame({
        '指标': ['有效持仓数(Neff)', '前5大持仓权重和', '前10大持仓权重和',
                '前15大权重和', '前20大持仓权重和', 
                '组合加权平均市值', '基准加权平均市值'],
        '数值': [
            eff_n_stocks,
            top_n_weights['前5大权重和'],
            top_n_weights['前10大权重和'],
            top_n_weights['前15大权重和'],
            top_n_weights['前20大权重和'],
            port_weighted_mcap,
            benchmark_weighted_mcap
        ]
    })
    
    # 保存到Excel文件（如果指定了路径）
    if output_path is not None:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            port_allocation_en.to_frame(name='超额暴露').to_excel(writer, sheet_name='Sheet1_英文因子暴露')
            port_allocation_cn.to_frame(name='超额暴露').to_excel(writer, sheet_name='Sheet2_行业因子暴露')
            select_ids.to_excel(writer, sheet_name='Sheet3_超额持仓股票', index=True)
            select_active_stats.to_excel(writer, sheet_name='Sheet3_超额持仓股票', startrow=len(select_ids)+3, index=False)
            hold_ids.to_excel(writer, sheet_name='Sheet4_持仓股票', index=True)
            hold_active_stats.to_excel(writer, sheet_name='Sheet4_持仓股票', startrow=len(hold_ids)+3, index=False)
            portfolio_stats.to_excel(writer, sheet_name='Sheet4_持仓股票', startrow=len(hold_ids)+len(hold_active_stats)+6, index=False)
        
        print(f"单日分析结果已保存到: {output_path}")
    
    return {
        'port_allocation_en': port_allocation_en,
        'port_allocation_cn': port_allocation_cn,
        'select_ids': select_ids,
        'select_active_stats': select_active_stats,
        'hold_ids': hold_ids,
        'hold_active_stats': hold_active_stats,
        'portfolio_stats': portfolio_stats
    }

def calculate_alpha_factor_exposure(df, w_cols=None):
    """
    计算组合在alpha因子上的配置情况（向量化实现）
    
    参数:
        df: 持仓信息DataFrame，包含权重列和alpha因子列
        w_cols: 权重列列表，默认为['w_opt_0.01', 'w_opt_0.1', 'w_opt_0.3', 'w_opt_0.5', 'w_opt_1', 'weight', "free_circulation"]
    
    返回:
        dict: 包含权重总和和因子暴露计算结果的字典
    """
    if w_cols is None:
        w_cols = ['w_opt_0.01', 'w_opt_0.1', 'w_opt_0.3', 'w_opt_0.5', 'w_opt_1', 'weight', "free_circulation"]
    
    alpha_cols = ['D1', 'D2', 'D3', 'D1_orth', 'D2_orth', 'D3_orth', "ret_rank"]
    
    # 确保权重列存在
    missing_w_cols = [col for col in w_cols if col not in df.columns]
    if missing_w_cols:
        raise ValueError(f"权重列 {missing_w_cols} 不存在于DataFrame中")
    
    # 确保alpha因子列存在
    missing_alpha_cols = [col for col in alpha_cols if col not in df.columns]
    if missing_alpha_cols:
        raise ValueError(f"Alpha因子列 {missing_alpha_cols} 不存在于DataFrame中")
    
    # 1. 计算各权重列的总和（保留四位小数）
    weight_sums = df[w_cols].sum().round(4).to_dict()
    
    # 2. 向量化计算加权平均值
    # weights_matrix: (n_stocks, n_weights)
    weights_matrix = df[w_cols].fillna(0).values
    # alpha_matrix: (n_stocks, n_factors)
    alpha_matrix = df[alpha_cols].fillna(0).values
    
    # 计算权重绝对值之和用于归一化
    weights_abs_sum = np.abs(weights_matrix).sum(axis=0)  # (n_weights,)
    weights_abs_sum[weights_abs_sum == 0] = 1  # 避免除零
    
    # 加权平均: (n_factors, n_weights) = (n_factors, n_stocks) @ (n_stocks, n_weights) / (n_weights,)
    weighted_means_matrix = (alpha_matrix.T @ weights_matrix) / weights_abs_sum
    weighted_means_matrix = weighted_means_matrix.round(4)
    
    # 转换为字典格式
    weighted_means = {}
    for i, alpha_col in enumerate(alpha_cols):
        for j, w_col in enumerate(w_cols):
            weighted_means[f'{alpha_col}_{w_col}'] = weighted_means_matrix[i, j]
    
    # 3. 计算简单算术平均
    simple_means = df[alpha_cols].mean().round(4).to_dict()
    simple_means = {f'{k}_simple_mean': v for k, v in simple_means.items()}
    
    return {
        'weight_sums': weight_sums,
        'weighted_means': weighted_means,
        'simple_means': simple_means
    }


def save_combined_results(all_results, output_path):
    """
    将所有日期的分析结果整合并保存到单个Excel文件中，包含alpha因子配置分析
    
    参数:
        all_results: 包含所有日期分析结果的列表
        output_path: 输出Excel文件路径
    """
    if not all_results:
        print("没有可保存的分析结果")
        return
    
    dates = [result['date'] for result in all_results]
    
    # 1. 整合 port_allocation_en
    port_allocation_en_df = pd.DataFrame()
    for result in all_results:
        port_allocation_en_df[result['date']] = result['port_allocation_en']
    port_allocation_en_df = port_allocation_en_df.T
    port_allocation_en_df.index.name = '日期'
    
    # 2. 整合 port_allocation_cn
    port_allocation_cn_df = pd.DataFrame()
    for result in all_results:
        port_allocation_cn_df[result['date']] = result['port_allocation_cn']
    port_allocation_cn_df = port_allocation_cn_df.T
    port_allocation_cn_df.index.name = '日期'
    
    # 3. 整合 select_active_stats
    select_active_stats_df = pd.DataFrame()
    for result in all_results:
        stats = result['select_active_stats'].set_index('指标')['数值']
        select_active_stats_df[result['date']] = stats
    select_active_stats_df = select_active_stats_df.T
    select_active_stats_df.index.name = '日期'
    
    # 4. 整合 hold_active_stats
    hold_active_stats_df = pd.DataFrame()
    for result in all_results:
        stats = result['hold_active_stats'].set_index('指标')['数值']
        hold_active_stats_df[result['date']] = stats
    hold_active_stats_df = hold_active_stats_df.T
    hold_active_stats_df.index.name = '日期'
    
    # 5. 整合 portfolio_stats
    portfolio_stats_df = pd.DataFrame()
    for result in all_results:
        stats = result['portfolio_stats'].set_index('指标')['数值']
        portfolio_stats_df[result['date']] = stats
    portfolio_stats_df = portfolio_stats_df.T
    portfolio_stats_df.index.name = '日期'
    
    # 6. 整合 alpha因子配置分析（与前面逻辑一致）
    alpha_factor_df = pd.DataFrame()
    for result in all_results:
        dt = result['date']
        alpha_result = result.get('alpha_factor')
        
        if alpha_result is not None:
            # 将所有指标合并到一行
            row_data = {}
            row_data.update(alpha_result['weight_sums'])
            row_data.update(alpha_result['weighted_means'])
            row_data.update(alpha_result['simple_means'])
            
            alpha_factor_df[dt] = pd.Series(row_data)
    
    alpha_factor_df = alpha_factor_df.T
    alpha_factor_df.index.name = '日期'
    
    # 保存到Excel文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        port_allocation_en_df.to_excel(writer, sheet_name='port_allocation_en')
        port_allocation_cn_df.to_excel(writer, sheet_name='port_allocation_cn')
        select_active_stats_df.to_excel(writer, sheet_name='select_active_stats')
        hold_active_stats_df.to_excel(writer, sheet_name='hold_active_stats')
        portfolio_stats_df.to_excel(writer, sheet_name='portfolio_stats')
        
        # 添加alpha因子配置分析工作表（合并到一张）
        if not alpha_factor_df.empty:
            alpha_factor_df.to_excel(writer, sheet_name='alpha_factor_exposure')
    
    print(f"所有日期的分析结果已整合保存到: {output_path}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    
    alpha_name = "D1_orth"
    w_col = "w_opt_0.01"
    wdir = f"E:/SJTU/实习/国泰海通/barra因子/result/组合优化/lgb_持仓信息/{alpha_name}_barra正交"
    srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base"

    start_dt = "2025-01-02"
    end_dt = "2026-03-25"
    trdates = pd.read_pickle(f"{srcdir}/trading_dates.pkl")
    tr_filter_op = [d for d in trdates if (d >= start_dt) and (d <= end_dt)]
    df_name = pd.read_excel(f"{srcdir}/全A代码_名称.xlsx",index_col=0)
    barra_cols = [
        'size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price',
        'leverage', 'growth', 'earnings_yield', 'beta', 'residual_volatility',
        '银行', '计算机', '环保', '商贸零售', '电力设备', '建筑装饰', '建筑材料',
        '农林牧渔', '电子', '交通运输', '汽车', '纺织服饰', '医药生物', '房地产',
        '通信', '公用事业', '综合', '机械设备', '石油石化', '有色金属', '传媒',
        '家用电器', '基础化工', '非银金融', '社会服务', '轻工制造', '国防军工',
        '美容护理', '煤炭', '食品饮料', '钢铁'
    ]
    
    # 收集所有日期的分析结果
    all_results = []
    
    for dt in tr_filter_op[1:]: #2:D1_orth!
        print(f"Processing date: {dt}")
        
        df = pd.read_csv(f"{wdir}/w_opt持仓信息_{alpha_name}_{dt}.csv").set_index("order_book_id")
        df_copy = df.copy()
        
        stock_names = df_copy.index.map(df_name['stock_name']).fillna('')
        df_copy.insert(0, 'name', stock_names)
        
        b_dt = pd.Series(tr_filter_op)[pd.Series(tr_filter_op) < dt].iloc[-1]
        X_center = pd.read_pickle(f"{srcdir}/barra_data/000905标准化3_含行业/{b_dt}.pkl").set_index("order_book_id")
        X_center = X_center.reindex(df.index)
        df[barra_cols] = X_center[barra_cols]

        # 保存单日分析报告（保留原有保存习惯）
        daily_output_path = f"{wdir}/持仓分析报告/{w_col}_{dt}.xlsx"
        results = analyze_portfolio(df_copy, df, w_col, barra_cols, daily_output_path)
        
        # 添加日期信息
        results['date'] = dt
        
        # 计算并添加alpha因子配置分析结果
        try:
            alpha_result = calculate_alpha_factor_exposure(df_copy)
            results['alpha_factor'] = alpha_result
        except Exception as e:
            print(f"计算日期 {dt} 的alpha因子配置时发生错误: {str(e)}")
            results['alpha_factor'] = None
        
        # 收集结果
        all_results.append(results)

        
    
    #将所有日期的分析结果整合到单个Excel文件（包含alpha因子配置分析）
    combined_output_path = f"{wdir}/持仓分析报告/综合分析报告_{w_col}_{start_dt}_to_{end_dt}.xlsx"
    save_combined_results(all_results, combined_output_path)
    