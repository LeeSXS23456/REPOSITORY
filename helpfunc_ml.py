def evaluate_factor(pred, ret, sample_weight=None):
    """
    评估因子预测效果
    
    参数:
        pred: 预测值数组
        ret: 实际收益率数组
        sample_weight: 样本权重（可选）
    
    返回:
        dict: 包含评估指标的字典
    """
    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    import numpy as np
    
    # 计算相关系数
    pearson_ic = pearsonr(pred, ret)[0]
    rank_ic = spearmanr(pred, ret)[0]
    
    # 计算 R²
    ss_res = np.sum((ret - pred) ** 2)
    ss_tot = np.sum((ret - ret.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # 十组分组测试
    df_eval = pd.DataFrame({"pred": pred, "ret": ret})
    df_eval["group"] = pd.qcut(df_eval["pred"], 10, labels=False, duplicates='drop')
    group_ret = df_eval.groupby("group")["ret"].mean()
    
    # 计算多空收益
    ls_ret = group_ret.iloc[-1] - group_ret.iloc[0]
    
    # 获取第1组和第10组的收益
    group1_ret = group_ret.iloc[0]
    group10_ret = group_ret.iloc[-1]
    
    # 计算贡献度/强度比（根据两组收益符号判断）
    # 判断两组收益是否同号（都不为0时）
    if group1_ret != 0 and group10_ret != 0:
        is_same_sign = np.sign(group1_ret) == np.sign(group10_ret)
    else:
        is_same_sign = False
    
    if is_same_sign:
        # ①同号：计算强度比 = 第10组收益 / 第1组收益
        strength_ratio = group10_ret / group1_ret
        contribution_type = "强度"
        group1_contribution_ratio = np.nan
        group10_contribution_ratio = np.nan
    else:
        # ②异号：计算贡献占比
        if ls_ret != 0:
            group1_contribution_ratio = (-group1_ret) / ls_ret
            group10_contribution_ratio = group10_ret / ls_ret
        else:
            group1_contribution_ratio = np.nan
            group10_contribution_ratio = np.nan
        strength_ratio = np.nan
        contribution_type = "贡献"
    
    # 返回结果字典
    return {
        "pearson_ic": pearson_ic,
        "rank_ic": rank_ic,
        "r2": r2,
        "ls_ret": ls_ret,# 贡献度/强度指标
        "group1_ret": group1_ret,          # 第1组收益
        "group10_ret": group10_ret,        # 第10组收益
        "strength_ratio": strength_ratio,           # 强度比（同号时）
        "group1_contribution_ratio": group1_contribution_ratio,  # 第1组贡献占比（异号时）
        "group10_contribution_ratio": group10_contribution_ratio,  # 第10组贡献占比（异号时）
        "contribution_type": contribution_type,      # 标记："强度" 或 "贡献"
        "group_ret": group_ret.to_dict()  # Series 转为字典
    }