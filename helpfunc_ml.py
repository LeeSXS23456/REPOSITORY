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
    
    # 返回结果字典
    return {
        "pearson_ic": pearson_ic,
        "rank_ic": rank_ic,
        "r2": r2,
        "ls_ret": ls_ret,
        "group_ret": group_ret.to_dict()  # Series 转为字典
    }