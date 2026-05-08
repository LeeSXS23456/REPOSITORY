import numpy as np
import pandas as pd

srcdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露"

print("=" * 60)
print("分析 R2 与测算误差的相关性")
print("=" * 60)

print("\n1. 读取数据...")
df_R2 = pd.read_excel(f"{srcdir}/stats/all_fund_stats.xlsx", index_col=0)
print(f"   df_R2形状: {df_R2.shape}")

df_w = pd.read_excel(f"{srcdir}/weight_exposure/脱敏barra暴露偏离数据2020-2026.xlsx", index_col=0)
df_w.index = pd.to_datetime(df_w.index)
df_w.sort_index(inplace=True)
print(f"   df_w形状: {df_w.shape}")

ids = df_w["编码"].unique()
print(f"   总编码数量: {len(ids)}")

style_factors = ["贝塔暴露", "账面市值比暴露", "盈利率暴露", "成长性暴露", "杠杆率暴露", "流动性暴露", "动量暴露", "非线性市值暴露", "残余波动率暴露", "规模暴露"]
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

all_results = []

print("\n2. 处理每个编码...")
found_count = 0
empty_count = 0

for i, id in enumerate(ids):
    exposure_path = f"{srcdir}/excess_exposure/{id}_relative_exposure.xlsx"
    
    if not pd.io.common.file_exists(exposure_path):
        continue
    
    df_exposure = pd.read_excel(exposure_path, index_col=0)
    
    english_cols = [col for col in list(factor_name_map.values()) if col in df_exposure.columns]
    if not english_cols:
        continue
    
    df_exposure = df_exposure[english_cols]
    df_exposure_shifted = df_exposure.shift(-1)
    
    df_w_id = df_w[df_w["编码"] == id]
    chinese_cols = [col for col in style_factors if col in df_w_id.columns]
    if not chinese_cols:
        continue
    
    df_w_id = df_w_id[chinese_cols]
    
    merged = df_w_id.merge(df_exposure_shifted, left_index=True, right_index=True, how='inner')
    
    if merged.empty:
        empty_count += 1
        continue
    
    found_count += 1
    
    merged['测算误差'] = 0
    for chinese_name, english_name in factor_name_map.items():
        if chinese_name in merged.columns and english_name in merged.columns:
            merged['测算误差'] += np.abs(merged[chinese_name] - merged[english_name])
    
    merged['编码'] = id
    merged['date'] = merged.index
    
    all_results.append(merged[['date', '编码', '测算误差']])
    
    if (i + 1) % 20 == 0:
        print(f"   已处理 {i + 1}/{len(ids)} 个编码")

print(f"\n   找到文件数量: {found_count}")
print(f"   合并后为空的数量: {empty_count}")
print(f"   有效结果数量: {len(all_results)}")

if all_results:
    df_combined = pd.concat(all_results).copy()
    print(f"   合并后数据形状: {df_combined.shape}")
    
    print("\n3. 合并 R2 数据...")
    df_R2_shifted = df_R2.shift(-1)
    df_R2_shifted = df_R2_shifted.reset_index()
    df_R2_shifted['fund'] = df_R2_shifted['fund'].astype(str)
    print(f"   df_R2_shifted形状: {df_R2_shifted.shape}")
    
    df_combined['fund'] = df_combined['编码'].astype(str)
    
    final_df = df_combined.merge(df_R2_shifted, left_on=['date', 'fund'], right_on=['date', 'fund'], how='inner')
    print(f"   最终合并后形状: {final_df.shape}")
    
    
    if not final_df.empty:
        print("\n4. 计算总体相关系数...")
        pearson_corr = final_df['r_squared_pca'].corr(final_df['测算误差'], method='pearson')
        spearman_corr = final_df['r_squared_pca'].corr(final_df['测算误差'], method='spearman')
        
        print("=" * 60)
        print("总体相关系数")
        print("=" * 60)
        print(f"皮尔逊相关系数: {pearson_corr:.4f}")
        print(f"斯皮尔曼相关系数: {spearman_corr:.4f}")
        print("=" * 60)
        
        output_path = f"{srcdir}/weight_result/weight_with_reg/R2_vs_error_analysis.xlsx"
        #final_df.to_excel(output_path, index=False)
        print(f"分析结果已保存到: {output_path}")
        
        print("\n5. 按日期分组计算相关系数...")
        grouped = final_df.groupby('date')
        
        corr_results = []
        for date, group in grouped:
            if len(group) >= 2:
                p_corr = group['r_squared_pca'].corr(group['测算误差'], method='pearson')
                s_corr = group['r_squared_pca'].corr(group['测算误差'], method='spearman')
                corr_results.append({
                    'date': date,
                    'corr_p': p_corr,
                    'corr_s': s_corr,
                    '测算样本数': len(group)
                })
        
        corr_df = pd.DataFrame(corr_results)
        
        corr_df_filtered = corr_df[corr_df['测算样本数'] >= 30]
        
        avg_corr_p = corr_df_filtered['corr_p'].mean()
        avg_corr_s = corr_df_filtered['corr_s'].mean()
        
        abs_avg_corr_p = corr_df_filtered['corr_p'].abs().mean()
        abs_avg_corr_s = corr_df_filtered['corr_s'].abs().mean()
        
        print("\n" + "=" * 60)
        print("按日期分组相关系数统计")
        print("=" * 60)
        print(f"筛选条件: 测算样本数 >=30")
        print(f"筛选后日期数: {len(corr_df_filtered)}")
        print(f"corr_p 列平均值: {avg_corr_p:.4f}")
        print(f"corr_s 列平均值: {avg_corr_s:.4f}")
        print(f"|corr_p| 列平均值: {abs_avg_corr_p:.4f}")
        print(f"|corr_s| 列平均值: {abs_avg_corr_s:.4f}")
        print("=" * 60)
        
        corr_output_path = f"{srcdir}/weight_result/weight_with_reg/daily_correlation_analysis.xlsx"
        #corr_df.to_excel(corr_output_path, index=False)
        print(f"每日相关系数结果已保存到: {corr_output_path}")
    else:
        print("警告: 最终合并后数据为空")
else:
    print("没有找到任何有效的暴露文件")