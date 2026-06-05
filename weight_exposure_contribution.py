import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

srcdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_result/风格因子超额暴露分析"
retdir = "E:/SJTU/实习/国泰海通/业绩回测/result/中证500指增产品各收益回测_超额收益贡献"


def parse_nature_period(nature_period):
    if nature_period == "成立以来":
        return None, None
    
    if nature_period.endswith('年') and 'Q' not in nature_period:
        year = int(nature_period[:4])
        return pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31')
    
    if 'Q' in nature_period:
        year = int(nature_period[:4])
        quarter = int(nature_period[-1])
        quarter_dates = {
            1: ('01-01', '03-31'),
            2: ('04-01', '06-30'),
            3: ('07-01', '09-30'),
            4: ('10-01', '12-31')
        }
        start_month_day, end_month_day = quarter_dates[quarter]
        return pd.to_datetime(f'{year}-{start_month_day}'), pd.to_datetime(f'{year}-{end_month_day}')
    
    return None, None


def generate_time_periods(start_year=2020, end_year=2026, end_quarter=1):
    periods = ['成立以来']
    
    for year in range(start_year, end_year + 1):
        periods.append(f'{year}年')
        
        max_q = 4
        if year == end_year:
            max_q = end_quarter
        
        for q in range(1, max_q + 1):
            periods.append(f'{year}年Q{q}')
    
    return periods


def clean_filename(name):
    invalid_chars = '<>:\"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


def analyze_exposure_results(input_file, sheet_name='成立以来'):
    img_dir = f"{srcdir}/图片"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    print(f"1. 加载数据：{sheet_name}")
    df = pd.read_excel(input_file, sheet_name=sheet_name, index_col=0)
    print(f"   数据形状: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")
    
    print("\n2. 统计分析 - t绝对值>1.645的因子数")
    t_count_col = 't绝对值>1.645的因子数'
    if t_count_col in df.columns:
        t_count_data = df[t_count_col].dropna()
        t_count_mean = t_count_data.mean()
        t_count_median = t_count_data.median()
        print(f"   均值: {t_count_mean:.2f}")
        print(f"   中位数: {t_count_median:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(t_count_data, bins='auto', edgecolor='black')
        ax.set_title(f'{t_count_col} 频率分布直方图')
        ax.set_xlabel(t_count_col)
        ax.set_ylabel('频率')
        ax.grid(axis='y', alpha=0.75)
        plt.savefig(f"{img_dir}/{clean_filename(t_count_col)}_直方图.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   直方图已保存: {t_count_col}_直方图.png")
    else:
        print(f"   警告：未找到列 '{t_count_col}'")
    
    print("\n3. 统计分析 - 总风格超额暴露强度")
    exposure_col = '总风格超额暴露强度'
    if exposure_col in df.columns:
        exposure_data = df[exposure_col].dropna()
        exposure_mean = exposure_data.mean()
        exposure_median = exposure_data.median()
        print(f"   均值: {exposure_mean:.4f}")
        print(f"   中位数: {exposure_median:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(exposure_data, bins=20, edgecolor='black')
        ax.set_title(f'{exposure_col} 频率分布直方图')
        ax.set_xlabel(exposure_col)
        ax.set_ylabel('频率')
        ax.grid(axis='y', alpha=0.75)
        plt.savefig(f"{img_dir}/{clean_filename(exposure_col)}_直方图.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   直方图已保存: {exposure_col}_直方图.png")
    else:
        print(f"   警告：未找到列 '{exposure_col}'")
    
    print("\n4. 因子暴露深度分析")
    top_columns = ['top1_因子', 'top2_因子', 'top3_因子']
    t_value_columns = ['top1_t值', 'top2_t值', 'top3_t值']
    mean_columns = ['top1_平均值', 'top2_平均值', 'top3_平均值']
    
    top_stats = {}
    
    for i in range(3):
        level = f'top{i+1}'
        top_stats[level] = {
            'factors': [],
            't_values': [],
            'means': []
        }
        
        if top_columns[i] in df.columns:
            factors = df[top_columns[i]].dropna().tolist()
            top_stats[level]['factors'] = factors
            
            if t_value_columns[i] in df.columns:
                t_vals = df[[top_columns[i], t_value_columns[i]]].dropna()
                top_stats[level]['t_values'] = [(f, t) for f, t in zip(t_vals[top_columns[i]], t_vals[t_value_columns[i]])]
            
            if mean_columns[i] in df.columns:
                means = df[[top_columns[i], mean_columns[i]]].dropna()
                top_stats[level]['means'] = [(f, m) for f, m in zip(means[top_columns[i]], means[mean_columns[i]])]
    
    all_factor_counts = []
    all_factor_stats = []
    inconsistent_factors = []
    
    for level in ['top1', 'top2', 'top3']:
        factors = top_stats[level]['factors']
        t_values = top_stats[level]['t_values']
        means = top_stats[level]['means']
        
        factor_counts = pd.Series(factors).value_counts().reset_index()
        factor_counts.columns = ['因子名称', '出现频次']
        factor_counts['占比(%)'] = (factor_counts['出现频次'] / factor_counts['出现频次'].sum() * 100).round(2)
        factor_counts['top层级'] = level
        all_factor_counts.append(factor_counts)
        
        print(f"\n   {level} 因子出现频次统计:")
        print(factor_counts)
        
        t_value_dict = defaultdict(list)
        for factor, t_val in t_values:
            t_value_dict[factor].append(t_val)
        
        mean_value_dict = defaultdict(list)
        for factor, mean_val in means:
            mean_value_dict[factor].append(mean_val)
        
        for factor in factor_counts['因子名称']:
            t_vals = t_value_dict.get(factor, [])
            mean_vals = mean_value_dict.get(factor, [])
            
            t_mean = np.mean(t_vals) if t_vals else None
            t_has_positive = any(v > 0 for v in t_vals)
            t_has_negative = any(v < 0 for v in t_vals)
            t_inconsistent = t_has_positive and t_has_negative
            
            m_mean = np.mean(mean_vals) if mean_vals else None
            m_has_positive = any(v > 0 for v in mean_vals)
            m_has_negative = any(v < 0 for v in mean_vals)
            m_inconsistent = m_has_positive and m_has_negative
            
            if t_inconsistent or m_inconsistent:
                exists = False
                for item in inconsistent_factors:
                    if item['因子名称'] == factor:
                        item[f'{level}_t值方向不一致'] = t_inconsistent
                        item[f'{level}_平均值方向不一致'] = m_inconsistent
                        exists = True
                        break
                if not exists:
                    inconsistent_factors.append({
                        '因子名称': factor,
                        f'{level}_t值方向不一致': t_inconsistent,
                        f'{level}_平均值方向不一致': m_inconsistent
                    })
            
            all_factor_stats.append({
                '因子名称': factor,
                'top层级': level,
                't值均值': t_mean,
                '平均值均值': m_mean,
                '出现频次': factor_counts[factor_counts['因子名称'] == factor]['出现频次'].iloc[0],
                't值方向不一致': t_inconsistent,
                '平均值方向不一致': m_inconsistent
            })
    
    factor_counts_df = pd.concat(all_factor_counts, ignore_index=True)
    factor_stats_df = pd.DataFrame(all_factor_stats)
    
    print("\n   合并因子统计详情:")
    print(factor_stats_df)
    
    if inconsistent_factors:
        inconsistent_df = pd.DataFrame(inconsistent_factors)
        print("\n   方向不一致的因子:")
        print(inconsistent_df)
    else:
        print("\n   未发现方向不一致的因子")
    
    print("\n6. 输出分析结果")
    output_file = f"{srcdir}/深度分析/风格因子超额暴露分析结果_explore.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        summary_data = {
            '统计指标': ['t绝对值>1.645的因子数_均值', 't绝对值>1.645的因子数_中位数',
                       '总风格超额暴露强度_均值', '总风格超额暴露强度_中位数'],
            '数值': [t_count_mean, t_count_median, exposure_mean, exposure_median]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
        
        factor_counts_df.to_excel(writer, sheet_name='因子频次统计', index=False)
        
        factor_stats_df.to_excel(writer, sheet_name='因子统计详情', index=False)
        
        if inconsistent_factors:
            inconsistent_df.to_excel(writer, sheet_name='方向不一致因子', index=False)
        else:
            pd.DataFrame({'说明': ['未发现方向不一致的因子']}).to_excel(writer, sheet_name='方向不一致因子', index=False)
        
        df.to_excel(writer, sheet_name='原始数据')
    
    print(f"   分析结果已保存至: {output_file}")
    return output_file


def run_single_period_analysis(nature_period, start_date=None,end_date=None,ret_cols = ["累计超额收益", "累计残差贡献", "累计风格因子贡献"], bmk=905, choice="resid_style"):
    """
    运行单个时间段的分析
    
    参数:
        nature_period: 时间段描述，如"2024年"、"2024年Q1"、"成立以来"
        ret_cols: 收益指标列名列表
        bmk: 基准代码
        choice: 分析类型，"exp_ret_vol"为风格暴露vs收益/波动，"resid_style"为残差vs风格贡献
    
    返回:
        corr_results: 相关系数分析结果列表
    """
    print(f"\n{'='*60}")
    print(f"处理时间段: {nature_period}") if nature_period else print(f"处理时间范围: {start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}")
    print(f"分析类型: {choice}")
    print('='*60)
    from weight_factor_exposure import run_factor_exposure_analysis
    from weight_contribution import run_contribution_analysis

    if nature_period:
        shadow_sd, shadow_ed = parse_nature_period(nature_period)
        sheet = nature_period
        exp_path = f"{srcdir}/风格因子超额暴露分析结果_{bmk}.xlsx"
        ret_path = run_contribution_analysis(bmk,shadow_sd, shadow_ed)
    else:
        sheet = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        exp_path = run_factor_exposure_analysis(bmk,start_date,end_date)
        ret_path = run_contribution_analysis(bmk,start_date,end_date)
    
    corr_results = []
    
    if ret_path:
        exp_xls = pd.ExcelFile(exp_path)
        if sheet not in exp_xls.sheet_names:
            print(f"     警告：Excel文件中不存在sheet '{sheet}'，跳过该时间段")
            return corr_results
        
        df_exp = exp_xls.parse(sheet, index_col=0)
        df_ret = pd.read_excel(ret_path, sheet_name="详细信息", index_col=0).sort_index()
        
        merge_dir = f"{srcdir}/综合分析_{choice}"
        if not os.path.exists(merge_dir):
            os.makedirs(merge_dir)
        
        output_path = f"{merge_dir}/综合分析结果_{sheet}_{bmk}.xlsx"
        
        with pd.ExcelWriter(output_path) as writer:
            if choice == "exp_ret_vol":
                # 原来的逻辑：逐个col循环
                for col in ret_cols:
                    filtered_ret = df_ret[df_ret["指标"] == col]
                    if nature_period == "成立以来":
                        filtered_ret = filtered_ret[filtered_ret["时间区间"]==nature_period]
                    
                    merged_df = df_exp.merge(filtered_ret, left_index=True, right_index=True, how='inner')
                    merged_df = merged_df.dropna(how='any')
                    merged_df.to_excel(writer, sheet_name=col[:31])
                    
                    print(f"\n   {col} 相关系数分析:")
                    
                    if '总风格超额暴露强度' in merged_df.columns:
                        exposure_data = merged_df['总风格超额暴露强度']
                        
                        if len(exposure_data) < 2:
                            print(f"     警告：数据量不足（{len(exposure_data)}条），跳过相关系数计算")
                        else:
                            if '年化收益' in merged_df.columns:
                                pearson_r, pearson_p = stats.pearsonr(exposure_data, merged_df['年化收益'])
                                spearman_r, spearman_p = stats.spearmanr(exposure_data, merged_df['年化收益'])
                                
                                corr_results.append({
                                    '时间段': nature_period,
                                    '指标': col,
                                    '相关类型': '总风格超额暴露强度 vs 年化收益',
                                    '皮尔逊系数': pearson_r,
                                    '皮尔逊p值': pearson_p,
                                    '斯皮尔曼系数': spearman_r,
                                    '斯皮尔曼p值': spearman_p
                                })
                            
                            if '年化波动' in merged_df.columns:
                                pearson_r, pearson_p = stats.pearsonr(exposure_data, merged_df['年化波动'])
                                spearman_r, spearman_p = stats.spearmanr(exposure_data, merged_df['年化波动'])
                                
                                corr_results.append({
                                    '时间段': nature_period,
                                    '指标': col,
                                    '相关类型': '总风格超额暴露强度 vs 年化波动',
                                    '皮尔逊系数': pearson_r,
                                    '皮尔逊p值': pearson_p,
                                    '斯皮尔曼系数': spearman_r,
                                    '斯皮尔曼p值': spearman_p
                                })
                    else:
                        print(f"     警告：未找到 '总风格超额暴露强度' 列")
            
            elif choice == "resid_style":
                # 新逻辑：直接获取两个col的数据，不需要暴露数据
                col1, col2 = "累计残差贡献", "累计风格因子贡献"
                
                filtered_ret1 = df_ret[df_ret["指标"] == col1]
                filtered_ret2 = df_ret[df_ret["指标"] == col2]
                if nature_period == "成立以来":
                    filtered_ret1 = filtered_ret1[filtered_ret1["时间区间"]==nature_period]
                    filtered_ret2 = filtered_ret2[filtered_ret2["时间区间"]==nature_period]
                
                # 重命名列以便合并
                data1 = filtered_ret1[["区间收益"]].rename(columns={"区间收益": col1})
                data2 = filtered_ret2[["区间收益"]].rename(columns={"区间收益": col2})
                
                # 直接合并两个col的数据
                merged_df = data1.join(data2, how='inner').dropna()
                merged_df.to_excel(writer, sheet_name=f"{col1[:15]}_{col2[:15]}")
                
                print(f"\n   {col1} vs {col2} 相关系数分析:")
                
                if col1 in merged_df.columns and col2 in merged_df.columns:
                    x_data = merged_df[col1]
                    y_data = merged_df[col2]
                    
                    if len(x_data) >= 2 and len(y_data) >= 2:
                        pearson_r, pearson_p = stats.pearsonr(x_data, y_data)
                        spearman_r, spearman_p = stats.spearmanr(x_data, y_data)
                        
                        corr_results.append({
                            '时间段': nature_period,
                            '指标': f"{col1}+{col2}",
                            '相关类型': f"{col1} vs {col2}",
                            '皮尔逊系数': pearson_r,
                            '皮尔逊p值': pearson_p,
                            '斯皮尔曼系数': spearman_r,
                            '斯皮尔曼p值': spearman_p
                        })
                    else:
                        print(f"     警告：数据量不足，跳过计算")
                else:
                    print(f"     警告：缺少必要列")
            
            print(f"\n   - 综合分析结果已保存至: {output_path}")
    
    return corr_results


def run_multi_period_analysis(start_year=2020, end_year=2026, end_quarter=1, ret_cols=None, bmk=905, choice="resid_style"):
    """
    运行多个时间段的分析并汇总结果
    
    参数:
        start_year: 开始年份
        end_year: 结束年份
        end_quarter: 结束季度
        ret_cols: 收益指标列名列表
        bmk: 基准代码
        choice: 分析类型，"exp_ret_vol"为风格暴露vs收益/波动，"resid_style"为残差vs风格贡献
    """
    if ret_cols is None:
        ret_cols = ["累计超额收益", "累计残差贡献", "累计风格因子贡献"]
    
    periods = generate_time_periods(start_year, end_year, end_quarter)
    print(f"生成的时间段列表: {periods}")
    print(f"分析类型: {choice}")
    
    all_corr_results = []
    
    for period in periods:
        results = run_single_period_analysis(period, ret_cols=ret_cols, bmk=bmk, choice=choice)
        all_corr_results.extend(results)
    
    if all_corr_results:
        corr_df = pd.DataFrame(all_corr_results)
        
        summary_dir = f"{srcdir}/综合分析_{choice}"
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        
        summary_path = f"{summary_dir}/相关系数汇总分析_{start_year}-{end_year}Q{end_quarter}_{bmk}.xlsx"
        with pd.ExcelWriter(summary_path) as writer:
            corr_df.to_excel(writer, sheet_name='所有结果', index=False)
            
            grouped = corr_df.groupby(['指标', '相关类型'])
            for (indicator, corr_type), group in grouped:
                numeric_cols = ['皮尔逊系数', '皮尔逊p值', '斯皮尔曼系数', '斯皮尔曼p值']
                avg_df = group.groupby('时间段')[numeric_cols].mean().reset_index()
                avg_df.to_excel(writer, sheet_name=f"{indicator}_{corr_type[:20]}", index=False)
        
        print(f"\n{'='*60}")
        print("相关系数汇总统计（各时间段平均值）")
        print('='*60)
        
        grouped = corr_df.groupby(['指标', '相关类型'])
        for (indicator, corr_type), group in grouped:
            print(f"\n{indicator} - {corr_type}:")
            print(f"  皮尔逊系数均值: {group['皮尔逊系数'].mean():.4f}")
            print(f"  皮尔逊p值均值: {group['皮尔逊p值'].mean():.4f}")
            print(f"  斯皮尔曼系数均值: {group['斯皮尔曼系数'].mean():.4f}")
            print(f"  斯皮尔曼p值均值: {group['斯皮尔曼p值'].mean():.4f}")
        
        print(f"\n   - 相关系数汇总分析已保存至: {summary_path}")
        return summary_path
    else:
        print("\n   - 未生成任何相关系数结果")
        return None


def combine_bmk_corr_results(bmk_list, start_year=2020, end_year=2026, end_quarter=1, choice="resid_style"):
    """
    综合多个bmk的相关系数汇总分析结果
    
    参数:
        bmk_list: 基准代码列表，如 [300, 905, 852, 932000]
        start_year: 开始年份
        end_year: 结束年份
        end_quarter: 结束季度
        choice: 分析类型，"exp_ret_vol"为风格暴露vs收益/波动，"resid_style"为残差vs风格贡献
    
    返回:
        combined_path: 综合后的Excel文件路径
    """
    print(f"\n{'='*60}")
    print("综合多个bmk的相关系数汇总分析结果")
    print('='*60)
    print(f"待综合的bmk列表: {bmk_list}")
    print(f"分析类型: {choice}")
    
    summary_dir = f"{srcdir}/综合分析_{choice}"
    all_sheet_data = defaultdict(dict)
    all_sheet_names = set()
    
    for bmk in bmk_list:
        file_path = f"{summary_dir}/相关系数汇总分析_{start_year}-{end_year}Q{end_quarter}_{bmk}.xlsx"
        
        if not os.path.exists(file_path):
            print(f"   警告：未找到文件 {file_path}，跳过")
            continue
        
        print(f"   加载文件: {file_path}")
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        all_sheet_names.update(sheet_names)
        
        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            all_sheet_data[sheet_name][bmk] = df
    
    if not all_sheet_data:
        print("   未找到任何有效的数据文件")
        return None
    
    combined_path = f"{summary_dir}/相关系数汇总分析_{start_year}-{end_year}Q{end_quarter}_综合版_{choice}.xlsx"
    with pd.ExcelWriter(combined_path) as writer:
        for sheet_name in sorted(all_sheet_names):
            bmk_dfs = all_sheet_data[sheet_name]
            if not bmk_dfs:
                continue
            
            merged_df = None
            for bmk, df in bmk_dfs.items():
                df_copy = df.copy()
                df_copy.set_index('时间段', inplace=True)
                df_copy.columns = [f"{col}_{bmk}" for col in df_copy.columns]
                
                if merged_df is None:
                    merged_df = df_copy
                else:
                    merged_df = merged_df.merge(df_copy, left_index=True, right_index=True, how='outer')
            
            merged_df.to_excel(writer, sheet_name=sheet_name, index=True)
            print(f"   Sheet '{sheet_name}' 已横向合并 {len(bmk_dfs)} 个bmk的数据")
    
    print(f"\n   - 综合分析结果已保存至: {combined_path}")
    return combined_path


if __name__ == "__main__":
    ret_cols = ["累计超额收益", "累计残差贡献", "累计风格因子贡献"]
    # bmk = 932000
    # run_multi_period_analysis(start_year=2020, end_year=2026, end_quarter=1, ret_cols=ret_cols, bmk=bmk)
    bmk_list = [300, 905, 852, 932000]
    
    for bmk in bmk_list:
        print(f"\n{'='*60}")
        print(f"处理基准: {bmk}")
        print('='*60)
        run_multi_period_analysis(start_year=2020, end_year=2026, end_quarter=1, ret_cols=ret_cols, bmk=bmk)
    
    combine_bmk_corr_results(bmk_list, start_year=2020, end_year=2026, end_quarter=1)
