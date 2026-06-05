import pandas as pd
import numpy as np

def analyze_style_stocks(start_dt, end_dt):
    """分析风格因子高分/低分股票统计"""
    srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/barra_data/whole_mkt"
    trdir = "E:/SJTU/实习/国泰海通/barra因子/data_base"
    outdir = "E:/SJTU/实习/国泰海通/barra因子/result/风格典型股票"
    
    styles = ['size', 'non_linear_size', 'momentum', 'liquidity', 
              'book_to_price', 'leverage', 'growth', 'earnings_yield',
              'beta', 'residual_volatility']
    
    # 1. 获取交易日期和个股名称
    trdates = pd.read_pickle(f"{trdir}/trading_dates.pkl")
    filter_dates = [d for d in trdates if start_dt <= d <= end_dt]
    df_names = pd.read_excel(f"{trdir}/全A代码_名称.xlsx")
    
    # 2. 初始化存储结构
    high_dfs = {s: [] for s in styles}
    low_dfs = {s: [] for s in styles}
    
    # 3. 遍历日期收集数据
    for i, date in enumerate(filter_dates):
        if i % 20 == 0:
            print(f"Processing {i+1}/{len(filter_dates)}: {date}")
        
        try:
            df = pd.read_pickle(f"{srcdir}/{date}.pkl")
        except FileNotFoundError:
            continue
        
        df = df.merge(df_names, on='order_book_id', how='left')
        df = df.set_index(['order_book_id',"stock_name"])

        high_mask = df[styles] >= df[styles].quantile(0.9)
        low_mask = df[styles] <= df[styles].quantile(0.1)
        
        for s in styles:
            high_dfs[s].append(df.loc[high_mask[s], [s]])
            low_dfs[s].append(df.loc[low_mask[s], [s]])
    
    # 4. 定义统计处理函数（只定义一次）
    def process_and_save(df_list, style, writer, suffix):
        if not df_list:
            return
        
        combined = pd.concat(df_list)
        stats = combined.groupby(combined.index)[style].agg([
            ('上榜次数', 'count'), ('平均值', 'mean'), ('中位数', 'median'),
            ('波动率', 'std'), ('最小值', 'min'), ('最大值', 'max')
        ])
        stats['极差'] = stats['最大值'] - stats['最小值']
        stats = stats.sort_values('上榜次数', ascending=False).reset_index()
        stats.to_excel(writer, sheet_name=f'{style}{suffix}',index=False)
    
    # 5. 输出结果
    with pd.ExcelWriter(f"{outdir}/风格因子高分股票统计.xlsx", engine='openpyxl') as writer:
        for s in styles:
            process_and_save(high_dfs[s], s, writer, '_高分')
            process_and_save(low_dfs[s], s, writer, '_低分')
    
    print("统计完成！")

if __name__ == "__main__":
    start_dt, end_dt = "2025-01-01", "2026-03-25"

    analyze_style_stocks(start_dt, end_dt)
