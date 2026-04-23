import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from logging_utils import setup_logger

# 配置日志记录器
logger = setup_logger('delay_alpha')

# 提前计算市场收益率的滞后项
def precompute_market_lags(market_returns, max_lag=10):
    """提前计算市场收益率的滞后项"""
    market_data = pd.DataFrame({'market': market_returns})
    for n in range(1, max_lag + 1):
        market_data[f'market_lag{n}'] = market_returns.shift(n)
    return market_data

# 2. 计算EWMA权重
def calculate_ewma_weights(window, half_life=63):
    """计算EWMA权重"""
    decay_factor = 0.5 ** (1 / half_life)
    weights = [decay_factor ** i for i in range(window-1, -1, -1)]
    weights = np.array(weights) / sum(weights)
    return weights


# 单只股票的延迟计算
def calculate_stock_delay(stock, stock_returns, market_returns, window=126, lag=10):
    """计算单只股票的延迟指标"""
    # 注意：这里不要使用logger，因为在子进程中使用会导致重复日志
    d1 = pd.Series(index=stock_returns.index, name=stock)
    d2 = pd.Series(index=stock_returns.index, name=stock)
    d3 = pd.Series(index=stock_returns.index, name=stock)
    
    # 获取股票收益
    r = stock_returns[stock]
    
    # 向量化计算：创建滚动窗口矩阵
    # 提取股票收益和市场数据
    stock_series = r.dropna()
    market_series = market_returns.loc[stock_series.index]
    
    # 确保数据长度足够
    if len(stock_series) < window:
        return d1, d2, d3
    
    # 构建滞后矩阵
    def create_lag_matrix(data, max_lag):
        """创建滞后矩阵"""
        n = len(data)
        lag_matrix = np.zeros((n, max_lag + 1))
        lag_matrix[:, 0] = data.values
        for i in range(1, max_lag + 1):
            lag_matrix[i:, i] = data.values[:-i]
        return lag_matrix
    
    # 创建股票收益和市场收益的滞后矩阵
    stock_lags = create_lag_matrix(stock_series, lag)
    market_lags = create_lag_matrix(market_series, lag)
    
    # 滚动窗口计算
    for i in range(window, len(stock_series)):
        # 获取窗口数据
        window_start = i - window
        window_end = i
        
        # 提取窗口数据
        y_window = stock_lags[window_start:window_end, 0]
        market_window = market_lags[window_start:window_end, :]
        stock_window = stock_lags[window_start:window_end, :]
        
        # 模型1: 仅同期市场收益
        X0 = np.column_stack([np.ones(len(y_window)), market_window[:, 0]])
        
        # 模型2: 市场延迟模型（同期+滞后市场收益）
        X_market = np.column_stack([np.ones(len(y_window)), market_window])
        
        # 模型3: 自身延迟模型（同期市场收益+滞后自身收益）
        X_own = np.column_stack([np.ones(len(y_window)), market_window[:, 0], stock_window[:, 1:]])
        
        # 模型4: 全模型（同期+滞后市场收益+滞后自身收益）
        X_full = np.column_stack([np.ones(len(y_window)), market_window, stock_window[:, 1:]])
        
        # 计算R²
        try:
            # 计算R²的辅助函数
            def calculate_r2(X, y):
                if len(y) < 2:
                    return 0.0
                y_mean = np.mean(y)
                y_var = np.sum((y - y_mean)**2)
                if y_var == 0:
                    return 0.0
                # 最小二乘解
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = np.dot(X, coeffs)
                ess = np.sum((y_pred - y_mean)**2)
                return ess / y_var
            
            # 计算各个模型的R²
            r2_0 = calculate_r2(X0, y_window)
            r2_market = calculate_r2(X_market, y_window)
            r2_own = calculate_r2(X_own, y_window)
            r2_full = calculate_r2(X_full, y_window)
            
            # 计算延迟指标
            if r2_full > 0:
                d1_val = 1 - r2_0 / r2_full
            else:
                d1_val = np.nan
            
            if r2_market > 0:
                d2_val = 1 - r2_0 / r2_market
            else:
                d2_val = np.nan
            
            if r2_own > 0:
                d3_val = 1 - r2_0 / r2_own
            else:
                d3_val = np.nan
            
            # 存储结果
            date = stock_series.index[i]
            d1.loc[date] = d1_val
            d2.loc[date] = d2_val
            d3.loc[date] = d3_val
            
        except Exception as e:
            # 这里也不要使用logger，避免重复日志
            print(f"计算股票 {stock} 在日期 {stock_series.index[i-1]} 的延迟指标时出错: {e}")
            continue
    
    return d1, d2, d3

# 3. 计算延迟指标 - 批量处理版本
# def calculate_delay(stock_returns, market_returns, window=126, lag=10, half_life=63):
#     """计算日频延迟指标 - 批量处理多只股票"""
    
#     # 初始化结果
#     d1 = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)
#     d2 = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)
#     d3 = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)
    
#     logger.info(f"开始批量计算延迟指标 | {datetime.now()}")
    
#     # 批量处理每只股票
#     stock_list = stock_returns.columns.tolist()
#     total_stocks = len(stock_list)
    
#     for i, stock in enumerate(stock_list):
#         if (i + 1) % 100 == 0 or (i + 1) == total_stocks:
#             logger.info(f"处理进度: {i + 1}/{total_stocks} | {datetime.now()}")
        
#         try:
#             # 计算单只股票的延迟指标
#             d1_stock, d2_stock, d3_stock = calculate_stock_delay(stock, stock_returns,market_returns, window, lag)
            
#             # 存储结果
#             d1[stock] = d1_stock
#             d2[stock] = d2_stock
#             d3[stock] = d3_stock
            
#         except Exception as e:
#             logger.error(f"计算股票 {stock} 时出错: {e}")
#             continue
    
#     logger.info(f"批量计算完成 | {datetime.now()}")
#     return d1, d2, d3, {}

def init_worker(stock_returns, market_returns):
    global GLOBAL_STOCK_RETURNS, GLOBAL_MARKET_RETURNS
    GLOBAL_STOCK_RETURNS = stock_returns
    GLOBAL_MARKET_RETURNS = market_returns


def worker(stock, window, lag):
    try:
        return stock, calculate_stock_delay(
            stock,
            GLOBAL_STOCK_RETURNS,
            GLOBAL_MARKET_RETURNS,
            window,
            lag
        )
    except Exception as e:
        return stock, e

GLOBAL_STOCK_RETURNS = None
GLOBAL_MARKET_RETURNS = None

def calculate_delay(stock_returns, market_returns, window=126, lag=10, half_life=63):
    """计算日频延迟指标 - 修复Windows多进程版本"""
    
    # 初始化结果字典
    delay_dict = {}
    
    num_cores = max(1, cpu_count() - 4)
    logger.info(f"使用 {num_cores} 个核心进行并行计算 | {datetime.now()}")
    
    stock_list = stock_returns.columns.tolist()
    total_stocks = len(stock_list)
    
    results = []
    errors = {}
    
    with ProcessPoolExecutor(
        max_workers=num_cores,
        initializer=init_worker,
        initargs=(stock_returns, market_returns)
    ) as executor:
        
        futures = [
            executor.submit(worker, stock, window, lag)
            for stock in stock_list
        ]
        
        for i, future in enumerate(futures):
            stock = stock_list[i]
            try:
                result = future.result()
                
                if isinstance(result[1], Exception):
                    errors[stock] = result[1]
                    logger.error(f"计算股票 {stock} 时出错: {result[1]}")
                else:
                    d1_stock, d2_stock, d3_stock = result[1]
                    results.append((stock, d1_stock, d2_stock, d3_stock))
                
                if (i + 1) % 100 == 0 or (i + 1) == total_stocks:
                    logger.info(f"处理进度: {i + 1}/{total_stocks} | {datetime.now()}")
            
            except Exception as e:
                errors[stock] = e
                logger.error(f"计算股票 {stock} 时出错: {e}")
    
    # 收集结果到字典
    logger.info("整理结果数据...")
    # 首先创建日期到股票数据的映射
    date_data = {}
    for stock, d1_stock, d2_stock, d3_stock in results:
        for date in stock_returns.index:
            d1_val = d1_stock.loc[date]
            d2_val = d2_stock.loc[date]
            d3_val = d3_stock.loc[date]
            if pd.notna(d1_val) or pd.notna(d2_val) or pd.notna(d3_val):
                if date not in date_data:
                    date_data[date] = []
                date_data[date].append((stock, d1_val, d2_val, d3_val))
    
    # 转换为字典格式
    for date, stock_data in date_data.items():
        # 创建当天的DataFrame
        df = pd.DataFrame(stock_data, columns=['order_book_id', 'D1', 'D2', 'D3'])
        df.set_index('order_book_id', inplace=True)
        delay_dict[date] = df
    
    logger.info(f"计算完成 | {datetime.now()}")
    return delay_dict, errors



if __name__ == "__main__":
    # 主程序入口，只有主进程会执行这里的代码
    srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/stk_ret"
    desdir = "E:/SJTU/实习/国泰海通/barra因子/result/延迟alpha"
    os.makedirs(desdir, exist_ok=True)

    # 读取数据
    logger.info("开始读取数据...")
    ret_dict = pd.read_pickle(f"{srcdir}/全A_ret_24_2603D_dict.pkl")
    idx_df = pd.read_excel(f"{srcdir}/866011.RI_ret_24_2603D.xlsx",index_col=0)

    market_returns = idx_df['866011.RI']
    # 确保市场收益率数据的索引是日期类型
    market_returns.index = pd.to_datetime(market_returns.index)
    logger.info(f"数据读取完毕 | {datetime.now()}")
     
    # 转换数据格式
    # 1. 整理股票收益率数据
    all_dates = sorted(ret_dict.keys())
    start_date = datetime(2024, 1, 1)
    relevant_dates = [date for date in all_dates if date >= start_date]

    # 构建股票列表
    all_stocks = set()
    for date in relevant_dates:
        stocks = ret_dict[date].index.get_level_values(1).unique()
        all_stocks.update(stocks)
    all_stocks = list(all_stocks)
    logger.info(f"股票列表构建完毕 | {datetime.now()}")

    # 构建股票收益率DataFrame
    logger.info(f"开始构建股票收益率DataFrame | {datetime.now()}")
    # 优化：使用列表推导式收集数据，然后一次性构建DataFrame
    data = []
    for date in relevant_dates:
        if date in ret_dict:
            # 获取当天数据
            day_data = ret_dict[date].droplevel("date").squeeze()
            # 转换为适合构建DataFrame的格式
            for stock, ret in day_data.items():
                data.append([date, stock, ret])

    # 构建DataFrame并重塑
    temp_df = pd.DataFrame(data, columns=['date', 'stock', 'ret'])
    stock_returns_df = temp_df.pivot(index='date', columns='stock', values='ret')
    logger.info(f"股票收益率DataFrame构建完毕 | {datetime.now()}")

    # 5. 主计算
    logger.info("开始计算延迟指标...")
    delay_dict, errors = calculate_delay(stock_returns_df, market_returns, half_life=63)

    # 6. 保存结果
    logger.info("保存结果...")
    # 保存为pickle文件
    import pickle
    with open(f"{desdir}/delay_measures_2024_2026.pkl", 'wb') as f:
        pickle.dump(delay_dict, f)
    logger.info(f"保存字典格式结果到 {desdir}/delay_measures_2024_2026.pkl")

    # 可选：保存为Excel文件（如果需要）
    # 由于数据量可能很大，这里只保存前100天的数据作为示例
    sample_dates = list(delay_dict.keys())[:22]
    if sample_dates:
        with pd.ExcelWriter(f"{desdir}/延迟指标_2024_2026_sample.xlsx") as writer:
            for date in sample_dates:
                sheet_name = date.strftime('%Y-%m-%d')
                delay_dict[date].to_excel(writer, sheet_name=sheet_name)
        logger.info(f"保存示例数据到 {desdir}/延迟指标_2024_2026_sample.xlsx")

    logger.info("计算完成！")
    if errors:
        logger.warning(f"计算过程中出现 {len(errors)} 个错误")
