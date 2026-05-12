import pandas as pd
import numpy as np

def calculate_performance(nav_series, freq='D', output_path=None, rf_rate=0.015, bmk_nav=None):
    """
    计算回测业绩指标
    
    Parameters:
        nav_series (pd.Series): 产品净值数据，索引为日期
        freq (str): 频率参数，计算年化时使用：Y:252, W:52, M:12, D:252
        output_path (str): 输出Excel文件路径，可选
        rf_rate (float): 无风险利率，默认0.015
        bmk_nav (pd.Series): 基准净值数据，可选，用于计算超额收益指标
    
    Returns:
        pd.DataFrame: 业绩指标结果
    """
    freq_dict = {'Y': 252, 'W': 52, 'M': 12, 'D': 252}
    periods_per_year = freq_dict.get(freq, 252)
    
    results = []
    
    def calc_single_period_metrics(nav, period_factor, days_count):
        if len(nav) < 2:
            return {'年化收益': np.nan, '年化波动': np.nan, '夏普比率': np.nan, '最大回撤': np.nan, '卡玛比率': np.nan}
        
        total_return = nav.iloc[-1] / nav.iloc[0] 
        annualized_return = total_return ** (period_factor / days_count) - 1

        returns = nav.pct_change().dropna()
        volatility = returns.std() * np.sqrt(period_factor)
        sharpe_ratio = (annualized_return - rf_rate) / volatility if volatility != 0 else np.nan
        
        max_drawdown = (nav / nav.cummax() - 1).min()
        
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan
        
        return {
            '年化收益': annualized_return,
            '年化波动': volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '卡玛比率': calmar_ratio
        }
    
    def calc_yearly_metrics(nav):
        years = nav.index.year.unique()
        yearly_results = []
        
        for year in sorted(years):
            year_nav = nav[nav.index.year == year]
            if len(year_nav) < 2:
                continue
            
            prev_year = year - 1
            prev_year_nav = nav[nav.index.year == prev_year]
            
            if not prev_year_nav.empty:
                prev_last = prev_year_nav.iloc[-1]
                curr_last = year_nav.iloc[-1]
                year_return = curr_last / prev_last - 1
            else:
                year_return = (year_nav.iloc[-1] / year_nav.iloc[0]) - 1
            
            year_returns = year_nav.pct_change().dropna()
            volatility = year_returns.std() * np.sqrt(periods_per_year)
            sharpe_ratio = (year_return - rf_rate) / volatility if volatility != 0 else np.nan
            
            max_drawdown = (year_nav / year_nav.cummax() - 1).min() 
            calmar_ratio = year_return / max_drawdown if max_drawdown != 0 else np.nan
            
            yearly_results.append({
                '时间区间': f'{year}年',
                '年化收益': year_return,
                '年化波动': volatility,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown,
                '卡玛比率': calmar_ratio
            })
        
        return yearly_results
    
    nav = nav_series.dropna()
    if len(nav) >= 2:
        days_count = len(nav)
        
        overall = calc_single_period_metrics(nav, periods_per_year, days_count)
        overall['时间区间'] = '成立以来'
        results.append(overall)
        
        results.extend(calc_yearly_metrics(nav))
    
    if bmk_nav is not None:
        if len(bmk_nav) >= 2:

            common_dates = nav.index.intersection(bmk_nav.index)
            common_ret = nav[common_dates].pct_change()
            common_bmk_ret = bmk_nav[common_dates].pct_change()
            excess_ret = (common_ret - common_bmk_ret).dropna()
            excess_nav = (1 + excess_ret).cumprod()
            excess_nav = excess_nav / excess_nav.iloc[0]
            days_count = len(common_ret)
            print(f"共有的天数：{days_count}")
            
            excess_overall = calc_single_period_metrics(excess_nav, periods_per_year, days_count)
            excess_overall['时间区间'] = '成立以来(超额)'
            results.append(excess_overall)
            
            yearly_excess = calc_yearly_metrics(excess_nav)
            for item in yearly_excess:
                item['时间区间'] = item['时间区间'].replace('年', '年(超额)')
            results.extend(yearly_excess)
    
    result_df = pd.DataFrame(results)
    result_df = result_df[['时间区间', '年化收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率']]
    
    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"业绩指标已保存至: {output_path}")
    
    return result_df