from rqdatac import *
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
srcdir = os.path.join(BASE_DIR, "data_base", "basis","index_future_basics.pkl")

epsdir = os.path.join(BASE_DIR, "data_base", "basis","866011.RI_eps_24_25Q.pkl")
#df_eps = pd.read_pickle(epsdir)
dpsdir = os.path.join(BASE_DIR, "data_base", "basis","866011.RI_dps_24_25H.pkl")
#df_dps = pd.read_pickle(dpsdir)
timedir = os.path.join(BASE_DIR, "data_base", "basis","dividend_timeline.pkl")
#df_time = pd.read_pickle(timedir)
alldir = os.path.join(BASE_DIR, "data_base", "index_component_日频","866011.RI_20_26D_dict.pkl")
all_df = pd.read_pickle(alldir)
all_ids = list(all_df.values())[-1].index.tolist()

def add_basis_data(st,ed):
    df_future = all_instruments(type='Future')
    df_index = df_future[df_future["product"]=="Index"]
    df_index_real = df_index[df_index["maturity_date"]!="0000-00-00"]
    df_index_real.to_pickle(srcdir)

    contracts = df_index_real["order_book_id"].tolist()
    df_info = futures.get_basis(contracts, start_date=st, end_date=ed, fields=["settlement","close_index"], frequency='1d', dividend_adjusted=False, market='cn')
    df_info = df_info.reset_index(level=1)

    # 计算基础指标
    df_info["basis"] = df_info["settlement"] - df_info["close_index"]
    df_info["abs_ratio"] = df_info["basis"] / df_info["close_index"]

    df_index_real.set_index(["order_book_id"], inplace=True)
    df_info_m = df_info.merge(df_index_real[["listed_date","maturity_date"]], on=["order_book_id"], how="left")
    # 统一转为日期格式
    df_info_m["maturity_date"] = pd.to_datetime(df_info_m["maturity_date"])
    df_info_m["date"] = pd.to_datetime(df_info_m["date"])
    # 再计算间隔天数
    df_info_m["residual_day"] = np.where(
        (df_info_m["maturity_date"] - df_info_m["date"]).dt.days == 0,
        np.nan,
        (df_info_m["maturity_date"] - df_info_m["date"]).dt.days
    )

    df_info_m["ana_cost"] = df_info_m["abs_ratio"] / df_info_m["residual_day"] * 365

    return df_info_m

def pre_dividend_payratio(quarter:str,stk:str):
    """
    计算上一年的分红支付率
    """
    pre_quarter = str(int(quarter[:4]) - 1) + quarter[4:]

    #提取上一年的eps
    try:
        df_eps_pre = df_eps.loc[(stk, pre_quarter)]
        eps_pre = df_eps_pre["basic_earnings_per_share"].values[0]
    except:
        eps_pre = np.nan

    #提取上一年的分红金额
    try:
        df_dps_stk = df_dps[df_dps.index.get_level_values(0)==stk]
        df_dps_pre = df_dps_stk[df_dps_stk["quarter"]==pre_quarter]
        dps_pre = df_dps_pre["dividend_cash_before_tax"].values[0] / 10 * 0.9 #税后每股分红金额
    except:
        dps_pre = np.nan

    dividend_payratio = dps_pre / eps_pre

    return dividend_payratio if dividend_payratio else 0
    
def get_eps(quarter:str,stk:str):
    """
    获取指定报告期的eps
    """
    try:
        df_eps_pre = df_eps.loc[(stk, quarter)]
        eps = df_eps_pre["basic_earnings_per_share"].values[0]
    except:
        eps = np.nan
    return eps

def active_contract(dt):
    """
    获取当前日期后的所有活跃合约，包含合约代码和到期日期
    """
    df_index_real_new = pd.read_pickle(srcdir)
    df_index_real_new = df_index_real_new[df_index_real_new["maturity_date"] > dt]
    active_df = df_index_real_new[["order_book_id","maturity_date"]]
    return active_df

def get_info_d(c_id,dt):
    """
    获取指定合约在指定日期的成分股信息,包含成分股代码，成分股权重，成分股价格
    """
    # 合约前缀 -> 指数代码映射  IC:000905 IM:000852 IH:000016 IF:000300
    _prefix = str(c_id)[:2].upper()
    _idx_map = {"IC": "000905", "IM": "000852", "IH": "000016", "IF": "000300"}
    if _prefix not in _idx_map:
        return pd.DataFrame()
    _idx = _idx_map[_prefix] + ".XSHG"

    _d_str = pd.Timestamp(dt).strftime("%Y%m%d")
    df = index_weights_ex(_idx, start_date=_d_str, end_date=_d_str, market="cn").reset_index(level=0)

    ids = df.index.get_level_values(1).tolist() #dt天成分股
    df_price = get_price(ids, start_date=_d_str, end_date=_d_str, frequency='1d', fields=["close"], adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn').reset_index(level=1)
    df = df.merge(df_price, on=["order_book_id"], how="left")

    return ids, df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def get_index_d(dt):
    """
    获取指定日期的指数收盘价格
    """
    df_index = get_price(["000016.XSHG","000300.XSHG", "000905.XSHG","000852.XSHG"], start_date=dt, end_date=dt, frequency='1d', fields=["close"], adjust_type='pre', skip_suspended=False, expect_df=True, time_slice=None, market='cn')
    return df_index

def update_dps(dt):
    _dps_old = pd.read_pickle(dpsdir) if os.path.exists(dpsdir) else pd.DataFrame()
    _date_col = "ex_dividend_date"
    # 确定增量起点
    if not _dps_old.empty and _date_col is not None:
        _start = pd.Timestamp(_dps_old[_date_col].max()) + pd.Timedelta(days=1)
    else:
        _start = pd.Timestamp("2024-06-30")
    _end = pd.Timestamp(dt)
    if _start <= _end:
        _dps_new = get_dividend(all_ids,start_date=_start.strftime("%Y%m%d"),end_date=_end.strftime("%Y%m%d"),expect_df=True, market='cn')
        if isinstance(_dps_new, pd.DataFrame) and not _dps_new.empty:
            _dps_old = pd.concat([_dps_old, _dps_new], axis=0).drop_duplicates().sort_index()
    #存储
    _dps_old.to_pickle(dpsdir)
    return _dps_old

def update_eps(end_q):
    #增量更新 eps：读老数据 → 取老数据最大 quarter → 从下一季度拉到 end_q → 合并去重存回
    _eps_old = pd.read_pickle(epsdir) if os.path.exists(epsdir) else pd.DataFrame()

    _start_q = "2025q4"
    _new = get_pit_financials_ex(all_ids, ["basic_earnings_per_share"],start_quarter=_start_q, end_quarter=end_q,date=None, statements='latest', market='cn')

    if isinstance(_new, pd.DataFrame) and not _new.empty:
        _eps_old = pd.concat([_eps_old, _new], axis=0).drop_duplicates().sort_index()
        _eps_old.to_pickle(epsdir)

    return _eps_old

def update_timeline(end_q):
    #增量更新 分红时间时间线：读老数据 → 取老数据最大 quarter → 从下一季度拉到 end_q → 合并去重存回
    _time_old = pd.read_pickle(timedir) if os.path.exists(timedir) else pd.DataFrame()

    _start_q = "2025q4"
    _new = get_dividend_amount(all_ids, start_quarter = _start_q, end_quarter = end_q, date = None, market = 'cn')

    if isinstance(_new, pd.DataFrame) and not _new.empty:
        _time_old = pd.concat([_time_old, _new], axis=0).drop_duplicates().sort_index()
        _time_old.to_pickle(timedir)

    return _time_old

def cal_fhds(dt,new):

    #基础数据准备
    active_df = active_contract(dt)
    contracts = active_df["order_book_id"].tolist()
    origin_df = new[new.index.isin(contracts)] #new是之前更新过的基差面板df_info

    #判断有影响的报告期：dt 前一年 q2 / q4，当年 q2
    _y = pd.Timestamp(dt).year
    quarter_list = [f"{_y - 1}q2", f"{_y - 1}q4", f"{_y}q2"]

    #增量更新xps数据：读老数据 → 取老数据最新日期 → 只拉增量 → 合并去重 → 存回
    _dps = update_dps(dt)
    _eps = update_eps(quarter_list[-1])
    _time = update_timeline(quarter_list[-1])

    #









