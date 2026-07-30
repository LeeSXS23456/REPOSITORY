from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import pickle
from rqdatac import *
import pickle

ACPDIR = "E:/SJTU/intern/gtht/barra/data_base/index_component_日频"
ARTDIR = "E:/SJTU/intern/gtht/barra/data_base/stk_ret"

def update_Aret(end):

    df = pd.read_pickle(f"{ACPDIR}/866011.RI_19_26D_dict.pkl")
    df_r = pd.read_pickle(f"{ARTDIR}/全A_ret_19_26D_dict.pkl")

    dates = sorted(list(df.keys()))
    md = dates[-1]

    if pd.to_datetime(end) <= pd.to_datetime(md):
        return df_r
    
    temp = index_weights("866011.RI", start_date=md, end_date=end, market='cn')
    for dt in temp.index.get_level_values(0).unique():
        if dt in dates:
            continue
        df[dt] = temp.loc[dt]["weight"]
        stk = temp.loc[dt].index.tolist()
        stk_fb = [s for s in stk if not s.endswith(".BJSE")]
        df_ret = get_price_change_rate(stk_fb, start_date=dt, end_date=dt, expect_df=True, market='cn')
        df_r[dt] = df_ret.T[dt] 
    
    with open(f"{ACPDIR}/866011.RI_19_26D_dict.pkl", "wb") as f:
        pickle.dump(df, f)
    
    with open(f"{ARTDIR}/全A_ret_19_26D_dict.pkl", "wb") as f:
        pickle.dump(df_r, f)
    
    return df_r

def _daily_stats(dates, ret_dict):
    """日频：一次遍历计算 cs_vol + up/down/adr"""
    rows = []
    for d in dates:
        r = ret_dict[d].values
        up = (r > 0).mean()
        dn = (r < 0).mean()
        adr = up / dn if dn > 0 else np.nan
        rows.append({
            "date": d,
            "cs_vol": r.std(),
            "up_pct": up,
            "down_pct": dn,
            "adr": adr,
            "log_adr": np.log(adr) if (up > 0 and dn > 0) else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def _weekly_stats(dates, ret_dict):
    """周频：一次遍历（复合+统计）计算 cs_vol + up/down/adr + n_days"""
    weeks = defaultdict(list)
    for d in dates:
        weeks[d.isocalendar()[:2]].append(d)

    rows = []
    for iso, wk_dates in sorted(weeks.items()):
        prod = {}
        for d in wk_dates:
            for stk, r in ret_dict[d].items():
                prod[stk] = prod.get(stk, 1.0) * (1 + r)

        r = np.array([p - 1 for p in prod.values()])
        up = (r > 0).mean()
        dn = (r < 0).mean()
        adr = up / dn if dn > 0 else np.nan
        n = len(wk_dates)
        rows.append({
            "date": max(wk_dates),
            "cs_vol": r.std(),
            "up_pct": up,
            "down_pct": dn,
            "adr": adr,
            "log_adr": np.log(adr) if (up > 0 and dn > 0) else np.nan,
            "n_days": n,
            "short_week": n < 5,
        })
    return pd.DataFrame(rows).set_index("date")


def calc_cs_stats(ret_dict, freq="daily"):
    """
    一次遍历计算截面波动率 + 涨跌比 + ADR，避免重复循环。
    日期筛选由调用方负责（传入 ret_dict 前自行过滤，或对返回的 DataFrame 切片）。

    Parameters
    ----------
    ret_dict : dict {pd.Timestamp: pd.Series(order_book_id → return)}
    freq     : "daily" | "weekly"

    Returns
    -------
    pd.DataFrame  index=date
        cs_vol     : 截面波动率
        up_pct     : 上涨比例
        down_pct   : 下跌比例
        adr        : up / down
        n_days     : 本周交易天数 (仅 weekly)
        short_week : 交易天数 < 5    (仅 weekly)
    """
    dates = sorted(ret_dict.keys())
    if freq == "daily":
        return _daily_stats(dates, ret_dict)
    if freq == "weekly":
        return _weekly_stats(dates, ret_dict)
    raise ValueError(f"freq must be 'daily' or 'weekly', got {freq!r}")


def calc_vol_percentile(cs_vol, window=252):
    """
    滚动百分位排名：cs_vol[t] 在过去 window 个值中的分位数（含自身）。

    Parameters
    ----------
    cs_vol : pd.Series  date → cs_vol
    window : int  日频 252（≈1年），周频 52

    Returns
    -------
    pd.Series  date → pct_rank (0~1)
    """
    vals = cs_vol.values
    n = len(vals)
    pct = np.full(n, np.nan)
    for i in range(window - 1, n):
        hist = vals[i - window + 1 : i + 1]
        pct[i] = (hist <= vals[i]).mean()
    return pd.Series(pct, index=cs_vol.index, name="pct_rank")


def _fmt_dates(ax):
    """自适应密集日期标注：~18 刻度，短区间 %m-%d / 中区间 %Y-%m / 长区间 YYYYQn"""
    lo, hi = ax.get_xlim()
    days = (mdates.num2date(hi) - mdates.num2date(lo)).days
    months = max(1, int(days / 30.44))
    interval = max(1, months // 18)

    if months > 72:
        loc = mdates.MonthLocator(bymonth=(1, 4, 7, 10), bymonthday=1)
        fmt = FuncFormatter(
            lambda x, _: f"{mdates.num2date(x).year}Q{(mdates.num2date(x).month - 1) // 3 + 1}"
        )
    elif months <= 4:
        loc = mdates.MonthLocator()
        fmt = mdates.DateFormatter("%m-%d")
    else:
        loc = mdates.MonthLocator() if interval <= 1 else mdates.MonthLocator(interval=interval)
        fmt = mdates.DateFormatter("%Y-%m")

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(7.5)


def plot_vol_series(stats, pct_rank, freq="daily"):
    """
    三张独立图片：波动率（颜色编码）、历史排位、ADR。

    Returns
    -------
    dict  {"vol": fig, "rank": fig, "adr": fig}
    """
    x = stats.index
    common = dict(figsize=(15, 4), constrained_layout=True)

    # ===== Fig 1: 绝对波动（颜色编码排位）=====
    fig1, ax1 = plt.subplots(**common)
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(0, 1)
    y = stats["cs_vol"].values
    c = pct_rank.values

    for i in range(len(x) - 1):
        if not np.isnan(c[i]):
            ax1.fill_between(x[i:i+2], y[i:i+2], 0,
                             color=cmap(norm(c[i])), alpha=0.35, lw=0)
    ax1.plot(x, y, color="#333333", lw=0.9)
    ax1.set_ylabel("CS Volatility")
    ax1.set_title(f"Market Cross-Sectional Volatility ({freq})")
    ax1.set_ylim(0, None)
    ax1.grid(alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1, orientation="vertical", shrink=0.95, pad=0.02)
    cbar.set_label("Percentile Rank", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    _fmt_dates(ax1)

    # ===== Fig 2: 历史排位 =====
    fig2, ax2 = plt.subplots(**common)
    ax2.fill_between(x, pct_rank.values, alpha=0.12, color="#ff7f0e")
    ax2.plot(x, pct_rank.values, color="#ff7f0e", lw=1.2)
    for level, ls in [(0.25, ":"), (0.50, "--"), (0.75, ":")]:
        ax2.axhline(level, color="gray", ls=ls, lw=0.8)
    ax2.set_ylabel("Percentile Rank")
    ax2.set_title(f"Volatility Percentile Rank ({freq})")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)
    _fmt_dates(ax2)

    # ===== Fig 3: log ADR =====
    fig3, ax3 = plt.subplots(**common)
    ax3.fill_between(x, stats["log_adr"], alpha=0.12, color="#2ca02c")
    ax3.plot(x, stats["log_adr"], color="#2ca02c", lw=1.2)
    ax3.axhline(0, color="gray", ls="--", lw=0.8)
    ax3.set_ylabel("log ADR")
    ax3.set_title(f"Advance/Decline Ratio ({freq})")
    ax3.grid(alpha=0.25)
    _fmt_dates(ax3)

    return {"vol": fig1, "rank": fig2, "adr": fig3}


def main(ret_dict, freq="daily", start=None, end=None, lookback=None):
    """
    一键计算 + 画图。排位始终基于全样本，start/end 仅影响展示区间。

    Returns
    -------
    figs : dict  {"vol": fig, "rank": fig, "adr": fig}
    df   : pd.DataFrame  全量统计结果（未截断）
    """
    if lookback is None:
        lookback = 252 if freq == "daily" else 52

    df_full = calc_cs_stats(ret_dict, freq)
    pct_full = calc_vol_percentile(df_full["cs_vol"], window=lookback)

    df = df_full
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    pct = pct_full.loc[df.index]

    figs = plot_vol_series(df, pct, freq)
    return figs, df_full


if __name__ == "__main__":
    srcdir = "E:/SJTU/intern/gtht/barra/data_base/stk_ret"
    ret_dict = pd.read_pickle(f"{srcdir}/全A_ret_20_26D_dict.pkl")
    figs, stats = main(ret_dict, freq="daily", start="2021-01-01")
    plt.show()
