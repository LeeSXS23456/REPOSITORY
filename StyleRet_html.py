import os, re
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "data_base", "fac_ret", "whole_mkt", "factor_returns_20_2603.pkl")


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RQ_OK = False
try:
    import rqdatac
    rqdatac.init(
    username = "license",
    password = "ZZ-u7ZWosqrntc3VY3TJzJLPsb-A0o4zehYoiNpDvIBXiwvRIOUmFe7medtMhwu4qiaNxqFSc6ONdGcGeVYgUVd-w5QKScPkmzBEmYVEt94lz9sQZoHwdtQXWWRGGrJqtr7ehiQACydlPS7RcPBfJrpyeTJFsGF1E1guZbpLnvU=XouX9YSi7Pcyo0rSLCMydvHs3nrVq6Rwjda-jI9H_gfGlp53ot0ZnIA6g-ZtvwPDAb62K38pHIqYYyTAyER7FBtZ5HumXzOrWW42LHpUn5-vbnLMxiwbimJ9ns41CaMbjpFEgNcfO52l5wiqDqFCkZNy_OKSDjepfa9GxHsLZZE="
)
    RQ_OK = True
except Exception as _rq_err:
    st.sidebar.error(f"rqdatac 初始化失败: {_rq_err}")


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_pickle(CACHE_FILE)
        if isinstance(df, pd.DataFrame):
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df.sort_index()
    except:
        pass
    return pd.DataFrame()


def save_cache(df):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    pd.to_pickle(df, CACHE_FILE)


def fetch_from_api(start, end):
    from rqdatac import get_factor_return
    raw = get_factor_return(start, end, factors=None, universe="whole_market",
                            method="implicit", industry_mapping="sws_2021",
                            model="v1", market="cn")
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_index()


def get_data(sd, ed):
    _d = []
    df = load_cache()
    _d.append(f"缓存区间: {df.index.min().date()} ~ {df.index.max().date()}" if not df.empty else "缓存为空")

    if not df.empty:
        c_min, c_max = df.index.min(), df.index.max()
        _d.append(f"缓存区间: {c_min.date()} ~ {c_max.date()}")
        _d.append(f"sd >= c_min? {sd >= c_min}  |  ed <= c_max? {ed <= c_max}")
        if (sd >= c_min) and (ed <= c_max):
            _d.append("缓存完全覆盖 → 直接返回")
            return df, _d
        _d.append("缓存不足 → 需增量更新")
    else:
        _d.append("缓存为空")

    _d.append(f"RQ_OK (API可用)? {RQ_OK}")
    if not RQ_OK:
        _d.append("API不可用 → 返回现有缓存")
        return df if not df.empty else pd.DataFrame(), _d

    if not df.empty:
        if ed > c_max:
            fetch_start = c_max 
            fetch_end = ed 
        elif sd < c_min:
            fetch_start = sd 
            fetch_end = c_min 
        else:
            return df, _d
    else:
        fetch_start = sd 
        fetch_end = ed 
    _d.append(f"计划拉取: {fetch_start.date()} ~ {fetch_end.date()}")
    st.info(f"📡 正在提取新数据 ({fetch_start.date()} ~ {fetch_end.date()})…")
    try:
        new = fetch_from_api(fetch_start.strftime("%Y%m%d"), fetch_end.strftime("%Y%m%d"))
    except Exception as e:
        _d.append(f"API 异常: {e}")
        return df, _d

    _d.append(f"API 返回行数: {len(new)}")
    if new.empty:
        _d.append("API 返回空")
        return df, _d

    if not df.empty:
        df = pd.concat([df, new]).drop_duplicates(keep="last").sort_index()
        _d.append(f"合并后行数: {len(df)}")
    else:
        df = new.sort_index()
        _d.append(f"首次拉取行数: {len(df)}")
    try:
        save_cache(df)
        _d.append("缓存已保存 ✅")
    except Exception as e:
        _d.append(f"缓存写入失败: {e}")
    return df, _d


st.set_page_config(page_title="Barra 因子净值", layout="wide")
st.markdown("""
<style>
div[data-testid="metric-container"] label { font-size: 0.7rem !important; white-space: nowrap !important; }
div[data-testid="metric-container"] div { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)
st.title("Barra 因子净值可视化")
st.sidebar.header("配置")
st.session_state.sd = st.sidebar.date_input("起始", pd.Timestamp("2020-01-02"), max_value=pd.Timestamp("2036-03-25"))
st.session_state.ed = st.sidebar.date_input("结束", pd.Timestamp("2026-03-25"), max_value=pd.Timestamp("2036-03-25"))
mode = st.sidebar.radio("模式", ["大类综合", "单因子详细"])

sd = pd.Timestamp(st.session_state.sd)
ed = pd.Timestamp(st.session_state.ed)
with st.spinner("加载数据中..."):
    df_full, debug_log = get_data(sd, ed)
if df_full.empty:
    st.error("无可用数据")
    st.stop()

df_view = df_full[(df_full.index >= sd) & (df_full.index <= ed)]
if df_view.empty:
    st.error("区间无数据")
    st.stop()

cols = [c for c in df_full.columns if str(c).lower() != "comovement"]
style_cols = [c for c in cols if not re.search(r"[\u4e00-\u9fff]", str(c))]
industry_cols = [c for c in cols if re.search(r"[\u4e00-\u9fff]", str(c))]

c1, c2, c3, c4 = st.columns(4)
c1.metric("开始时间", f"{df_view.index.min().date()}")
c2.metric("结束时间", f"{df_view.index.max().date()}")
c3.metric("交易日", len(df_view))
c4.metric("因子", f"风格{len(style_cols)} / 行业{len(industry_cols)}")

if mode == "大类综合":
    cat = st.radio("类别", ["风格因子", "行业因子"], horizontal=True)
    target = style_cols if cat == "风格因子" else industry_cols
    if not target:
        st.stop()
    nav = (df_view[target] + 1).cumprod()
    order = nav.iloc[-1].sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in order:
        ax.plot(nav.index, nav[c], label=str(c), lw=1.2)
    ax.axhline(1, color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.set_title(f"{cat} 区间净值")

    ax.legend(loc="upper left", bbox_to_anchor=(-0.15, 1), fontsize=7.5, ncol=1)
    # 用tight_layout自动适配图例，避免手动设置left
    plt.tight_layout(rect=[0.15, 0, 1, 1])

    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

else:
    sub_cat = st.radio("类型", ["风格因子", "行业因子"], horizontal=True)
    pool = style_cols if sub_cat == "风格因子" else industry_cols
    factor = st.selectbox("因子", pool) if pool else st.stop()
    ret = df_view[factor].dropna()
    nav = (ret + 1).cumprod()
    ref = df_full[(df_full.index >= pd.Timestamp("2020-01-02")) & (df_full.index <= ed)][factor].dropna()
    ref_vals = ref.values
    pct = ret.apply(lambda x: float((ref_vals < x).sum()) / len(ref_vals) * 100)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(nav.index, nav, color="#1f77b4", lw=2, zorder=5)
    ax.axhline(1, color="gray", ls="--", lw=0.6, alpha=0.6)
    y_pad = (nav.max() - nav.min()) * 0.08
    ax.set_ylim(nav.min() - y_pad, nav.max() + y_pad)

    ax2 = ax.twinx()
    ax2.scatter(pct.index, pct, color="#f39c12", s=10, alpha=0.55, label="日收益分位数(%)")
    ax2.axhline(50, color="#f39c12", ls=":", lw=0.8, alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("分位数(%)", color="#f39c12", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#f39c12")

    ax.set_title(f"{factor}", fontsize=14, fontweight="bold")
    ax.set_ylabel("净值", fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best", framealpha=0.7)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("📋 数据加载日志"):
    for line in debug_log:
        st.markdown(f"- {line}")