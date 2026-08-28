# -*- coding: utf-8 -*-
"""
2024.01.18~2024.06.25 量化超额回撤事件 - Barra 风格五维度一次性计算
"""
import pandas as pd
import numpy as np
import re
import sys, os

sys.path.insert(0, r"E:\SJTU\intern\gtht\barra")
sys.path.insert(0, r"E:\SJTU\intern\gtht\barra\comb")

# ========== 配置 ==========
start_date = pd.Timestamp("2024-01-18")
end_date   = pd.Timestamp("2024-06-25")
sd_padded  = start_date - pd.Timedelta(days=400)  # 波动率 rank 用

# ========== 1. 加载数据 ==========
df_full = pd.read_pickle(r"data_base/fac_ret/whole_mkt/factor_returns_10_2608.pkl")
df_full.index = pd.to_datetime(df_full.index)

style_cols = [c for c in df_full.columns
              if not re.search(r"[一-鿿]", str(c))
              and str(c).lower() != "comovement"]

print(f"风格因子共 {len(style_cols)} 个: {style_cols}")

df_padded = df_full[(df_full.index >= sd_padded) & (df_full.index <= end_date)]
df_view   = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]

print(f"\n区间: {start_date.date()} ~ {end_date.date()}, {len(df_view)} 个交易日")

# ========== 2. 回测指标 ==========
nav = (1 + df_view[style_cols]).cumprod()
nav = nav / nav.iloc[0]
n_days = len(df_view)
yf = n_days / 252

cum_ret = nav.iloc[-1] - 1
ann_ret = nav.iloc[-1] ** (1/yf) - 1
ann_vol = df_view[style_cols].std() * np.sqrt(252)
sharpe  = ann_ret / ann_vol.replace(0, np.nan)

rmax = nav.cummax()
dd   = (nav - rmax) / rmax
max_dd = dd.min()
calmar = ann_ret / (-max_dd).replace(0, np.nan)
win_rate = (df_view[style_cols] > 0).mean()

bt = pd.DataFrame({
    "累计收益%": (cum_ret*100).round(2),
    "年化收益%": (ann_ret*100).round(2),
    "年化波动%": (ann_vol*100).round(2),
    "夏普": sharpe.round(3),
    "最大回撤%": (max_dd*100).round(2),
    "卡玛": calmar.round(3),
    "胜率%": (win_rate*100).round(1),
}).sort_values("累计收益%", ascending=False)

print("\n" + "="*80)
print("=== 回测指标（按累计收益降序）===")
print("="*80)
print(bt.to_string())

# 分阶段收益演变（三等分，标注具体日期）
print("\n" + "="*80)
print("=== 分阶段累计收益演变（前/中/后各1/3）===")
print("="*80)
n = len(df_view)
t1 = n // 3
t2 = 2 * n // 3
phase_info = [
    ("初期", df_view.iloc[:t1]),
    ("中期", df_view.iloc[t1:t2]),
    ("末期", df_view.iloc[t2:]),
]
for name, sub in phase_info:
    sd, ed = sub.index[0].date(), sub.index[-1].date()
    print(f"  {name}: {sd} ~ {ed} ({len(sub)}天)")

phase_cum = {}
for name, sub in phase_info:
    sub_nav = (1 + sub[style_cols]).cumprod()
    sub_nav = sub_nav / sub_nav.iloc[0]
    phase_cum[name] = (sub_nav.iloc[-1] - 1) * 100
phase_df = pd.DataFrame(phase_cum).round(2)
phase_df = phase_df.sort_values("初期", ascending=False)
print(phase_df.to_string())

# ========== 2.5 三大价差 ==========
print("\n" + "="*80)
print("=== 三大价差（累计收益）===")
print("="*80)

spread_gv = (df_view["growth"] + df_view["momentum"]) / 2 - (df_view["book_to_price"] + df_view["earnings_yield"]) / 2
spread_size = df_view["size"] - df_view["non_linear_size"]
spread_sv = (df_view["beta"] + df_view["liquidity"]) / 2 - df_view["residual_volatility"]

spreads = pd.DataFrame({
    "成长-价值(spread_gv)": spread_gv,
    "大小盘(spread_size)": spread_size,
    "系统弹性-特质波动(spread_sv)": spread_sv,
})
spreads_nav = (1 + spreads).cumprod()
spreads_nav = spreads_nav / spreads_nav.iloc[0]
spreads_cum = (spreads_nav.iloc[-1] - 1) * 100
print(spreads_cum.round(2).to_string())

# ========== 3. 波动率（10日滚动，年化，250d 分位）==========
print("\n" + "="*80)
print("=== 风格波动率（10日滚动年化，250日历史分位）===")
print("="*80)

vol = df_padded[style_cols].rolling(10).std() * np.sqrt(252)
rank_250 = vol.rolling(250, min_periods=250).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

vol_v  = vol[vol.index >= start_date]
rank_v = rank_250[rank_250.index >= start_date]

print("\n--- 期末波动率水平 ---")
end_vol = pd.DataFrame({
    "年化波动%": (vol_v.iloc[-1]*100).round(2),
    "250d分位": rank_v.iloc[-1].round(3),
}).sort_values("年化波动%", ascending=False)
print(end_vol.to_string())

# 横截面
thr = 0.8
high_mask = rank_v >= thr
high_days = pd.DataFrame({
    "高位天数": high_mask.sum().astype(int),
    "占比%": (high_mask.mean()*100).round(1),
}).sort_values("高位天数", ascending=False)

all_high  = int(high_mask.all(axis=1).sum())
half_high = int((high_mask.sum(axis=1) >= 6).sum())
avg_rank  = rank_v.mean(axis=1)
cross_std = rank_v.std(axis=1)

print(f"\n--- 波动率横截面（阈值={thr*100:.0f}%分位）---")
print(f"全因子同时高位: {all_high} / {len(vol_v)} ({all_high/len(vol_v)*100:.1f}%)")
print(f"6个以上因子高位: {half_high} / {len(vol_v)} ({half_high/len(vol_v)*100:.1f}%)")
print(f"平均波动率分位: 均值={avg_rank.mean():.3f}, 期末={avg_rank.iloc[-1]:.3f}")
print(f"因子间离散度(分位std): 均值={cross_std.mean():.3f}")

print(f"\n--- 各因子高位天数（{int(thr*100)}%分位以上）---")
print(high_days.to_string())

# 波动率分位区间分布
print("\n--- 各因子波动率分位区间分布（天数）---")
bins = [0, 0.3, 0.5, 0.7, 0.8, 1.0]
labels = ["<30%低位", "30-50%中低", "50-70%中高", "70-80%高位", ">80%极端"]
dist_df = pd.DataFrame()
for c in style_cols:
    dist_df[c] = pd.cut(rank_v[c], bins=bins, labels=labels).value_counts().reindex(labels)
print(dist_df.T.to_string())

# ========== 4. 相关性 ==========
print("\n" + "="*80)
print("=== 因子相关性 ===")
print("="*80)

pearson  = df_view[style_cols].corr('pearson')
spearman = df_view[style_cols].corr('spearman')

def top_pairs(corr, n=15):
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i,j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]

print("\n--- Pearson 相关 Top 15 ---")
for a,b,r in top_pairs(pearson, 15):
    print(f"  {a:22s} vs {b:22s}: {r:+.3f}")

print("\n--- Spearman 秩相关 Top 15 ---")
for a,b,r in top_pairs(spearman, 15):
    print(f"  {a:22s} vs {b:22s}: {r:+.3f}")

# 分阶段相关性变化
print("\n--- 分阶段关键因子对相关性变化 ---")
key_pairs = [
    ("size", "non_linear_size"),
    ("growth", "book_to_price"),
    ("growth", "momentum"),
    ("beta", "residual_volatility"),
    ("size", "earnings_yield"),
    ("liquidity", "residual_volatility"),
    ("book_to_price", "leverage"),
]
for pname, sub in phase_info:
    pcorr = sub[style_cols].corr('pearson')
    print(f"\n  {pname} 关键因子对:")
    for a, b in key_pairs:
        print(f"    {a:22s} vs {b:22s}: {pcorr.loc[a,b]:+.3f}")

# ========== 5. 特质收益波动率 ==========
print("\n" + "="*80)
print("=== 特质收益波动率（选股端 Alpha 环境观测）===")
print("="*80)

from helpfunc_specificr import vol_pipeline
spe_summary = vol_pipeline(str(start_date.date()), str(end_date.date()),
                           vol_window=10, lookback_days=242, auto_update=False)

sv_avg = spe_summary["avg_vol"]
sr_avg = spe_summary["avg_vol_rank"]

print(f"\n平均特质波动(均值): {sv_avg.mean()*100:.4f}%")
print(f"平均特质波动(峰值): {sv_avg.max()*100:.4f}% ({spe_summary.loc[sv_avg.idxmax(),'date'].date()})")
print(f"平均特质波动(期末): {sv_avg.iloc[-1]*100:.4f}%")
print(f"平均历史分位(均值): {sr_avg.mean():.3f}")
print(f"平均历史分位(峰值): {sr_avg.max():.3f} ({spe_summary.loc[sr_avg.idxmax(),'date'].date()})")
print(f"平均历史分位(期末): {sr_avg.iloc[-1]:.3f}")

print("\n--- 分位区间天数分布 ---")
for thr in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    d = (sr_avg >= thr).sum()
    r = d / len(spe_summary) * 100
    print(f"  分位>={thr*100:.0f}%天数: {d}/{len(spe_summary)} ({r:.1f}%)")

# 高波动平台期（连续>=3天，分位>70%）
high_mask_spe = sr_avg.values > 0.7
segments = []
in_seg = False
seg_start = 0
for i, h in enumerate(high_mask_spe):
    if h and not in_seg:
        in_seg = True; seg_start = i
    elif not h and in_seg:
        in_seg = False
        if i - seg_start >= 3:
            segments.append((seg_start, i-1))
if in_seg and len(high_mask_spe) - seg_start >= 3:
    segments.append((seg_start, len(high_mask_spe)-1))

print(f"\n高波动平台期(>=3天, 分位>70%): {len(segments)} 段")
for s, e in segments:
    seg = spe_summary.iloc[s:e+1]
    peak_idx = seg["avg_vol_rank"].idxmax()
    print(f"  {seg.iloc[0]['date'].date()} ~ {seg.iloc[-1]['date'].date()}"
          f" ({e-s+1}天), 平均分位={seg['avg_vol_rank'].mean():.3f}"
          f", 峰值={seg.loc[peak_idx,'avg_vol_rank']:.3f}({seg.loc[peak_idx,'date'].date()})")

# 特质波动率激增日（日增长 > 5%）
spe_summary["vol_pct_change"] = spe_summary["avg_vol"].pct_change()
spike_days = spe_summary[spe_summary["vol_pct_change"] > 0.05].copy()
print(f"\n特质波动率激增日(日增>5%): {len(spike_days)} 天")
if len(spike_days) > 0:
    for _, row in spike_days.iterrows():
        print(f"  {row['date'].date()}: +{row['vol_pct_change']*100:.1f}%, "
              f"分位={row['avg_vol_rank']:.3f}")

# ========== 6. 风格切换检测 ==========
print("\n" + "="*80)
print("=== 风格切换事件 ===")
print("="*80)

from detect_regime_switches import run_detection

detect_start = start_date - pd.Timedelta(days=120)
events_df, diag_df = run_detection(start=detect_start, end=end_date)

events_in = events_df[
    (events_df["start_date"] >= start_date) & (~events_df["is_overlapping"])
].copy()

print(f"区间内独立事件数: {len(events_in)} (已去重叠)")
if not events_in.empty:
    cols = ["event_id", "start_date", "confirm_date", "end_date",
            "direction", "regime_before", "regime_after",
            "dominant_spread", "strengthening_factors", "confidence_level"]
    cols = [c for c in cols if c in events_in.columns]
    print(events_in[cols].to_string(index=False))

print(f"\n总事件数(含重叠): {len(events_df[events_df['start_date'] >= start_date])}")

# 保存关键数据供后续画图
print("\n" + "="*80)
print("=== 数据保存 ===")
print("="*80)

# 保存净值数据
nav.to_pickle(r"comb\outputs\2024h1_dd_nav.pkl")
spe_summary.to_pickle(r"comb\outputs\2024h1_dd_spe_vol.pkl")
rank_v.to_pickle(r"comb\outputs\2024h1_dd_vol_rank.pkl")
vol_v.to_pickle(r"comb\outputs\2024h1_dd_vol.pkl")

print("数据已保存到 comb/outputs/")
print("\n=== 计算全部完成 ===")
