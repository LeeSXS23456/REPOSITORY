"""比较两份 Barra 风格暴露数据的异同"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 解决 matplotlib 中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

srcdir = "E:/SJTU/intern/gtht/barra/通联数据源/因子暴露"

# ========== 配置 ==========
# 通联数据源：'csv' 读 barra_exp.csv / 'pkl' 读按日pkl目录
tl_source   = 'pkl'
base_pkl    = r'E:\SJTU\intern\gtht\barra\通联数据源\因子暴露\格式对齐_nsize'
nsize_version = 'size3'   # 'size3' / 'lnmv3'
include_bjse  = False      # 是否包含北交所
reg_weighted  = True     # 回归是否加权
tag           = 'nsize自算'   # 输出前缀标识

# 自动拼接 pkl 目录
bjse_tag = 'withBJ' if include_bjse else 'noBJ'
w_tag    = 'wls' if reg_weighted else 'ols'
tl_pkl_dir = f'{base_pkl}/{nsize_version}_{bjse_tag}_{w_tag}'

# 输出tag自动带上参数
if tag:
    tag = f'{tag}_{nsize_version}_{bjse_tag}_{w_tag}'

# 日期范围（只对 pkl 模式生效，设为 None 表示全量）
start_date = '2026-08-10'
end_date   = '2026-08-14'

outdir = os.path.join(srcdir, "comparison_outputs")
os.makedirs(outdir, exist_ok=True)

# ========== 1. 读取并标准化数据 ==========
# 因子名称映射 (rq 列名 -> 统一列名)
factor_map = {
    "beta": "BETA",
    "momentum": "MOMENTUM",
    "size": "SIZE",
    "earnings_yield": "EARNYILD",
    "residual_volatility": "RESVOL",
    "growth": "GROWTH",
    "book_to_price": "BTOP",
    "leverage": "LEVERAGE",
    "liquidity": "LIQUIDTY",
    "non_linear_size": "SIZENL",
}
factors = list(factor_map.values())

# 米筐/RQ 版 (rq_barra_exp.csv)
df2 = pd.read_csv(os.path.join(srcdir, "rq_barra_exp_demo.csv"))
df2["date"] = pd.to_datetime(df2["date"])
df2["stock"] = df2["order_book_id"].astype(str).str.split(".").str[0].str.zfill(6)
df2 = df2.rename(columns=factor_map)[["date", "stock"] + factors].copy()

# 通联版
if tl_source == 'csv':
    df1 = pd.read_csv(os.path.join(srcdir, "barra_exp.csv"))
    df1["date"] = pd.to_datetime(df1["TRADE_DATE"].astype(str), format="%Y%m%d")
    df1["stock"] = df1["TICKER_SYMBOL"].astype(str).str.zfill(6)
    df1 = df1[["date", "stock"] + factors].copy()
else:
    # 从按日 pkl 目录读取
    files = sorted(os.listdir(tl_pkl_dir))
    if start_date:
        files = [f for f in files if f[:10] >= start_date]
    if end_date:
        files = [f for f in files if f[:10] <= end_date]
    rows = []
    for f in files:
        d = pd.read_pickle(os.path.join(tl_pkl_dir, f))
        d["date"] = pd.to_datetime(f[:10])
        d["stock"] = d["order_book_id"].astype(str).str.split(".").str[0].str.zfill(6)
        # 小写列名 -> 统一大写因子名
        d = d.rename(columns=factor_map)
        rows.append(d[["date", "stock"] + factors])
    df1 = pd.concat(rows, ignore_index=True)

# 对齐日期范围（取交集）
common_dates = set(df1["date"]) & set(df2["date"])
df1 = df1[df1["date"].isin(common_dates)].copy()
df2 = df2[df2["date"].isin(common_dates)].copy()

# 输出文件名前缀
prefix = f"{tag}_" if tag else ""

print(f"通联版: {len(df1)} 行, {df1['date'].nunique()} 个交易日, {df1['stock'].nunique()} 只股票")
print(f"米筐版: {len(df2)} 行, {df2['date'].nunique()} 个交易日, {df2['stock'].nunique()} 只股票")
print(f"共同交易日: {len(set(df1['date']) & set(df2['date']))}")
print(f"共同股票数: {len(set(df1['stock']) & set(df2['stock']))}")

# ========== 2. 合并对齐 ==========
merged = df1.merge(df2, on=["date", "stock"], how="inner", suffixes=("_TLY", "_RQ"))
print(f"\n对齐后共同样本: {len(merged)} 行, {merged['date'].nunique()} 个交易日, {merged['stock'].nunique()} 只股票")

# ========== 3. 逐因子差异统计 ==========
diff_stats = []
for f in factors:
    diff = merged[f + "_TLY"] - merged[f + "_RQ"]
    corr = merged[f + "_TLY"].corr(merged[f + "_RQ"])
    diff_stats.append({
        "factor": f,
        "mean_diff": diff.mean(),
        "std_diff": diff.std(),
        "mean_abs_diff": diff.abs().mean(),
        "max_abs_diff": diff.abs().max(),
        "correlation": corr,
        "pct_within_01": (diff.abs() < 0.1).mean() * 100,
        "pct_within_05": (diff.abs() < 0.5).mean() * 100,
    })
diff_stats_df = pd.DataFrame(diff_stats).sort_values("correlation")
print("\n===== 各因子差异统计（通联 - 米筐） =====")
print(diff_stats_df.to_string(index=False, float_format="%.4f"))
diff_stats_df.to_csv(os.path.join(outdir, f"{prefix}factor_diff_stats.csv"), index=False, encoding="utf-8-sig")

# ========== 4. 每个交易日各风格因子分布对比 ==========
# 计算每日的均值、标准差
daily_stats_tly = df1.groupby("date")[factors].agg(["mean", "std"])
daily_stats_rq = df2.groupby("date")[factors].agg(["mean", "std"])

# 绘制每日均值对比（分两排）
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
for i, f in enumerate(factors):
    ax = axes[i // 2, i % 2]
    daily_stats_tly[f]["mean"].plot(ax=ax, label="通联 mean", alpha=0.8)
    daily_stats_rq[f]["mean"].plot(ax=ax, label="米筐 mean", alpha=0.8, linestyle="--")
    ax.set_title(f"{f} 每日均值对比")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{prefix}daily_mean_comparison.png"), dpi=100)
plt.close()

# 每日标准差对比
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
for i, f in enumerate(factors):
    ax = axes[i // 2, i % 2]
    daily_stats_tly[f]["std"].plot(ax=ax, label="通联 std", alpha=0.8)
    daily_stats_rq[f]["std"].plot(ax=ax, label="米筐 std", alpha=0.8, linestyle="--")
    ax.set_title(f"{f} 每日标准差对比")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"{prefix}daily_std_comparison.png"), dpi=100)
plt.close()

# ========== 5. 个股因子暴露时序走势（举例） ==========
# 选几只有代表性的股票
sample_stocks = ["000001", "600519", "300750", "000002", "601318"]
sample_stocks = [s for s in sample_stocks if s in merged["stock"].values]
if sample_stocks:
    for f in factors[:5]:  # 画前5个因子即可
        fig, axes = plt.subplots(len(sample_stocks), 1, figsize=(12, 3 * len(sample_stocks)), sharex=True)
        for j, stk in enumerate(sample_stocks):
            ax = axes[j] if len(sample_stocks) > 1 else axes
            sub = merged[merged["stock"] == stk].sort_values("date")
            ax.plot(sub["date"], sub[f + "_TLY"], label="通联", alpha=0.8)
            ax.plot(sub["date"], sub[f + "_RQ"], label="米筐", alpha=0.8, linestyle="--")
            ax.set_title(f"{stk} - {f}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"stock_timeseries_{f}.png"), dpi=100)
        plt.close()

# ========== 6. 逐日截面分布（KDE 核密度图 + 分位数表） ==========
kde_dir = os.path.join(outdir, f"{prefix}daily_kde")
os.makedirs(kde_dir, exist_ok=True)

quantile_list = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
all_quantile_rows = []

dates = sorted(set(df1["date"]) & set(df2["date"]))

for f in factors:
    n_dates = len(dates)
    fig, axes = plt.subplots(n_dates, 1, figsize=(10, 3.2 * n_dates), sharex=True)
    if n_dates == 1:
        axes = [axes]

    for k, d in enumerate(dates):
        ax = axes[k]
        v1 = df1.loc[df1["date"] == d, f].dropna()
        v2 = df2.loc[df2["date"] == d, f].dropna()

        v1.plot.kde(ax=ax, label=f"通联 (n={len(v1)})", linewidth=1.5, alpha=0.85)
        v2.plot.kde(ax=ax, label=f"米筐 (n={len(v2)})", linewidth=1.5, alpha=0.85, linestyle="--")

        # 标注均值和中位数
        ax.axvline(v1.mean(), color="tab:blue", linestyle=":", alpha=0.6, linewidth=1)
        ax.axvline(v2.mean(), color="tab:orange", linestyle=":", alpha=0.6, linewidth=1)

        ax.set_title(f"{f} - {d.strftime('%Y-%m-%d')}  截面分布")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylabel("密度")

        # 收集分位数
        q1 = v1.quantile(quantile_list)
        q2 = v2.quantile(quantile_list)
        all_quantile_rows.append({"factor": f, "date": d, "source": "通联", "mean": v1.mean(), "std": v1.std(), **{f"q{int(q*100):02d}": q1[q] for q in quantile_list}})
        all_quantile_rows.append({"factor": f, "date": d, "source": "米筐", "mean": v2.mean(), "std": v2.std(), **{f"q{int(q*100):02d}": q2[q] for q in quantile_list}})

    axes[-1].set_xlabel("因子暴露取值")
    plt.tight_layout()
    plt.savefig(os.path.join(kde_dir, f"kde_{f}.png"), dpi=120)
    plt.close()
    print(f"已生成 {f} KDE 图")

quantile_df = pd.DataFrame(all_quantile_rows)
quantile_df.to_csv(os.path.join(kde_dir, "daily_quantiles.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
print(f"\n逐日分位数表已保存至 {os.path.join(kde_dir, 'daily_quantiles.csv')}")

# ========== 6b. 剔除北交所 vs 全量 对比（所有因子，4 组直方图 / 子图） ==========
# 北交所股票：代码以 8 或 4 开头（83/87/88/920 等）
def is_bse(code):
    return code.startswith(("920"))

df1_no = df1[~df1["stock"].apply(is_bse)].copy()
df2_no = df2[~df2["stock"].apply(is_bse)].copy()
n_bse_tly = df1["stock"].nunique() - df1_no["stock"].nunique()
n_bse_rq = df2["stock"].nunique() - df2_no["stock"].nunique()
print(f"\n北交所股票：通联 {n_bse_tly} 只, 米筐 {n_bse_rq} 只")
print(f"剔除后：通联 {df1_no['stock'].nunique()} 只, 米筐 {df2_no['stock'].nunique()} 只")

bse_dir = os.path.join(outdir, f"{prefix}bse_hist_compare")
os.makedirs(bse_dir, exist_ok=True)

N_BINS = 180  # 细粒度直方图

# 颜色 / 线型：蓝=通联，橙=米筐；实线=全量，虚线=剔北交
styles = [
    ("通联 全量",  "tab:blue",   "-"),
    ("通联 剔北交", "tab:blue",   "--"),
    ("米筐 全量",  "tab:orange", "-"),
    ("米筐 剔北交", "tab:orange", "--"),
]

all_q_rows = []
for f in factors:
    n_dates = len(dates)
    fig, axes = plt.subplots(n_dates, 1, figsize=(11, 3.2 * n_dates), sharex=True)
    if n_dates == 1:
        axes = [axes]

    for k, d in enumerate(dates):
        ax = axes[k]
        series = [
            df1.loc[df1["date"] == d, f].dropna(),
            df1_no.loc[df1_no["date"] == d, f].dropna(),
            df2.loc[df2["date"] == d, f].dropna(),
            df2_no.loc[df2_no["date"] == d, f].dropna(),
        ]

        # 统一 bins 范围（取四个序列的全局 min/max）
        xmin = min(s.min() for s in series)
        xmax = max(s.max() for s in series)
        bins = np.linspace(xmin, xmax, N_BINS + 1)

        for s, (lab, col, ls) in zip(series, styles):
            ax.hist(s, bins=bins, density=False, histtype="step",
                    color=col, linestyle=ls, linewidth=1.4, alpha=0.9,
                    label=f"{lab} (n={len(s)})")

        ax.set_title(f"{f} - {d.strftime('%Y-%m-%d')}  全量 vs 剔除北交所")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_ylabel("频数（只）")

        # 分位数
        for s, src, scope in zip(series, ["通联", "通联", "米筐", "米筐"], ["全量", "剔北交", "全量", "剔北交"]):
            q = s.quantile(quantile_list)
            all_q_rows.append({
                "factor": f, "date": d, "source": src, "scope": scope, "n": len(s),
                "mean": s.mean(), "std": s.std(),
                **{f"q{int(p*100):02d}": q[p] for p in quantile_list},
            })

    axes[-1].set_xlabel("因子暴露取值")
    plt.tight_layout()
    plt.savefig(os.path.join(bse_dir, f"hist_{f}_bse_compare.png"), dpi=120)
    plt.close()
    print(f"已生成 {f} 北交所对比直方图")

qdf_all = pd.DataFrame(all_q_rows)
qdf_all.to_csv(os.path.join(bse_dir, "quantiles_bse_compare.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
print(f"\n分位数汇总表已保存至 {os.path.join(bse_dir, 'quantiles_bse_compare.csv')}")

# ========== 7. 差异最大的 Top N 股票-日期样本 ==========
print("\n===== 差异最大的 10 个 (股票, 日期) 样本（按最大绝对差因子排序） =====")
for f in factors:
    merged[f + "_absdiff"] = (merged[f + "_TLY"] - merged[f + "_RQ"]).abs()
absdiff_cols = [f + "_absdiff" for f in factors]
merged["max_absdiff"] = merged[absdiff_cols].max(axis=1)
merged["max_diff_factor"] = merged[absdiff_cols].idxmax(axis=1).str.replace("_absdiff", "")

top_diff = merged.nlargest(20, "max_absdiff")[["date", "stock", "max_diff_factor", "max_absdiff"]]
print(top_diff.to_string(index=False))
top_diff.to_csv(os.path.join(outdir, f"{prefix}top_diff_samples.csv"), index=False, encoding="utf-8-sig")

print(f"\n所有结果已保存至: {outdir}")