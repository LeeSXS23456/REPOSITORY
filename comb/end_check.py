"""
Alpha私募超额指数回撤分析 — 识别2021年以来量化产品主要回撤期
"""
import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(__file__).parent / "end_input" / "Alpha私募超额指数.xlsx"
OUTPUT = Path(__file__).parent / "outputs" / "Alpha私募超额指数_回撤分析.xlsx"

COL_NAMES = ["date", "300指增超额", "500指增超额", "1000指增超额", "市场中性"]


def load_data(path=INPUT):
    """读取数据并重命名列（原文件中文字符编码损坏）"""
    df = pd.read_excel(path)
    df.columns = COL_NAMES[:len(df.columns)]
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= "2020-12-31"].sort_values("date").reset_index(drop=True)


def drawdown_periods(nav, dates, min_dd=-0.02, min_days=5):
    """
    识别单条净值的回撤期（峰值→谷值→新高）。
    返回按深度排序的 DataFrame。
    """
    s = np.asarray(nav, float)
    n = len(s)
    if n < 3:
        return pd.DataFrame()

    rmax = np.maximum.accumulate(s)
    dd = s / rmax - 1.0

    # 新高位置 = 每段回撤起点
    peaks = np.zeros(n, bool)
    peaks[0] = True
    peaks[1:] = rmax[1:] > rmax[:-1] + 1e-12
    pi = np.where(peaks)[0]

    rows = []
    for i, start_i in enumerate(pi):
        end_i = pi[i + 1] if i + 1 < len(pi) else n - 1
        recovered = i + 1 < len(pi) or s[end_i] >= s[start_i] - 1e-12
        sub = dd[start_i:end_i + 1]
        t = np.argmin(sub)
        mdd = sub[t]
        dur = end_i - start_i + 1
        if mdd <= min_dd and dur >= min_days:
            rows.append({
                "start_date": dates[start_i],
                "trough_date": dates[start_i + t],
                "end_date": dates[end_i],
                "max_dd_pct": round(mdd * 100, 2),
                "duration_days": dur,
                "days_to_trough": t,
                "recovery_days": end_i - start_i - t if recovered else None,
                "recovered": recovered,
                "start_nav": s[start_i],
                "trough_nav": s[start_i + t],
                "end_nav": s[end_i],
            })

    res = pd.DataFrame(rows)
    return res.sort_values("max_dd_pct").reset_index(drop=True) if len(res) else res


def find_common_drawdowns(df, results, min_products=3, min_days=2):
    """识别多产品同时回撤的共性区间"""
    products = [c for c in df.columns if c != "date"]
    n = len(df)
    mask = np.zeros(n, int)

    for col in products:
        in_dd = np.zeros(n, bool)
        for _, r in results[col].iterrows():
            idx = (df["date"] >= r["start_date"]) & (df["date"] <= r["end_date"])
            in_dd |= idx.values
        mask += in_dd.astype(int)

    is_common = mask >= min_products
    # 找连续段
    diff = np.diff(is_common.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if is_common[0]:
        starts = np.r_[0, starts]
    if is_common[-1]:
        ends = np.r_[ends, n - 1]

    rows = []
    for s, e in zip(starts, ends):
        if e - s + 1 >= min_days:
            rows.append({
                "start_date": df["date"].iloc[s],
                "end_date": df["date"].iloc[e],
                "duration_days": e - s + 1,
                "peak_overlap_day": df["date"].iloc[s + np.argmax(mask[s:e + 1])],
                "max_simultaneous": int(mask[s:e + 1].max()),
                "coverage_ratio": round(mask[s:e + 1].max() / len(products), 2),
            })
    return pd.DataFrame(rows)


def summary_stats(df, results):
    """各产品总体指标"""
    rows = []
    for col, p in results.items():
        ann = (df[col].iloc[-1] / df[col].iloc[0]) ** (252 / len(df)) - 1
        rec = p[p["recovered"]]
        rows.append({
            "产品": col,
            "累计收益(%)": round((df[col].iloc[-1] / df[col].iloc[0] - 1) * 100, 2),
            "年化收益(%)": round(ann * 100, 2),
            "最大回撤(%)": p["max_dd_pct"].min() if len(p) else 0,
            "回撤次数": len(p),
            "平均回撤(%)": round(p["max_dd_pct"].mean(), 2) if len(p) else 0,
            "平均持续天数": round(p["duration_days"].mean(), 1) if len(p) else 0,
            "平均修复天数": round(rec["recovery_days"].mean(), 1) if len(rec) else 0,
            "回撤天数占比(%)": round(p["duration_days"].sum() / len(df) * 100, 1),
            "未修复回撤数": int((~p["recovered"]).sum()),
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    df = load_data()
    products = [c for c in df.columns if c != "date"]
    results = {c: drawdown_periods(df[c].values, df["date"].values) for c in products}

    summary = summary_stats(df, results)
    common = find_common_drawdowns(df, results, min_products=3)

    # 打印
    print(f"数据区间: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}, {len(df)} 交易日")
    print("\n=== 汇总统计 ===")
    print(summary.to_string(index=False))
    print(f"\n=== 共性回撤期（≥3/4产品同时，共{len(common)}段）===")
    print(common.to_string(index=False) if len(common) else "无")
    for col, p in results.items():
        print(f"\n--- {col} TOP5 回撤 ---")
        print(p[["start_date", "trough_date", "end_date", "max_dd_pct",
                 "duration_days", "days_to_trough", "recovery_days", "recovered"]]
              .head().to_string(index=False))

    # 保存
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="汇总统计", index=False)
        common.to_excel(w, sheet_name="共性回撤期", index=False)
        for col, p in results.items():
            p.to_excel(w, sheet_name=col[:28], index=False)
    print(f"\n结果已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
