"""
更新 Alpha 私募超额指数净值 + 重新计算回撤分析

用法:
    python update_alpha_excess.py --new "期间国泰海通私募标准指数净值走势_0907.xls"

输入:
    comb/end_input/Alpha私募超额指数.xlsx   旧超额净值（基准=1，2020-12-31起）
    comb/end_input/<新期间文件>.xls           新的区间私募指数净值
输出:
    comb/end_input/Alpha私募超额指数.xlsx   覆盖更新
    comb/outputs/Alpha私募超额指数_回撤分析.xlsx  重新生成

凭据: 从 comb/.env 文件读取 RQDATAC_USER / RQDATAC_PASS
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
COMB_DIR = BASE_DIR / "comb"
INPUT_DIR = COMB_DIR / "end_input"
OUTPUT_DIR = COMB_DIR / "outputs"

# 加载 rqdatac 凭据
load_dotenv(COMB_DIR / ".env")

import rqdatac  # noqa: E402
rqdatac.init(
    username=os.environ.get("RQDATAC_USER", "license"),
    password=os.environ["RQDATAC_PASS"],
)
from rqdatac import get_price_change_rate  # noqa: E402

NAV_FILE = INPUT_DIR / "Alpha私募超额指数.xlsx"
DD_FILE = OUTPUT_DIR / "Alpha私募超额指数_回撤分析.xlsx"

# 列名映射: (私募指增指数列名, 宽基代码, 输出超额列名)
ENHANCED = [
    ("300指增标准指数", "000300.XSHG", "300指增超额"),
    ("500指增标准指数", "000905.XSHG", "500指增超额"),
    ("1000指增标准指数", "000852.XSHG", "1000指增超额"),
]
NEUTRAL_COL = "市场中性标准指数"
NEUTRAL_OUT = "市场中性"

DATE_COL = "日期"


def load_old_nav():
    """读取旧超额净值，返回 DataFrame，index=date"""
    df = pd.read_excel(NAV_FILE)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    # 兼容乱码列名：按位置对应
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "date",
        cols[1]: "300指增超额",
        cols[2]: "500指增超额",
        cols[3]: "1000指增超额",
        cols[4]: "市场中性",
    })
    return df.set_index("date").sort_index()


def _read_xls_with_header(path: Path) -> pd.DataFrame:
    """
    读取国泰海通私募指数 xls，兼容两种格式：
      1. 第一行就是列名（旧格式）
      2. 第一行是合并大标题，第二行才是列名（新格式）
    判断方法：尝试 header=0，如果第一列列名像日期数据或第一行数据含"日期"字样，用 header=1。
    """
    # 先默认 header=0 读
    df = pd.read_excel(path)
    first_col = df.columns[0]
    first_val = str(df.iloc[0, 0])
    # 如果第一行第一列是"日期"（乱码或正常），说明真正的列名在第1行
    if "日期" in first_val or "date" in first_val.lower() or first_col.startswith("Unnamed"):
        df = pd.read_excel(path, header=1)
    return df


def load_new_nav(new_file: Path) -> pd.DataFrame:
    """读取新期间文件的私募指数净值，返回 index=date 的 DataFrame"""
    df = _read_xls_with_header(new_file)
    df = df.rename(columns={df.columns[0]: DATE_COL})
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # 去掉空行（日期为空的行）
    df = df.dropna(subset=[DATE_COL])
    return df.set_index(DATE_COL).sort_index()


def compute_excess_returns(new_nav: pd.DataFrame, old_last_date: pd.Timestamp) -> pd.DataFrame:
    """
    从新净值文件中取 old_last_date 之后的部分，计算超额日收益率。
    返回 index=date, columns=[300指增超额, 500指增超额, 1000指增超额, 市场中性]
    """
    # 只取旧数据最后一天之后（含最后一天，因为要算第一个收益率）
    mask = new_nav.index >= old_last_date
    sub = new_nav.loc[mask].copy()
    if len(sub) < 2:
        raise ValueError(f"新文件中 {old_last_date.date()} 之后数据不足，无法计算收益率")

    ret = sub.pct_change().dropna()

    # 指增：减宽基收益率
    out = {}
    for priv_col, bench_code, out_col in ENHANCED:
        bench_ret = get_price_change_rate(
            bench_code,
            start_date=ret.index[0].strftime("%Y-%m-%d"),
            end_date=ret.index[-1].strftime("%Y-%m-%d"),
            expect_df=True,
            market='cn',
        )
        bench_ret.index.name = DATE_COL
        bench_ret.index = pd.to_datetime(bench_ret.index)
        # 对齐日期
        aligned = ret[[priv_col]].join(bench_ret.rename(columns={bench_code: "bench"}), how="inner")
        out[out_col] = aligned[priv_col] - aligned["bench"]

    # 市场中性：直接用自身收益率
    out[NEUTRAL_OUT] = ret[NEUTRAL_COL]

    df = pd.DataFrame(out)
    df.index.name = "date"
    return df.sort_index()


def append_excess(old_nav: pd.DataFrame, excess_ret: pd.DataFrame) -> pd.DataFrame:
    """把新超额收益率累乘到旧超额净值末尾，返回完整超额净值"""
    last = old_nav.iloc[-1]
    new_nav = (1 + excess_ret).cumprod() * last
    # 去重（以防日期重叠），保留旧数据
    combined = pd.concat([old_nav, new_nav])
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


# ---------------- 回撤分析（从 end_check.py 搬过来，略精简） ----------------

def drawdown_periods(nav, dates, min_dd=-0.02, min_days=5):
    s = np.asarray(nav, float)
    n = len(s)
    rmax = np.maximum.accumulate(s)
    dd = s / rmax - 1.0
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
    return res.sort_values("start_date").reset_index(drop=True) if len(res) else res


def find_common_drawdowns(df, results, min_products=3, min_days=2):
    """
    识别多产品同时回撤的共性区间。

    定义：
    - start_date: 同时回撤产品数 ≥ min_products 的第一天
    - end_date:   第一个产品完成修复（创新高）的那天。
                  即：连续段结束后，首次有产品脱离回撤的交易日。
                  若到数据终点全部产品仍在回撤 → 未修复，end_date 留空。
    - 已修复标的: end_date 当天已修复的产品
    - 未修复标的: end_date 当天仍在回撤的产品 + 修复缺口（%）
                  缺口 = (回撤起点净值 - 当前净值) / 当前净值
    """
    products = [c for c in df.columns if c != "date"]
    dates = df["date"].values
    n = len(df)

    # 每个产品每天所在回撤的起点净值（NaN = 不在回撤中）
    dd_start_nav = {col: np.full(n, np.nan) for col in products}
    for col in products:
        for _, r in results[col].iterrows():
            idx = (df["date"] >= r["start_date"]) & (df["date"] <= r["end_date"])
            dd_start_nav[col][idx.values] = r["start_nav"]

    # 逐日同时回撤产品数
    mask = sum(~np.isnan(dd_start_nav[col]) for col in products).astype(int)
    is_common = mask >= min_products

    # 找连续段
    diff = np.diff(is_common.astype(int))
    seg_starts = np.where(diff == 1)[0] + 1
    seg_ends = np.where(diff == -1)[0]
    if is_common[0]:
        seg_starts = np.r_[0, seg_starts]
    if is_common[-1]:
        seg_ends = np.r_[seg_ends, n - 1]

    def gap_pct(col, i):
        """第 i 天产品 col 的修复缺口（%）"""
        sn = dd_start_nav[col][i]
        cn = df[col].iloc[i]
        if np.isnan(sn) or cn == 0:
            return None
        return round((sn - cn) / cn * 100, 2)

    rows = []
    for s, e in zip(seg_starts, seg_ends):
        if e - s + 1 < min_days:
            continue
        start_date = dates[s]
        end_date = dates[e]  # last day of common drawdown (all still suffering)

        # Determine recovery status and reference day for gap calculation
        # - Normal end (e < n-1): e+1 is the first day when mask drops below threshold,
        #   meaning at least one product has fully recovered by then.
        #   Use e+1 as ref_idx to count who's recovered / still hurting.
        # - At data end (e == n-1): check if all involved drawdowns eventually recover.
        if e < n - 1:
            ref_idx = e + 1
            is_recovered = True
        else:
            ref_idx = e
            any_unrecovered = False
            for col in products:
                if not np.isnan(dd_start_nav[col][e]):
                    for _, r in results[col].iterrows():
                        if r["start_date"] <= dates[e] <= r["end_date"]:
                            if not r["recovered"]:
                                any_unrecovered = True
                            break
            is_recovered = not any_unrecovered

        # Recovery status at ref_idx
        rec_list = [col for col in products if np.isnan(dd_start_nav[col][ref_idx])]
        unrec_list = [col for col in products if not np.isnan(dd_start_nav[col][ref_idx])]

        dur = e - s + 1
        peak_idx = s + np.argmax(mask[s:e + 1])
        peak_day = dates[peak_idx]

        if not is_recovered:
            row = {
                "start_date": start_date,
                "end_date": pd.NaT,
                "duration_days": None,
                "peak_overlap_day": pd.NaT,
                "max_simultaneous": int(mask[s:e + 1].max()),
                "coverage_ratio": round(mask[s:e + 1].max() / len(products), 2),
                "recovered": False,
                "已修复标的": "、".join(rec_list) if rec_list else "",
                "未修复标的": "、".join(
                    f"{col}({gap_pct(col, ref_idx)}%)" for col in unrec_list
                ),
            }
        else:
            row = {
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": dur,
                "peak_overlap_day": peak_day,
                "max_simultaneous": int(mask[s:e + 1].max()),
                "coverage_ratio": round(mask[s:e + 1].max() / len(products), 2),
                "recovered": True,
                "已修复标的": "、".join(rec_list),
                "未修复标的": "、".join(
                    f"{col}({gap_pct(col, ref_idx)}%)" for col in unrec_list
                ),
            }
        rows.append(row)

    return pd.DataFrame(rows)


def summary_stats(df, results):
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


def run_drawdown(nav_df: pd.DataFrame):
    df = nav_df.reset_index().rename(columns={"index": "date"})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2020-12-31"].sort_values("date").reset_index(drop=True)
    products = [c for c in df.columns if c != "date"]
    results = {c: drawdown_periods(df[c].values, df["date"].values) for c in products}
    summary = summary_stats(df, results)
    common = find_common_drawdowns(df, results, min_products=3)

    print(f"\n数据区间: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}, {len(df)} 交易日")
    print("\n=== 汇总统计 ===")
    print(summary.to_string(index=False))
    print(f"\n=== 共性回撤期（≥3/4产品同时，共{len(common)}段）===")
    print(common.to_string(index=False) if len(common) else "无")
    for col, p in results.items():
        print(f"\n--- {col} TOP5 回撤（按深度） ---")
        print(p.sort_values("max_dd_pct")[["start_date", "trough_date", "end_date", "max_dd_pct",
                                           "duration_days", "days_to_trough", "recovery_days", "recovered"]]
              .head().to_string(index=False))

    DD_FILE.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(DD_FILE, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="汇总统计", index=False)
        common.to_excel(w, sheet_name="共性回撤期", index=False)
        for col, p in results.items():
            p.to_excel(w, sheet_name=col[:28], index=False)
    print(f"\n回撤分析已保存: {DD_FILE}")


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser(description="更新 Alpha 私募超额指数 + 回撤分析")
    parser.add_argument("--new", required=True, help="新的期间私募指数净值 xls 文件路径")
    args = parser.parse_args()

    new_file = Path(args.new)
    if not new_file.exists():
        # 在 end_input 目录里按文件名查找
        candidate = INPUT_DIR / new_file.name
        if candidate.exists():
            new_file = candidate

    print(f"旧超额净值: {NAV_FILE}")
    print(f"新期间文件: {new_file}")

    # 1. 读旧超额净值
    old_nav = load_old_nav()
    old_last = old_nav.index[-1]
    print(f"旧数据区间: {old_nav.index[0].date()} ~ {old_last.date()}, {len(old_nav)} 条")

    # 2. 读新净值
    new_nav = load_new_nav(new_file)
    print(f"新文件区间: {new_nav.index[0].date()} ~ {new_nav.index[-1].date()}, {len(new_nav)} 条")

    # 3. 算新超额收益率（如果有新数据的话）
    if new_nav.index[-1] <= old_last:
        print("新文件数据未超出旧数据范围，无需更新净值。")
        full_nav = old_nav
    else:
        excess_ret = compute_excess_returns(new_nav, old_last)
        print(f"新增超额收益率: {excess_ret.index[0].date()} ~ {excess_ret.index[-1].date()}, {len(excess_ret)} 条")

        # 4. 追加到旧净值
        full_nav = append_excess(old_nav, excess_ret)
        print(f"更新后区间: {full_nav.index[0].date()} ~ {full_nav.index[-1].date()}, {len(full_nav)} 条")

        # 5. 保存超额净值
        full_nav.to_excel(NAV_FILE, index=True, index_label="date")
        print(f"超额净值已保存: {NAV_FILE}")

    # 6. 重算回撤分析
    run_drawdown(full_nav)


if __name__ == "__main__":
    main()
