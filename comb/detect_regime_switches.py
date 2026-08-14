from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datetime import datetime

SHORT_WINDOW = 20
MID_WINDOW = 40
LONG_WINDOW = 60
RANK_BASELINE_WINDOW = 60 # 识别当前rank_inertia是否处于近期历史高位
RANK_SPIKE_RATIO = 1.35
RANK_CALM_RATIO = 1.05
RANK_ABS_SHIFT_THRESHOLD = 2.0 #判断单因子强势/若是与否，可以是单日也可以是roll求和（和RANK_SHIFT_ACCUM_DAYS配合使用）
RANK_SHIFT_ACCUM_DAYS = 5 #辅助单因子rank日频变动，增加rolling求和指标
TOP_N = 3 #之前是3，为了增加overlap的多样性
CONVERGENCE_LOOKAHEAD = 20
SPREAD_PERSISTENCE_DAYS = 15
SPREAD_CONFIRM_WINDOWS = (20, 40)
OUTPUT_EVENT_FILE = "regime_switch_events.parquet"
OUTPUT_EVENT_XLSX = "regime_switch_events.xlsx"
OUTPUT_DIAG_FILE = "regime_switch_diagnostics.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SPREAD_ORDER = ("spread_gv", "spread_size", "spread_sv")


@dataclass(frozen=True)
class Step1Tables:
    factor_ts_metrics: pd.DataFrame
    factor_cross_section_rank: pd.DataFrame
    rank_inertia: pd.DataFrame
    spread_metrics: pd.DataFrame


@dataclass(frozen=True)
class TransitionStartSignal:
    start_date: pd.Timestamp
    rank_instability: float
    rank_instability_ma5: float
    rank_instability_ma10: float
    strengthening_factors: list[str]
    weakening_factors: list[str]
    lead_quality_score: float


@dataclass(frozen=True)
class TransitionMiddleSignal:
    confirm_date: pd.Timestamp | None
    convergence_score: float
    quality_score: float
    dominant_spread: str | None
    spread_sign: int
    spread_score: float


@dataclass(frozen=True)
class TransitionEndSignal:
    end_date: pd.Timestamp | None
    stabilized: bool
    persistence_days: int


def load_step1_tables(
    output_dir: Path = OUTPUT_DIR,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> Step1Tables:
    start = pd.to_datetime(start) if start is not None else None
    end = pd.to_datetime(end) if end is not None else None

    def clip(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        return df

    return Step1Tables(
        factor_ts_metrics=clip(pd.read_parquet(output_dir / "factor_ts_metrics_daily.parquet")),
        factor_cross_section_rank=clip(pd.read_parquet(output_dir / "factor_cross_section_rank_daily.parquet")),
        rank_inertia=clip(pd.read_parquet(output_dir / "rank_inertia_daily.parquet")),
        spread_metrics=clip(pd.read_parquet(output_dir / "spread_metrics_daily.parquet")),
    )


def prepare_rank_inertia_daily(rank_inertia_df: pd.DataFrame) -> pd.DataFrame:
    """计算单日总体rank变化的稳定性指标，越高说明越不稳定"""
    daily = (
        rank_inertia_df.assign(date=pd.to_datetime(rank_inertia_df["date"]))
        [["date", "window", "rank_instability", "rank_instability_ma5", "rank_instability_ma10"]]
        .drop_duplicates()
        .sort_values(["window", "date"], kind="stable")
        .reset_index(drop=True)
    )
    baseline = daily.groupby("window")["rank_instability"].transform(
        lambda s: s.rolling(RANK_BASELINE_WINDOW, min_periods=max(20, RANK_BASELINE_WINDOW // 3)).median()
    )
    base = baseline.replace(0.0, np.nan)
    daily["rank_ratio"] = daily["rank_instability"].div(base)
    daily["rank_ma5_ratio"] = daily["rank_instability_ma5"].div(base)
    return daily


def prepare_spread_daily(spread_metrics_df: pd.DataFrame) -> pd.DataFrame:
    return spread_metrics_df.assign(date=pd.to_datetime(spread_metrics_df["date"])).sort_values(
        ["spread_name", "window", "date"], kind="stable"
    )


def prepare_factor_window_views(factor_ts_metrics_df: pd.DataFrame) -> pd.DataFrame:
    return factor_ts_metrics_df.assign(date=pd.to_datetime(factor_ts_metrics_df["date"])).sort_values(
        ["date", "window", "factor_name"], kind="stable"
    )


def prepare_factor_rank_views(rank_df: pd.DataFrame) -> pd.DataFrame:
    """单因子的rank变化持续累积，避免单日异常导致的误判"""
    df = rank_df.assign(date=pd.to_datetime(rank_df["date"])).sort_values(["window", "factor_name", "date"], kind="stable")
    df["rank_delta_roll"] = df.groupby(["window", "factor_name"])["rank_delta"].transform(
        lambda s: s.rolling(RANK_SHIFT_ACCUM_DAYS, min_periods=1).sum()
    )
    return df.sort_values(["date", "window", "factor_name"], kind="stable")


def rank_ins_change(rank_inertia_daily: pd.DataFrame, window: int = SHORT_WINDOW) -> pd.DataFrame:
    """判断单日rank_instability是否显著抬升或下降，作为潜在切换的起点"""
    sub = rank_inertia_daily[rank_inertia_daily["window"] == window].copy()
    sub["is_spike"] = (sub["rank_ratio"] >= RANK_SPIKE_RATIO) | (sub["rank_ma5_ratio"] >= RANK_SPIKE_RATIO)
    sub["is_calm"] = (sub["rank_ratio"] <= RANK_CALM_RATIO) & (sub["rank_ma5_ratio"] <= RANK_CALM_RATIO)
    return sub


def factor_rank_shift_candidates(
    rank_rank_df: pd.DataFrame,
    date: pd.Timestamp,
    window: int = SHORT_WINDOW,
    shift_threshold: float = RANK_ABS_SHIFT_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """判断剧烈变化单日中，具体变得强势和弱势的因子，观测指标可以是当天的，也可以是滚动累积的"""
    current = rank_rank_df[(rank_rank_df["date"] == date) & (rank_rank_df["window"] == window)]
    #print(current)
    if current.empty:
        return [], []
    strengthen_mask = (current["rank_delta"] >= shift_threshold) | (current["rank_delta_roll"] >= shift_threshold)
    weaken_mask = (current["rank_delta"] <= -shift_threshold) | (current["rank_delta_roll"] <= -shift_threshold)
    strengthening = current.loc[strengthen_mask].sort_values("rank_delta_roll", ascending=False)["factor_name"].tolist()
    weakening = current.loc[weaken_mask].sort_values("rank_delta_roll")["factor_name"].tolist()
    return strengthening, weakening

#存疑：刚开始切换的时候，强势因子的cum收益表现可能还是垫底的？
def factor_quality_lead_signal(
    factor_ts_metrics_df: pd.DataFrame,
    date: pd.Timestamp,
    factors: Iterable[str],
    window: int = SHORT_WINDOW,
) -> float:
    factors = list(factors)
    if not factors:
        return 0.0
    sub = factor_ts_metrics_df[
        (factor_ts_metrics_df["date"] == date)
        & (factor_ts_metrics_df["window"] == window)
        & (factor_ts_metrics_df["factor_name"].isin(factors))
    ]
    if sub.empty:
        return 0.0
    return float(sub["cum_return"].mean() + sub["ts_sharpe"].mean() - 0.5 * sub["ts_vol"].mean())


def detect_transition_start(
    rank_inertia_daily: pd.DataFrame,
    rank_rank_df: pd.DataFrame,
    factor_ts_metrics_df: pd.DataFrame,
) -> list[TransitionStartSignal]:
    signals = []
    for _, row in rank_ins_change(rank_inertia_daily).loc[lambda x: x["is_spike"]].iterrows():
        date = pd.Timestamp(row["date"])
        strengthening, weakening = factor_rank_shift_candidates(rank_rank_df, date)
        if not strengthening and not weakening:
            continue
        signals.append(
            TransitionStartSignal(
                start_date=date,
                rank_instability=float(row["rank_instability"]),
                rank_instability_ma5=float(row["rank_instability_ma5"]),
                rank_instability_ma10=float(row["rank_instability_ma10"]),
                strengthening_factors=strengthening,
                weakening_factors=weakening,
                lead_quality_score=factor_quality_lead_signal(factor_ts_metrics_df, date, strengthening),
            )
        )
    return signals


def top_factor_set(factor_ts_metrics_df: pd.DataFrame, date: pd.Timestamp, window: int, top_n: int = TOP_N) -> list[str]:
    sub = factor_ts_metrics_df[(factor_ts_metrics_df["date"] == date) & (factor_ts_metrics_df["window"] == window)]
    return [] if sub.empty else sub.nlargest(top_n, ["cum_return", "ts_sharpe"])["factor_name"].tolist()


def window_convergence_score(
    factor_ts_metrics_df: pd.DataFrame,
    start_date: pd.Timestamp,
    lookahead: int = CONVERGENCE_LOOKAHEAD,
) -> tuple[float, pd.Timestamp | None]:
    """寻找强势因子见顶信号"""
    dates = factor_ts_metrics_df.loc[factor_ts_metrics_df["date"] >= start_date, "date"].drop_duplicates().sort_values().head(lookahead)
    best_score = 0.0
    best_date = None
    for date in dates:
        top20 = set(top_factor_set(factor_ts_metrics_df, date, SHORT_WINDOW))
        if not top20:
            continue
        overlap_40 = len(top20 & set(top_factor_set(factor_ts_metrics_df, date, MID_WINDOW))) / len(top20)
        overlap_60 = len(top20 & set(top_factor_set(factor_ts_metrics_df, date, LONG_WINDOW))) / len(top20)
        score = 0.6 * overlap_40 + 0.4 * overlap_60
        if score > best_score:
            best_score = score
            best_date = pd.Timestamp(date)
    return float(best_score), best_date


def factor_quality_strengthening(
    factor_ts_metrics_df: pd.DataFrame,
    factors: Iterable[str],
    start_date: pd.Timestamp,
    lookahead: int = CONVERGENCE_LOOKAHEAD,
) -> tuple[float, float, float]:
    """判断因子收益、sharpe、波动是否在持续改善"""
    factors = list(factors)
    if not factors:
        return 0.0, 0.0, 0.0
    sub = factor_ts_metrics_df[
        (factor_ts_metrics_df["date"] >= start_date)
        & (factor_ts_metrics_df["factor_name"].isin(factors))
        & (factor_ts_metrics_df["window"].isin([SHORT_WINDOW, MID_WINDOW, LONG_WINDOW]))
    ].copy()
    if sub.empty:
        return 0.0, 0.0, 0.0
    dates = sub["date"].drop_duplicates().sort_values().head(lookahead)
    sub = sub[sub["date"].isin(dates)].sort_values(["window", "factor_name", "date"], kind="stable")
    diffs = sub.groupby(["window", "factor_name"])[["cum_return", "ts_sharpe", "ts_vol"]].diff()
    return (
        float((diffs["cum_return"] > 0).mean()),
        float((diffs["ts_sharpe"] > 0).mean()),
        float((diffs["ts_vol"] <= 0).mean()),
    )


def spread_flip_signal(
    spread_daily: pd.DataFrame,
    spread_name: str,
    start_date: pd.Timestamp,
    window: int = SHORT_WINDOW,
) -> tuple[bool, int]:
    """起始日当天spread是否同步变号"""
    sub = spread_daily[(spread_daily["spread_name"] == spread_name) & (spread_daily["window"] == window)]
    prev = sub.loc[sub["date"] < start_date, "direction_sign"].tail(1)
    curr = sub.loc[sub["date"] >= start_date, "direction_sign"].head(1)
    if prev.empty or curr.empty:
        return False, 0
    prev_sign = int(prev.iloc[0])
    curr_sign = int(curr.iloc[0])
    return (prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign), curr_sign


def spread_confirmation_score(
    spread_daily: pd.DataFrame,
    spread_name: str,
    start_date: pd.Timestamp,
    target_sign: int,
    windows: Iterable[int] = SPREAD_CONFIRM_WINDOWS,
) -> float:
    """spread在后续窗口中是否持续确认了切换方向"""
    scores = []
    for window in windows:
        sub = spread_daily[
            (spread_daily["spread_name"] == spread_name)
            & (spread_daily["window"] == window)
            & (spread_daily["date"] >= start_date)
        ].head(SPREAD_PERSISTENCE_DAYS)
        if sub.empty:
            continue
        scores.append(
            #0.5 * (sub["direction_sign"] == target_sign).mean()
            #+ 0.3 * ((sub["cum_return"] * target_sign) > 0).mean()
            1 * (sub["ts_sharpe"] * (sub["direction_sign"] == target_sign)).mean()
        )
    return float(np.mean(scores)) if scores else 0.0


def choose_dominant_spread(spread_daily: pd.DataFrame, start_date: pd.Timestamp) -> tuple[str | None, int, float]:
    """在spread_order中选择最能解释切换方向的spread"""
    best_name = None
    best_sign = 0
    best_score = 0.0
    for spread_name in SPREAD_ORDER:
        flipped, sign = spread_flip_signal(spread_daily, spread_name, start_date)
        if not flipped:
            continue
        score = spread_confirmation_score(spread_daily, spread_name, start_date, sign)
        if score > best_score:
            best_name, best_sign, best_score = spread_name, sign, score
    return best_name, best_sign, float(best_score)


def evaluate_transition_middle(
    factor_ts_metrics_df: pd.DataFrame,
    spread_daily: pd.DataFrame,
    start_signal: TransitionStartSignal,
) -> TransitionMiddleSignal:
    convergence_score, confirm_date = window_convergence_score(factor_ts_metrics_df, start_signal.start_date)
    dominant_spread, spread_sign, spread_score = choose_dominant_spread(spread_daily, start_signal.start_date)
    return TransitionMiddleSignal(
        confirm_date=confirm_date, #如果confirm_date为None，意味着是一个错误的切换信号，因为在未来的20天内，20没有带动40/60上涨
        convergence_score=convergence_score,
        quality_score=factor_quality_strengthening(factor_ts_metrics_df, start_signal.strengthening_factors, start_signal.start_date),
        dominant_spread=dominant_spread,
        spread_sign=spread_sign,
        spread_score=spread_score,
    )


def rank_ins_stabilized(rank_inertia_daily: pd.DataFrame, date: pd.Timestamp) -> bool:
    row = rank_ins_change(rank_inertia_daily).loc[lambda x: x["date"] == date, "is_calm"]
    return bool(row.iloc[0]) if not row.empty else False


def new_regime_persistence(
    spread_daily: pd.DataFrame,
    spread_name: str | None,
    confirm_date: pd.Timestamp | None,
    target_sign: int,
) -> int:
    if spread_name is None or confirm_date is None or target_sign == 0:
        return 0
    sub = spread_daily[
        (spread_daily["spread_name"] == spread_name)
        & (spread_daily["window"] == SHORT_WINDOW)
        & (spread_daily["date"] >= confirm_date)
    ]
    return 0 if sub.empty else int(sub["positive_streak" if target_sign > 0 else "negative_streak"].max())


def confirm_transition_end(
    rank_inertia_daily: pd.DataFrame,
    spread_daily: pd.DataFrame,
    middle_signal: TransitionMiddleSignal,
) -> TransitionEndSignal:
    confirm_date = middle_signal.confirm_date
    if confirm_date is None:
        return TransitionEndSignal(end_date=None, stabilized=False, persistence_days=0)
    persistence_days = new_regime_persistence(spread_daily, middle_signal.dominant_spread, confirm_date, middle_signal.spread_sign)
    dates = (
        rank_inertia_daily.loc[rank_inertia_daily["date"] >= confirm_date, "date"]
        .drop_duplicates()
        .sort_values()
        .head(LONG_WINDOW)
    )
    for date in dates:
        if rank_ins_stabilized(rank_inertia_daily, date) and persistence_days >= SPREAD_PERSISTENCE_DAYS:
            return TransitionEndSignal(end_date=pd.Timestamp(date), stabilized=True, persistence_days=persistence_days)
    return TransitionEndSignal(end_date=confirm_date, stabilized=False, persistence_days=persistence_days)


def label_spread_regime(spread_name: str | None, sign: int) -> str | None:
    if spread_name is None or sign == 0:
        return None
    if spread_name == "spread_gv":
        return "growth_dominant" if sign > 0 else "value_dominant"
    if spread_name == "spread_size":
        return "linear_size_dominant" if sign > 0 else "non_linear_size_dominant"
    return "systematic_elasticity_dominant" if sign > 0 else "idiosyncratic_vol_dominant"



def label_factor_regime(factor_leaders: Iterable[str]) -> str:
    leaders = list(dict.fromkeys(factor_leaders))
    return "transition_mixed" if not leaders else " + ".join(f"{name}_dominant" for name in (leaders[:2] if len(leaders)>2 else leaders))


def leaders_at_date(factor_ts_metrics_df: pd.DataFrame, date: pd.Timestamp, window: int = SHORT_WINDOW) -> list[str]:
    return top_factor_set(factor_ts_metrics_df, date, window=window, top_n=4)


def choose_regime_after_factors(
    factor_ts_metrics_df: pd.DataFrame,
    date: pd.Timestamp,
    strengthening_factors: Iterable[str],
) -> list[str]:
    leaders = leaders_at_date(factor_ts_metrics_df, date)
    overlap = [name for name in leaders if name in set(strengthening_factors)]
    return overlap if overlap else leaders[:2]


def infer_pre_sign(spread_daily: pd.DataFrame, spread_name: str | None, date: pd.Timestamp) -> int:
    if spread_name is None:
        return 0
    sub = spread_daily[
        (spread_daily["spread_name"] == spread_name)
        & (spread_daily["window"] == SHORT_WINDOW)
        & (spread_daily["date"] < date)
    ]
    return 0 if sub.empty else int(sub.iloc[-1]["direction_sign"])


def label_regime_before_during_after(
    factor_ts_metrics_df: pd.DataFrame,
    spread_daily: pd.DataFrame,
    start_signal: TransitionStartSignal,
    middle_signal: TransitionMiddleSignal,
    end_signal: TransitionEndSignal,
) -> tuple[str, str, str]:
    regime_after = label_spread_regime(middle_signal.dominant_spread, middle_signal.spread_sign)
    if regime_after is None:
        regime_after = label_factor_regime(
            choose_regime_after_factors(
                factor_ts_metrics_df,
                end_signal.end_date or middle_signal.confirm_date or start_signal.start_date,
                start_signal.strengthening_factors,
            )
        )
    regime_before = label_spread_regime(middle_signal.dominant_spread, infer_pre_sign(spread_daily, middle_signal.dominant_spread, start_signal.start_date))
    if regime_before is None:
        regime_before = label_factor_regime(leaders_at_date(factor_ts_metrics_df, start_signal.start_date - pd.Timedelta(days=1)))
    return regime_before, "transition_mixed", regime_after


def build_reason_data(start_signal: TransitionStartSignal, middle_signal: TransitionMiddleSignal, end_signal: TransitionEndSignal) -> str:
    quality_score = middle_signal.quality_score
    if isinstance(quality_score, tuple):
        quality_score = {
            "cum_return_up_ratio": round(quality_score[0], 4),
            "ts_sharpe_up_ratio": round(quality_score[1], 4),
            "ts_vol_down_ratio": round(quality_score[2], 4),
        }
    else:
        quality_score = round(quality_score, 4)
    return json.dumps(
        {
            "rank_instability": round(start_signal.rank_instability, 4),
            "rank_instability_ma5": round(start_signal.rank_instability_ma5, 4),
            "rank_instability_ma10": round(start_signal.rank_instability_ma10, 4),
            "strengthening_factors": start_signal.strengthening_factors,
            "weakening_factors": start_signal.weakening_factors,
            "lead_quality_score": round(start_signal.lead_quality_score, 4),
            "convergence_score": round(middle_signal.convergence_score, 4),
            "quality_score": quality_score,
            "dominant_spread": middle_signal.dominant_spread,
            "spread_sign": middle_signal.spread_sign,
            "spread_score": round(middle_signal.spread_score, 4),
            "persistence_days": end_signal.persistence_days,
            "stabilized": end_signal.stabilized,
        },
        ensure_ascii=False,
    )


def build_reason_logic(start_signal: TransitionStartSignal, middle_signal: TransitionMiddleSignal, end_signal: TransitionEndSignal) -> str:
    parts = [
        "排序惯性先显著抬升，说明旧主导结构开始失稳。",
        f"随后 {', '.join(start_signal.strengthening_factors) if start_signal.strengthening_factors else '候选强势因子'} 在20日窗口先行走强。",
        "中段不再要求持续高重排，而是看40/60日是否向20日靠拢、强势因子收益质量是否强化。",
    ]
    if middle_signal.dominant_spread:
        parts.append(f"spread 主线以 {middle_signal.dominant_spread} 最能解释本次切换方向。")
    parts.append("最终排序重新稳定且新方向持续，故确认为一次完成的中期风格切换。" if end_signal.stabilized else "当前已出现切换迹象，但排序稳定性或持续性仍偏弱。")
    return "".join(parts)


def confidence_level(middle_signal: TransitionMiddleSignal, end_signal: TransitionEndSignal) -> str:
    score = middle_signal.convergence_score + middle_signal.spread_score
    if end_signal.stabilized and end_signal.persistence_days >= SPREAD_PERSISTENCE_DAYS and score >= 1.2:
        return "high"
    if score >= 0.8:
        return "medium"
    return "low"


def assemble_regime_event(
    event_index: int,
    start_signal: TransitionStartSignal,
    middle_signal: TransitionMiddleSignal,
    end_signal: TransitionEndSignal,
    factor_ts_metrics_df: pd.DataFrame,
    spread_daily: pd.DataFrame,
) -> dict:
    regime_before, regime_during, regime_after = label_regime_before_during_after(
        factor_ts_metrics_df,
        spread_daily,
        start_signal,
        middle_signal,
        end_signal,
    )
    start_date = start_signal.start_date
    confirm_date = middle_signal.confirm_date or start_date
    end_date = end_signal.end_date or confirm_date
    return {
        "event_id": f"RS_{start_date.strftime('%Y%m%d')}_{event_index:03d}",
        "start_date": start_date,
        "confirm_date": confirm_date,
        "end_date": end_date,
        "regime_before": regime_before,
        "regime_during": regime_during,
        "regime_after": regime_after,
        "dominant_spread": middle_signal.dominant_spread,
        "strengthening_factors": ",".join(start_signal.strengthening_factors),
        "weakening_factors": ",".join(start_signal.weakening_factors),
        "reason_data": build_reason_data(start_signal, middle_signal, end_signal),
        "reason_logic": build_reason_logic(start_signal, middle_signal, end_signal),
        "confidence_level": confidence_level(middle_signal, end_signal),
    }


def merge_overlapping_events(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    out = events_df.sort_values(["start_date", "end_date"], kind="stable").reset_index(drop=True).copy()
    out["is_overlapping"] = out["start_date"] <= out["end_date"].shift(1)
    return out


def build_diagnostics(
    start_signals: list[TransitionStartSignal],
    middle_signals: list[TransitionMiddleSignal],
    end_signals: list[TransitionEndSignal],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": [s.start_date for s in start_signals],
            "rank_instability": [s.rank_instability for s in start_signals],
            "strengthening_factors": [",".join(s.strengthening_factors) for s in start_signals],
            "weakening_factors": [",".join(s.weakening_factors) for s in start_signals],
            "lead_quality_score": [s.lead_quality_score for s in start_signals],
            "confirm_date": [m.confirm_date for m in middle_signals],
            "convergence_score": [m.convergence_score for m in middle_signals],
            "quality_score": [m.quality_score for m in middle_signals],
            "dominant_spread": [m.dominant_spread for m in middle_signals],
            "spread_score": [m.spread_score for m in middle_signals],
            "end_date": [e.end_date for e in end_signals],
            "persistence_days": [e.persistence_days for e in end_signals],
            "stabilized": [e.stabilized for e in end_signals],
        }
    )


def save_event_outputs(events_df: pd.DataFrame, diagnostics_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(output_dir / OUTPUT_EVENT_FILE, index=False)
    diagnostics_df.to_parquet(output_dir / OUTPUT_DIAG_FILE, index=False)
    with pd.ExcelWriter(output_dir / OUTPUT_EVENT_XLSX, engine="openpyxl") as writer:
        events_df.to_excel(writer, sheet_name="events", index=False)
        diagnostics_df.to_excel(writer, sheet_name="diagnostics", index=False)


def run_detection(
    output_dir: Path = OUTPUT_DIR,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[1/5] load tables from {output_dir} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tables = load_step1_tables(output_dir, start=start, end=end)
    factor_ts_metrics_df = prepare_factor_window_views(tables.factor_ts_metrics)
    rank_rank_df = prepare_factor_rank_views(tables.factor_cross_section_rank)
    rank_inertia_daily = prepare_rank_inertia_daily(tables.rank_inertia)
    spread_daily = prepare_spread_daily(tables.spread_metrics)

    print(f"[2/5] detect start signals | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_signals = detect_transition_start(rank_inertia_daily, rank_rank_df, factor_ts_metrics_df)
    if start_signals:
        print(f"  found {len(start_signals)} start signals, first start_date={start_signals[0].start_date.date()}")
    else:
        print("  found 0 start signals")

    print(f"[3/5] evaluate middle signals | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    middle_signals = [evaluate_transition_middle(factor_ts_metrics_df, spread_daily, s) for s in start_signals]

    print(f"[4/5] confirm end signals | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    end_signals = [confirm_transition_end(rank_inertia_daily, spread_daily, m) for m in middle_signals]

    events = [
        assemble_regime_event(i + 1, s, m, e, factor_ts_metrics_df, spread_daily)
        for i, (s, m, e) in enumerate(zip(start_signals, middle_signals, end_signals))
    ]
    events_df = merge_overlapping_events(pd.DataFrame(events)) if events else pd.DataFrame()
    diagnostics_df = build_diagnostics(start_signals, middle_signals, end_signals)
    if not events_df.empty:
        valid = (events_df["confirm_date"] - events_df["start_date"]).dt.days > 7
        events_df = events_df.loc[valid].reset_index(drop=True)
        diagnostics_df = diagnostics_df.loc[valid].reset_index(drop=True)

    print(f"[5/5] save outputs: events={len(events_df)}, diagnostics={len(diagnostics_df)} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    save_event_outputs(events_df, diagnostics_df, output_dir)
    return events_df, diagnostics_df


if __name__ == "__main__":
    events_df, diagnostics_df = run_detection(start="2023-09-30", end="2024-04-01")
    print(f"events: {events_df.shape}")
    print(f"diagnostics: {diagnostics_df.shape}")
    print(f"output_dir: {OUTPUT_DIR}")
