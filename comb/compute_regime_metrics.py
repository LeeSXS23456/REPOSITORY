from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WINDOWS = (20, 40, 60)
FACTOR_COLUMNS = [
    "beta",
    "book_to_price",
    "earnings_yield",
    "growth",
    "leverage",
    "liquidity",
    "momentum",
    "non_linear_size",
    "residual_volatility",
    "size",
]
SPREAD_ORDER = ["spread_gv", "spread_size", "spread_sv"]
SIGN_EPS = 1e-12
DEMO_ROWS = 120
TAIL_LOOKBACK = 80

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data_base" / "fac_ret" / "whole_mkt" / "factor_returns_10_2608.pkl"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class Paths:
    factor_return_daily: Path
    factor_ts_metrics_daily: Path
    factor_cross_section_rank_daily: Path
    rank_inertia_daily: Path
    spread_metrics_daily: Path
    demo_excel: Path


def build_paths(output_dir: Path) -> Paths:
    return Paths(
        factor_return_daily=output_dir / "factor_return_daily.parquet",
        factor_ts_metrics_daily=output_dir / "factor_ts_metrics_daily.parquet",
        factor_cross_section_rank_daily=output_dir / "factor_cross_section_rank_daily.parquet",
        rank_inertia_daily=output_dir / "rank_inertia_daily.parquet",
        spread_metrics_daily=output_dir / "spread_metrics_daily.parquet",
        demo_excel=output_dir / "regime_metrics_demo.xlsx",
    )


def load_factor_returns(path: Path = INPUT_PATH, factor_columns: Iterable[str] = FACTOR_COLUMNS) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = pd.DataFrame(df).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    missing = [col for col in factor_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing factor columns: {missing}")
    df = df.loc[:, list(factor_columns)].astype("float64")
    df.index.name = "date"
    return df


def rolling_cum_return(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return (1.0 + df).rolling(window=window, min_periods=window).apply(np.prod, raw=True) - 1.0


def rolling_vol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=window).std(ddof=0)


def rolling_sharpe(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = df.rolling(window=window, min_periods=window).mean()
    vol = rolling_vol(df, window)
    return mean.div(vol.replace(0.0, np.nan))


def build_factor_ts_metrics(factor_returns: pd.DataFrame, windows: Iterable[int] = WINDOWS) -> pd.DataFrame:
    frames = []
    for window in windows:
        cum_return = rolling_cum_return(factor_returns, window)
        ts_vol = rolling_vol(factor_returns, window)
        ts_sharpe = rolling_sharpe(factor_returns, window)

        frame = pd.concat(
            {
                "cum_return": cum_return.stack(dropna=False),
                "ts_vol": ts_vol.stack(dropna=False),
                "ts_sharpe": ts_sharpe.stack(dropna=False),
            },
            axis=1,
        ).reset_index()
        frame.columns = ["date", "factor_name", "cum_return", "ts_vol", "ts_sharpe"]
        frame.insert(2, "window", window)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "window", "factor_name"], kind="stable").reset_index(drop=True)
    return out


def build_cross_section_ranks(factor_ts_metrics: pd.DataFrame) -> pd.DataFrame:
    rank_df = factor_ts_metrics[["date", "factor_name", "window", "cum_return"]].copy()
    group = rank_df.groupby(["date", "window"])["cum_return"]
    rank_df["rank_by_cum_return"] = group.rank(ascending=True, method="average")
    rank_df["rank_pct_by_cum_return"] = group.rank(ascending=True, method="average", pct=True)

    rank_df = rank_df.sort_values(["window", "factor_name", "date"], kind="stable")
    prev_rank = rank_df.groupby(["window", "factor_name"])["rank_by_cum_return"].shift(1)
    rank_df["rank_delta"] = rank_df["rank_by_cum_return"] - prev_rank

    rank_df = rank_df.drop(columns=["cum_return"])
    rank_df = rank_df.sort_values(["date", "window", "factor_name"], kind="stable").reset_index(drop=True)
    return rank_df


def build_rank_inertia(rank_df: pd.DataFrame) -> pd.DataFrame:
    rank_changes = rank_df[["date", "window", "factor_name", "rank_delta"]].copy()
    rank_changes["rank_delta_abs"] = rank_changes["rank_delta"].abs()

    inertia = (
        rank_changes.groupby(["date", "window"], as_index=False)["rank_delta_abs"]
        .sum(min_count=1)
        .rename(columns={"rank_delta_abs": "rank_instability"})
        .sort_values(["window", "date"], kind="stable")
    )

    inertia["rank_instability_ma5"] = (
        inertia.groupby("window")["rank_instability"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    inertia["rank_instability_ma10"] = (
        inertia.groupby("window")["rank_instability"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    )

    out = rank_changes.merge(inertia, on=["date", "window"], how="left")
    out = out.sort_values(["date", "window", "factor_name"], kind="stable").reset_index(drop=True)
    return out


def build_spread_values(factor_returns: pd.DataFrame) -> pd.DataFrame:
    spread_df = pd.DataFrame(index=factor_returns.index)
    spread_df["spread_gv"] = (
        factor_returns["growth"] + factor_returns["momentum"]
    ) / 2.0 - (
        factor_returns["book_to_price"] + factor_returns["earnings_yield"]
    ) / 2.0
    spread_df["spread_size"] = factor_returns["size"] - factor_returns["non_linear_size"]
    spread_df["spread_sv"] = (
        factor_returns["beta"] + factor_returns["liquidity"]
    ) / 2.0 - factor_returns["residual_volatility"]
    spread_df.index.name = "date"
    return spread_df.loc[:, SPREAD_ORDER]


def sign_with_epsilon(values: pd.DataFrame, eps: float = SIGN_EPS) -> pd.DataFrame:
    signs = np.where(values.to_numpy() > eps, 1, np.where(values.to_numpy() < -eps, -1, 0))
    return pd.DataFrame(signs, index=values.index, columns=values.columns, dtype="int8")


def streak_length(sign_df: pd.DataFrame, target_sign: int) -> pd.DataFrame:
    arr = (sign_df.to_numpy() == target_sign).astype(np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    if len(arr) == 0:
        return pd.DataFrame(out, index=sign_df.index, columns=sign_df.columns)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = np.where(arr[i] == 1, out[i - 1] + 1, 0)
    return pd.DataFrame(out, index=sign_df.index, columns=sign_df.columns)


def build_spread_metrics(factor_returns: pd.DataFrame, windows: Iterable[int] = WINDOWS) -> pd.DataFrame:
    spread_values = build_spread_values(factor_returns)
    sign_df = sign_with_epsilon(spread_values)
    pos_streak = streak_length(sign_df, 1)
    neg_streak = streak_length(sign_df, -1)

    base = pd.concat(
        {
            "spread_value": spread_values.stack(dropna=False),
            "direction_sign": sign_df.stack(dropna=False),
            "positive_streak": pos_streak.stack(dropna=False),
            "negative_streak": neg_streak.stack(dropna=False),
        },
        axis=1,
    ).reset_index()
    base.columns = [
        "date",
        "spread_name",
        "spread_value",
        "direction_sign",
        "positive_streak",
        "negative_streak",
    ]

    frames = []
    for window in windows:
        cum_return = rolling_cum_return(spread_values, window)
        ts_vol = rolling_vol(spread_values, window)
        ts_sharpe = rolling_sharpe(spread_values, window)
        metrics = pd.concat(
            {
                "cum_return": cum_return.stack(dropna=False),
                "ts_vol": ts_vol.stack(dropna=False),
                "ts_sharpe": ts_sharpe.stack(dropna=False),
            },
            axis=1,
        ).reset_index()
        metrics.columns = ["date", "spread_name", "cum_return", "ts_vol", "ts_sharpe"]
        metrics.insert(2, "window", window)
        merged = base.merge(metrics, on=["date", "spread_name"], how="inner")
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "window", "spread_name"], kind="stable").reset_index(drop=True)
    out["direction_sign"] = out["direction_sign"].astype("int8")
    out["positive_streak"] = out["positive_streak"].astype("int32")
    out["negative_streak"] = out["negative_streak"].astype("int32")
    return out


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        fallback = path.with_suffix(".pkl")
        df.to_pickle(fallback)
        print(f"Parquet save failed, wrote pickle instead: {fallback}")


def load_existing_table(path: Path) -> pd.DataFrame | None:
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    pickle_path = path.with_suffix(".pkl")
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    return None


def get_existing_max_date(path: Path) -> pd.Timestamp | None:
    existing = load_existing_table(path)
    if existing is None or "date" not in existing.columns or existing.empty:
        return None
    return pd.to_datetime(existing["date"]).max()


def resolve_recompute_start(dates: pd.DatetimeIndex, old_max_date: pd.Timestamp | None, lookback: int) -> pd.Timestamp:
    if old_max_date is None:
        return dates.min()
    insert_pos = dates.searchsorted(old_max_date, side="right")
    start_pos = max(0, insert_pos - lookback)
    return dates[start_pos]


def tail_replace(old_df: pd.DataFrame | None, new_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return new_df.sort_values(list(new_df.columns[: min(3, len(new_df.columns))]), kind="stable").reset_index(drop=True)
    old_df = old_df.copy()
    old_df["date"] = pd.to_datetime(old_df["date"])
    new_df = new_df.copy()
    new_df["date"] = pd.to_datetime(new_df["date"])
    kept = old_df[old_df["date"] < start_date]
    out = pd.concat([kept, new_df], ignore_index=True)
    key_cols = [col for col in ["date", "window", "factor_name", "spread_name"] if col in out.columns]
    if key_cols:
        out = out.drop_duplicates(subset=key_cols, keep="last")
        out = out.sort_values(key_cols, kind="stable")
    else:
        out = out.sort_values("date", kind="stable")
    return out.reset_index(drop=True)


def persist_table(path: Path, new_df: pd.DataFrame, start_date: pd.Timestamp | None) -> pd.DataFrame:
    old_df = load_existing_table(path) if start_date is not None else None
    out = tail_replace(old_df, new_df, start_date) if start_date is not None else new_df.reset_index(drop=True)
    save_table(out, path)
    return out


def demo_slice(df: pd.DataFrame, n: int = DEMO_ROWS) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    half = n // 2
    return pd.concat([df.head(half), df.tail(n - half)], ignore_index=True)


def save_demo_excel(tables: dict[str, pd.DataFrame], path: Path, n: int = DEMO_ROWS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            demo_slice(df, n=n).to_excel(writer, sheet_name=sheet_name[:31], index=False)


def run(
    output_dir: Path = OUTPUT_DIR,
    input_path: Path = INPUT_PATH,
    incremental: bool = True,
    lookback: int = TAIL_LOOKBACK,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(output_dir)
    factor_returns = load_factor_returns(input_path)

    old_max_date = get_existing_max_date(paths.factor_return_daily) if incremental else None
    new_max_date = factor_returns.index.max()
    if incremental and old_max_date is not None and new_max_date <= old_max_date:
        print(f"No new dates detected. Existing max date: {old_max_date.date()}")
        return {}

    recompute_start = resolve_recompute_start(factor_returns.index, old_max_date, lookback) if incremental else None
    factor_returns_tail = factor_returns.loc[recompute_start:] if recompute_start is not None else factor_returns

    added_days = 0 if old_max_date is None else int((factor_returns.index > old_max_date).sum())
    mode = "incremental" if incremental and old_max_date is not None else "full"
    print(
        f"Run mode: {mode}; added trading days: {added_days}; "
        f"input max date: {new_max_date.date()}"
        + (f"; recompute start: {recompute_start.date()}" if recompute_start is not None else "")
    )

    new_tables = {
        "factor_return_daily": factor_returns_tail.reset_index(),
        "factor_ts_metrics_daily": build_factor_ts_metrics(factor_returns_tail),
    }
    new_tables["factor_cross_section_rank_daily"] = build_cross_section_ranks(new_tables["factor_ts_metrics_daily"])
    new_tables["rank_inertia_daily"] = build_rank_inertia(new_tables["factor_cross_section_rank_daily"])
    new_tables["spread_metrics_daily"] = build_spread_metrics(factor_returns_tail)

    persisted = {
        name: persist_table(getattr(paths, name), df, recompute_start)
        for name, df in new_tables.items()
    }
    save_demo_excel(persisted, paths.demo_excel)
    return persisted


if __name__ == "__main__":
    tables = run()
    for name, table in tables.items():
        print(f"{name}: {table.shape}")
    print(f"demo_excel: {build_paths(OUTPUT_DIR).demo_excel}")
