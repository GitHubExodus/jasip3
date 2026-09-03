import numpy as np
import pandas as pd

from numba import njit


# ============================================================
# CONFIGURATION
# ============================================================

TIMEFRAMES = {
    "5m": "5min",
    "30m": "30min",
    "4h": "4h",
    "1d": "1D",
    "1w": "W-FRI",
}

EMA_PERIODS = (
    3,
    5,
    9,
    14,
    21,
    30,
    50,
    100,
    200,
)


# ============================================================
# EMA
# ============================================================

@njit
def ema_numba(
    values,
    period,
):
    n = len(values)

    result = np.empty(
        n,
        dtype=np.float64,
    )

    result[:] = np.nan

    if n < period:
        return result

    total = 0.0

    for i in range(period):
        total += values[i]

    result[period - 1] = (
        total / period
    )

    alpha = (
        2.0
        / (period + 1.0)
    )

    for i in range(
        period,
        n,
    ):
        result[i] = (
            alpha * values[i]
            + (
                1.0 - alpha
            )
            * result[i - 1]
        )

    return result


# ============================================================
# PREPARE RAW DATA
# ============================================================

def prepare_raw_data(
    df,
):
    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]

    rename_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "vol": "volume",
    }

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset="timestamp"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    if "volume" in df.columns:
        numeric_columns.append(
            "volume"
        )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df


# ============================================================
# RESAMPLE
# ============================================================

def resample_timeframe(
    raw_df,
    timeframe,
):
    rule = TIMEFRAMES[
        timeframe
    ]

    x = raw_df.copy()

    # Convert to New York time before
    # creating US stock-market timeframes.
    x["timestamp"] = (
        x["timestamp"]
        .dt.tz_convert(
            "America/New_York"
        )
    )

    x = x.set_index(
        "timestamp"
    )

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in x.columns:
        aggregation["volume"] = "sum"

    result = (
        x.resample(
            rule,
            label="right",
            closed="right",
        )
        .agg(aggregation)
    )

    result = result.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    result = result.reset_index()

    return result


# ============================================================
# ADD ALL EMAS
# ============================================================

def add_indicators(
    df,
):
    df = df.copy()

    close = np.asarray(
        df["close"],
        dtype=np.float64,
    )

    for period in EMA_PERIODS:

        df[
            f"ema_{period}"
        ] = ema_numba(
            close,
            period,
        )

    return df


# ============================================================
# BUILD ALL TIMEFRAMES
# ============================================================

def build_all_timeframes(
    raw_df,
):
    raw_df = prepare_raw_data(
        raw_df
    )

    result = {}

    for timeframe in TIMEFRAMES:

        print(
            f"Creating {timeframe}...",
            flush=True,
        )

        timeframe_df = (
            resample_timeframe(
                raw_df,
                timeframe,
            )
        )

        timeframe_df = (
            add_indicators(
                timeframe_df
            )
        )

        result[
            timeframe
        ] = timeframe_df

        print(
            f"  {timeframe}: "
            f"{len(timeframe_df):,} bars",
            flush=True,
        )

    return result