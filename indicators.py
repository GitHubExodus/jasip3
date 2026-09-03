import numpy as np
import pandas as pd

from numba import njit


TIMEFRAMES = {
    "5m": "5min",
    "30m": "30min",
    "1d": "1D",
    "1w": "1W",
}


@njit
def ema_numpy(values, period):
    n = len(values)

    result = np.empty(n)
    result[:] = np.nan

    if n < period:
        return result

    alpha = 2.0 / (period + 1.0)

    total = 0.0

    for i in range(period):
        total += values[i]

    result[period - 1] = total / period

    for i in range(period, n):
        result[i] = (
            alpha * values[i]
            + (1.0 - alpha) * result[i - 1]
        )

    return result


def prepare_raw_data(df):
    df = df.copy()

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    rename_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "volume": "volume",
        "vol": "volume",
    }

    df = df.rename(
        columns=rename_map
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        x for x in required
        if x not in df.columns
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
        subset=["timestamp"]
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


def resample_timeframe(
    raw_df,
    timeframe,
):
    """
    Convert 1-minute data into the requested
    timeframe.

    The timestamp represents the CLOSE of
    the resulting candle.
    """

    rule = TIMEFRAMES[timeframe]

    x = raw_df.copy()

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


def add_indicators(df):
    df = df.copy()

    close = np.asarray(
        df["close"],
        dtype=np.float64,
    )

    df["ema_3"] = ema_numpy(
        close,
        3,
    )

    df["ema_5"] = ema_numpy(
        close,
        5,
    )

    df["ema_21"] = ema_numpy(
        close,
        21,
    )

    df["ema_200"] = ema_numpy(
        close,
        200,
    )

    return df


def build_all_timeframes(raw_df):
    raw_df = prepare_raw_data(
        raw_df
    )

    result = {}

    for timeframe in TIMEFRAMES:

        print(
            f"Creating {timeframe} candles...",
            flush=True,
        )

        data = resample_timeframe(
            raw_df,
            timeframe,
        )

        data = add_indicators(
            data
        )

        result[timeframe] = data

    return result