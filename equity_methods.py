import numpy as np
import pandas as pd

from numba import njit


STARTING_EQUITY = 100.000
MA_PERIOD = 5


@njit
def calculate_raw_equity_numba(
    exit_indices,
    returns,
):
    n = len(returns)

    equity = np.empty(
        n,
        dtype=np.float64,
    )

    value = STARTING_EQUITY

    for i in range(n):

        value *= (
            1.0
            + returns[i] / 100.0
        )

        value = np.round(
            value,
            3,
        )

        equity[i] = value

    return equity


def calculate_raw_equity(
    trades,
):
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "equity",
                "return_pct",
            ]
        )

    trades = trades.sort_values(
        "exit_time"
    ).reset_index(drop=True)

    returns = trades[
        "return_pct"
    ].to_numpy(
        dtype=np.float64
    )

    equity = calculate_raw_equity_numba(
        np.arange(len(returns)),
        returns,
    )

    return pd.DataFrame(
        {
            "timestamp":
                trades["exit_time"].to_numpy(),

            "equity":
                equity,

            "return_pct":
                returns,
        }
    )


def calculate_buying_power(
    current_value,
    ma,
    average_distance,
):
    if current_value <= ma:
        return 0.0

    distance = (
        (current_value - ma)
        / ma
    ) * 100.0

    if average_distance <= 0:
        return 1.0

    power = (
        distance
        / average_distance
    )

    return max(
        0.0,
        min(
            1.0,
            power,
        ),
    )


def build_equity_method(
    trades,
    raw_equity,
    method,
):
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "equity",
                "trade_return_pct",
                "buying_power",
                "effective_return_pct",
            ]
        )

    equity_times = (
        raw_equity[
            "timestamp"
        ].to_numpy()
    )

    equity_values = (
        raw_equity[
            "equity"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    trade_entries = (
        trades[
            "entry_time"
        ].to_numpy()
    )

    trade_exits = (
        trades[
            "exit_time"
        ].to_numpy()
    )

    trade_returns = (
        trades[
            "return_pct"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    output_times = []
    output_equity = []
    output_returns = []
    output_power = []
    output_effective = []

    equity = STARTING_EQUITY

    for i in range(len(trades)):

        entry_time = trade_entries[i]

        available = np.searchsorted(
            equity_times,
            entry_time,
            side="left",
        )

        if available < MA_PERIOD:
            continue

        window_start = (
            available - MA_PERIOD
        )

        window = equity_values[
            window_start:available
        ]

        ma = np.mean(window)

        if ma <= 0:
            continue

        current = (
            equity_values[
                available - 1
            ]
        )

        distance = (
            (current - ma)
            / ma
        ) * 100.0

        if current <= ma:
            power = 0.0

        elif method == 1:
            power = 1.0

        else:

            distances = []

            for j in range(
                MA_PERIOD - 1,
                available,
            ):

                start = (
                    j - MA_PERIOD + 1
                )

                historical_window = (
                    equity_values[
                        start:j + 1
                    ]
                )

                historical_ma = (
                    np.mean(
                        historical_window
                    )
                )

                if historical_ma <= 0:
                    continue

                historical_distance = (
                    (
                        historical_window[-1]
                        - historical_ma
                    )
                    / historical_ma
                ) * 100.0

                distances.append(
                    historical_distance
                )

            if len(distances) == 0:
                power = 1.0

            else:

                average_distance = (
                    np.mean(
                        np.asarray(
                            distances
                        )
                    )
                )

                if average_distance <= 0:
                    power = 1.0
                else:
                    power = (
                        distance
                        / average_distance
                    )

                    power = max(
                        0.0,
                        min(
                            1.0,
                            power,
                        ),
                    )

        if power <= 0:
            continue

        effective = (
            trade_returns[i]
            * power
        )

        equity *= (
            1.0
            + effective / 100.0
        )

        equity = round(
            equity,
            3,
        )

        output_times.append(
            trade_exits[i]
        )

        output_equity.append(
            equity
        )

        output_returns.append(
            trade_returns[i]
        )

        output_power.append(
            power
        )

        output_effective.append(
            effective
        )

    return pd.DataFrame(
        {
            "timestamp":
                output_times,

            "equity":
                output_equity,

            "trade_return_pct":
                output_returns,

            "buying_power":
                output_power,

            "effective_return_pct":
                output_effective,
        }
    )


def build_stock_method(
    trades,
    df,
    ema_column,
    method,
):
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "equity",
                "trade_return_pct",
                "buying_power",
                "effective_return_pct",
            ]
        )

    timestamps = (
        df["timestamp"].to_numpy()
    )

    close = (
        df["close"]
        .to_numpy(
            dtype=np.float64
        )
    )

    ema = (
        df[ema_column]
        .to_numpy(
            dtype=np.float64
        )
    )

    valid = ~np.isnan(ema)

    historical_distance = np.full(
        len(df),
        np.nan,
    )

    historical_distance[valid] = (
        (
            close[valid]
            - ema[valid]
        )
        / ema[valid]
    ) * 100.0

    trade_entries = (
        trades["entry_time"]
        .to_numpy()
    )

    trade_exits = (
        trades["exit_time"]
        .to_numpy()
    )

    returns = (
        trades["return_pct"]
        .to_numpy(
            dtype=np.float64
        )
    )

    equity = STARTING_EQUITY

    output_times = []
    output_equity = []
    output_returns = []
    output_power = []
    output_effective = []

    for i in range(len(trades)):

        entry = trade_entries[i]

        index = np.searchsorted(
            timestamps,
            entry,
            side="right",
        ) - 1

        if index < 0:
            continue

        if np.isnan(
            historical_distance[index]
        ):
            continue

        current_price = close[index]
        current_ema = ema[index]

        if current_price <= current_ema:
            continue

        if method == 1:

            power = 1.0

        else:

            history = (
                historical_distance[
                    :index + 1
                ]
            )

            history = history[
                ~np.isnan(history)
            ]

            if len(history) == 0:
                power = 1.0

            else:

                average_distance = (
                    np.mean(history)
                )

                if average_distance <= 0:
                    power = 1.0

                else:

                    current_distance = (
                        historical_distance[
                            index
                        ]
                    )

                    power = (
                        current_distance
                        / average_distance
                    )

                    power = max(
                        0.0,
                        min(
                            1.0,
                            power,
                        ),
                    )

        if power <= 0:
            continue

        effective = (
            returns[i]
            * power
        )

        equity *= (
            1.0
            + effective / 100.0
        )

        equity = round(
            equity,
            3,
        )

        output_times.append(
            trade_exits[i]
        )

        output_equity.append(
            equity
        )

        output_returns.append(
            returns[i]
        )

        output_power.append(
            power
        )

        output_effective.append(
            effective
        )

    return pd.DataFrame(
        {
            "timestamp":
                output_times,

            "equity":
                output_equity,

            "trade_return_pct":
                output_returns,

            "buying_power":
                output_power,

            "effective_return_pct":
                output_effective,
        }
    )


def build_all_equity_curves(
    trades,
    df,
):
    raw = calculate_raw_equity(
        trades
    )

    equity_1 = build_equity_method(
        trades,
        raw,
        method=1,
    )

    equity_2 = build_equity_method(
        trades,
        raw,
        method=2,
    )

    stock_1 = build_stock_method(
        trades,
        df,
        "ema_21",
        method=1,
    )

    stock_2 = build_stock_method(
        trades,
        df,
        "ema_21",
        method=2,
    )

    return {
        "raw": raw,
        "equity_method_1": equity_1,
        "equity_method_2": equity_2,
        "stock_method_1": stock_1,
        "stock_method_2": stock_2,
    }