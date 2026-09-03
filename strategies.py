import numpy as np
import pandas as pd

from numba import njit


RR_SETTINGS = {
    "5m": (0.005, 2.0),
    "30m": (0.010, 2.0),
    "1d": (0.020, 2.0),
    "1w": (0.050, 2.0),
}


@njit
def strategy_1_numba(
    close,
    high,
    low,
    ema3,
    ema5,
    stop_pct,
    rr,
):
    n = len(close)

    entry_indices = np.empty(
        n,
        dtype=np.int64,
    )

    exit_indices = np.empty(
        n,
        dtype=np.int64,
    )

    entry_prices = np.empty(
        n,
        dtype=np.float64,
    )

    exit_prices = np.empty(
        n,
        dtype=np.float64,
    )

    returns = np.empty(
        n,
        dtype=np.float64,
    )

    count = 0

    active = False

    entry_index = -1
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0

    for i in range(1, n):

        if (
            np.isnan(ema3[i])
            or np.isnan(ema5[i])
        ):
            continue

        cross_up = (
            ema3[i - 1] <= ema5[i - 1]
            and ema3[i] > ema5[i]
        )

        if not active and cross_up:

            active = True

            entry_index = i
            entry_price = close[i]

            stop_price = (
                entry_price
                * (1.0 - stop_pct)
            )

            target_price = (
                entry_price
                * (
                    1.0
                    + stop_pct * rr
                )
            )

            continue

        if not active:
            continue

        stop_hit = (
            low[i] <= stop_price
        )

        target_hit = (
            high[i] >= target_price
        )

        if stop_hit:

            entry_indices[count] = (
                entry_index
            )

            exit_indices[count] = i

            entry_prices[count] = (
                entry_price
            )

            exit_prices[count] = (
                stop_price
            )

            returns[count] = (
                stop_price
                / entry_price
                - 1.0
            ) * 100.0

            count += 1
            active = False

        elif target_hit:

            entry_indices[count] = (
                entry_index
            )

            exit_indices[count] = i

            entry_prices[count] = (
                entry_price
            )

            exit_prices[count] = (
                target_price
            )

            returns[count] = (
                target_price
                / entry_price
                - 1.0
            ) * 100.0

            count += 1
            active = False

    if active:

        i = n - 1

        entry_indices[count] = (
            entry_index
        )

        exit_indices[count] = i

        entry_prices[count] = (
            entry_price
        )

        exit_prices[count] = (
            close[i]
        )

        returns[count] = (
            close[i]
            / entry_price
            - 1.0
        ) * 100.0

        count += 1

    return (
        entry_indices[:count],
        exit_indices[:count],
        entry_prices[:count],
        exit_prices[:count],
        returns[:count],
    )


@njit
def strategy_break_ema_numba(
    close,
    high,
    ema,
):
    n = len(close)

    entry_indices = np.empty(
        n,
        dtype=np.int64,
    )

    exit_indices = np.empty(
        n,
        dtype=np.int64,
    )

    entry_prices = np.empty(
        n,
        dtype=np.float64,
    )

    exit_prices = np.empty(
        n,
        dtype=np.float64,
    )

    returns = np.empty(
        n,
        dtype=np.float64,
    )

    count = 0

    position_indices = np.empty(
        n,
        dtype=np.int64,
    )

    position_prices = np.empty(
        n,
        dtype=np.float64,
    )

    position_count = 0

    for i in range(1, n):

        if np.isnan(ema[i]):
            continue

        cross_down = (
            close[i - 1] >= ema[i - 1]
            and close[i] < ema[i]
        )

        if cross_down:

            for p in range(
                position_count
            ):
                entry_index = (
                    position_indices[p]
                )

                entry_price = (
                    position_prices[p]
                )

                entry_indices[count] = (
                    entry_index
                )

                exit_indices[count] = i

                entry_prices[count] = (
                    entry_price
                )

                exit_prices[count] = (
                    close[i]
                )

                returns[count] = (
                    close[i]
                    / entry_price
                    - 1.0
                ) * 100.0

                count += 1

            position_count = 0

            continue

        above_ema = (
            close[i] > ema[i]
        )

        breaks_high = (
            high[i] > high[i - 1]
        )

        if above_ema and breaks_high:

            position_indices[
                position_count
            ] = i

            position_prices[
                position_count
            ] = high[i]

            position_count += 1

    if position_count > 0:

        i = n - 1

        for p in range(
            position_count
        ):

            entry_index = (
                position_indices[p]
            )

            entry_price = (
                position_prices[p]
            )

            entry_indices[count] = (
                entry_index
            )

            exit_indices[count] = i

            entry_prices[count] = (
                entry_price
            )

            exit_prices[count] = (
                close[i]
            )

            returns[count] = (
                close[i]
                / entry_price
                - 1.0
            ) * 100.0

            count += 1

    return (
        entry_indices[:count],
        exit_indices[:count],
        entry_prices[:count],
        exit_prices[:count],
        returns[:count],
    )


@njit
def strategy_cross_ema_numba(
    close,
    ema,
):
    n = len(close)

    entry_indices = np.empty(
        n,
        dtype=np.int64,
    )

    exit_indices = np.empty(
        n,
        dtype=np.int64,
    )

    entry_prices = np.empty(
        n,
        dtype=np.float64,
    )

    exit_prices = np.empty(
        n,
        dtype=np.float64,
    )

    returns = np.empty(
        n,
        dtype=np.float64,
    )

    count = 0

    active = False
    entry_index = -1
    entry_price = 0.0

    for i in range(1, n):

        if np.isnan(ema[i]):
            continue

        cross_up = (
            close[i - 1] <= ema[i - 1]
            and close[i] > ema[i]
        )

        cross_down = (
            close[i - 1] >= ema[i - 1]
            and close[i] < ema[i]
        )

        if not active and cross_up:

            active = True

            entry_index = i
            entry_price = close[i]

        elif active and cross_down:

            entry_indices[count] = (
                entry_index
            )

            exit_indices[count] = i

            entry_prices[count] = (
                entry_price
            )

            exit_prices[count] = (
                close[i]
            )

            returns[count] = (
                close[i]
                / entry_price
                - 1.0
            ) * 100.0

            count += 1

            active = False

    if active:

        i = n - 1

        entry_indices[count] = (
            entry_index
        )

        exit_indices[count] = i

        entry_prices[count] = (
            entry_price
        )

        exit_prices[count] = (
            close[i]
        )

        returns[count] = (
            close[i]
            / entry_price
            - 1.0
        ) * 100.0

        count += 1

    return (
        entry_indices[:count],
        exit_indices[:count],
        entry_prices[:count],
        exit_prices[:count],
        returns[:count],
    )


def make_trade_dataframe(
    df,
    result,
):
    (
        entry_indices,
        exit_indices,
        entry_prices,
        exit_prices,
        returns,
    ) = result

    if len(returns) == 0:

        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "return_pct",
            ]
        )

    return pd.DataFrame(
        {
            "entry_time":
                df["timestamp"]
                .iloc[entry_indices]
                .to_numpy(),

            "exit_time":
                df["timestamp"]
                .iloc[exit_indices]
                .to_numpy(),

            "entry_price":
                entry_prices,

            "exit_price":
                exit_prices,

            "return_pct":
                returns,
        }
    )


def strategy_1_ema_rr(
    df,
    timeframe,
):
    stop_pct, rr = RR_SETTINGS[
        timeframe
    ]

    return make_trade_dataframe(
        df,
        strategy_1_numba(
            df["close"].to_numpy(
                dtype=np.float64
            ),
            df["high"].to_numpy(
                dtype=np.float64
            ),
            df["low"].to_numpy(
                dtype=np.float64
            ),
            df["ema_3"].to_numpy(
                dtype=np.float64
            ),
            df["ema_5"].to_numpy(
                dtype=np.float64
            ),
            stop_pct,
            rr,
        ),
    )


def strategy_price_break_ema(
    df,
    ema_column,
):
    return make_trade_dataframe(
        df,
        strategy_break_ema_numba(
            df["close"].to_numpy(
                dtype=np.float64
            ),
            df["high"].to_numpy(
                dtype=np.float64
            ),
            df[ema_column].to_numpy(
                dtype=np.float64
            ),
        ),
    )


def strategy_price_cross_ema(
    df,
    ema_column,
):
    return make_trade_dataframe(
        df,
        strategy_cross_ema_numba(
            df["close"].to_numpy(
                dtype=np.float64
            ),
            df[ema_column].to_numpy(
                dtype=np.float64
            ),
        ),
    )


def generate_strategies(
    df,
    timeframe,
):
    return {
        "strategy_1":
            strategy_1_ema_rr(
                df,
                timeframe,
            ),

        "strategy_2":
            strategy_price_break_ema(
                df,
                "ema_21",
            ),

        "strategy_3":
            strategy_price_cross_ema(
                df,
                "ema_21",
            ),

        "strategy_4":
            strategy_price_break_ema(
                df,
                "ema_200",
            ),

        "strategy_5":
            strategy_price_cross_ema(
                df,
                "ema_200",
            ),
    }
