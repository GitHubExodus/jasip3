import numpy as np
import pandas as pd

from numba import njit


# ============================================================
# CONFIGURATION
# ============================================================

EMA_PAIRS = (
    (3, 5),
    (5, 9),
    (9, 14),
    (14, 21),
    (21, 30),
    (30, 50),
    (50, 100),
    (100, 200),
)

STOP_LOSSES = (
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
)

RISK_REWARDS = (
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
)


# ============================================================
# BUILD 25 EXIT COMBINATIONS
# ============================================================

def build_exit_combinations():

    stop_values = []
    rr_values = []

    for stop_pct in STOP_LOSSES:

        for rr in RISK_REWARDS:

            stop_values.append(
                stop_pct
            )

            rr_values.append(
                rr
            )

    return (
        np.asarray(
            stop_values,
            dtype=np.float64,
        ),
        np.asarray(
            rr_values,
            dtype=np.float64,
        ),
    )


STOP_ARRAY, RR_ARRAY = (
    build_exit_combinations()
)


# ============================================================
# FIND EMA CROSSOVER SIGNALS
# ============================================================

@njit
def find_crossovers_numba(
    fast_ema,
    slow_ema,
):
    n = len(
        fast_ema
    )

    signals = np.zeros(
        n,
        dtype=np.uint8,
    )

    for i in range(
        1,
        n,
    ):

        if (
            np.isnan(
                fast_ema[i - 1]
            )
            or np.isnan(
                slow_ema[i - 1]
            )
            or np.isnan(
                fast_ema[i]
            )
            or np.isnan(
                slow_ema[i]
            )
        ):
            continue

        if (
            fast_ema[i - 1]
            <= slow_ema[i - 1]
            and
            fast_ema[i]
            > slow_ema[i]
        ):
            signals[i] = 1

    return signals


# ============================================================
# SIMULATE ALL 25 RR COMBINATIONS
# ============================================================

@njit
def simulate_all_rr_numba(
    close,
    high,
    low,
    signals,
    stop_percentages,
    risk_rewards,
):
    n = len(close)

    combination_count = (
        len(stop_percentages)
    )

    # --------------------------------------------------------
    # Maximum possible trades per combination is n.
    # --------------------------------------------------------

    entry_indices = np.full(
        (
            combination_count,
            n,
        ),
        -1,
        dtype=np.int64,
    )

    exit_indices = np.full(
        (
            combination_count,
            n,
        ),
        -1,
        dtype=np.int64,
    )

    entry_prices = np.zeros(
        (
            combination_count,
            n,
        ),
        dtype=np.float64,
    )

    exit_prices = np.zeros(
        (
            combination_count,
            n,
        ),
        dtype=np.float64,
    )

    returns = np.zeros(
        (
            combination_count,
            n,
        ),
        dtype=np.float64,
    )

    trade_counts = np.zeros(
        combination_count,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # One independent active trade for every
    # SL/RR combination.
    # --------------------------------------------------------

    active = np.zeros(
        combination_count,
        dtype=np.uint8,
    )

    active_entry_index = np.full(
        combination_count,
        -1,
        dtype=np.int64,
    )

    active_entry_price = np.zeros(
        combination_count,
        dtype=np.float64,
    )

    stop_prices = np.zeros(
        combination_count,
        dtype=np.float64,
    )

    target_prices = np.zeros(
        combination_count,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Process bars chronologically.
    # --------------------------------------------------------

    for i in range(n):

        # ----------------------------------------------------
        # First process exits for already-active trades.
        #
        # We intentionally process exits before new entries
        # on the same bar.
        # ----------------------------------------------------

        for c in range(
            combination_count
        ):

            if active[c] == 0:
                continue

            stop_hit = (
                low[i]
                <= stop_prices[c]
            )

            target_hit = (
                high[i]
                >= target_prices[c]
            )

            if stop_hit:

                trade_number = (
                    trade_counts[c]
                )

                entry_indices[
                    c,
                    trade_number,
                ] = (
                    active_entry_index[c]
                )

                exit_indices[
                    c,
                    trade_number,
                ] = i

                entry_prices[
                    c,
                    trade_number,
                ] = (
                    active_entry_price[c]
                )

                exit_prices[
                    c,
                    trade_number,
                ] = (
                    stop_prices[c]
                )

                returns[
                    c,
                    trade_number,
                ] = (
                    -stop_percentages[c]
                    * 100.0
                )

                trade_counts[c] += 1

                active[c] = 0

            elif target_hit:

                trade_number = (
                    trade_counts[c]
                )

                entry_indices[
                    c,
                    trade_number,
                ] = (
                    active_entry_index[c]
                )

                exit_indices[
                    c,
                    trade_number,
                ] = i

                entry_prices[
                    c,
                    trade_number,
                ] = (
                    active_entry_price[c]
                )

                exit_prices[
                    c,
                    trade_number,
                ] = (
                    target_prices[c]
                )

                returns[
                    c,
                    trade_number,
                ] = (
                    stop_percentages[c]
                    * risk_rewards[c]
                    * 100.0
                )

                trade_counts[c] += 1

                active[c] = 0

        # ----------------------------------------------------
        # New EMA crossover entry.
        #
        # Entry occurs at the close of the signal bar.
        # ----------------------------------------------------

        if signals[i] == 1:

            for c in range(
                combination_count
            ):

                if active[c] != 0:
                    continue

                entry_price = close[i]

                stop_pct = (
                    stop_percentages[c]
                )

                rr = (
                    risk_rewards[c]
                )

                active[c] = 1

                active_entry_index[c] = i

                active_entry_price[c] = (
                    entry_price
                )

                stop_prices[c] = (
                    entry_price
                    * (
                        1.0
                        - stop_pct
                    )
                )

                target_prices[c] = (
                    entry_price
                    * (
                        1.0
                        + stop_pct
                        * rr
                    )
                )

    # --------------------------------------------------------
    # Close any remaining trades at final close.
    # --------------------------------------------------------

    final_index = n - 1

    for c in range(
        combination_count
    ):

        if active[c] == 0:
            continue

        trade_number = (
            trade_counts[c]
        )

        entry_price = (
            active_entry_price[c]
        )

        exit_price = (
            close[final_index]
        )

        entry_indices[
            c,
            trade_number,
        ] = (
            active_entry_index[c]
        )

        exit_indices[
            c,
            trade_number,
        ] = final_index

        entry_prices[
            c,
            trade_number,
        ] = entry_price

        exit_prices[
            c,
            trade_number,
        ] = exit_price

        returns[
            c,
            trade_number,
        ] = (
            exit_price
            / entry_price
            - 1.0
        ) * 100.0

        trade_counts[c] += 1

    return (
        entry_indices,
        exit_indices,
        entry_prices,
        exit_prices,
        returns,
        trade_counts,
    )


# ============================================================
# BUILD TRADE DATAFRAME
# ============================================================

def build_trade_dataframe(
    df,
    strategy_number,
    combination_number,
    entry_indices,
    exit_indices,
    entry_prices,
    exit_prices,
    returns,
):
    if len(returns) == 0:
        return pd.DataFrame()

    entry_times = (
        df["timestamp"]
        .iloc[entry_indices]
        .to_numpy()
    )

    exit_times = (
        df["timestamp"]
        .iloc[exit_indices]
        .to_numpy()
    )

    stop_pct = (
        STOP_ARRAY[
            combination_number
        ]
        * 100.0
    )

    rr = (
        RR_ARRAY[
            combination_number
        ]
    )

    return pd.DataFrame(
        {
            "strategy":
                strategy_number,

            "stop_loss_pct":
                stop_pct,

            "risk_reward":
                rr,

            "entry_time":
                entry_times,

            "exit_time":
                exit_times,

            "entry_price":
                entry_prices,

            "exit_price":
                exit_prices,

            "return_pct":
                returns,
        }
    )


# ============================================================
# RUN ONE STRATEGY
# ============================================================

def run_strategy(
    df,
    strategy_number,
    fast_period,
    slow_period,
):
    close = np.asarray(
        df["close"],
        dtype=np.float64,
    )

    high = np.asarray(
        df["high"],
        dtype=np.float64,
    )

    low = np.asarray(
        df["low"],
        dtype=np.float64,
    )

    fast_ema = np.asarray(
        df[
            f"ema_{fast_period}"
        ],
        dtype=np.float64,
    )

    slow_ema = np.asarray(
        df[
            f"ema_{slow_period}"
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Find EMA crossover signals once.
    # --------------------------------------------------------

    signals = (
        find_crossovers_numba(
            fast_ema,
            slow_ema,
        )
    )

    # --------------------------------------------------------
    # Simulate all 25 exits at once.
    # --------------------------------------------------------

    (
        entry_indices,
        exit_indices,
        entry_prices,
        exit_prices,
        returns,
        trade_counts,
    ) = simulate_all_rr_numba(
        close,
        high,
        low,
        signals,
        STOP_ARRAY,
        RR_ARRAY,
    )

    # --------------------------------------------------------
    # Convert Numba results into DataFrames.
    # --------------------------------------------------------

    results = {}

    for combination_number in range(25):

        count = (
            trade_counts[
                combination_number
            ]
        )

        trades = (
            build_trade_dataframe(
                df,
                strategy_number,
                combination_number,
                entry_indices[
                    combination_number,
                    :count,
                ],
                exit_indices[
                    combination_number,
                    :count,
                ],
                entry_prices[
                    combination_number,
                    :count,
                ],
                exit_prices[
                    combination_number,
                    :count,
                ],
                returns[
                    combination_number,
                    :count,
                ],
            )
        )

        results[
            combination_number
        ] = trades

    return results


# ============================================================
# RUN ALL 8 STRATEGIES
# ============================================================

def generate_strategies(
    df,
):
    results = {}

    total = len(
        EMA_PAIRS
    )

    for strategy_number, (
        fast_period,
        slow_period,
    ) in enumerate(
        EMA_PAIRS,
        start=1,
    ):

        print(
            f"    Strategy "
            f"{strategy_number}/{total}: "
            f"EMA {fast_period} -> "
            f"EMA {slow_period}",
            flush=True,
        )

        strategy_results = (
            run_strategy(
                df,
                strategy_number,
                fast_period,
                slow_period,
            )
        )

        results[
            f"strategy_{strategy_number}"
        ] = strategy_results

    return results