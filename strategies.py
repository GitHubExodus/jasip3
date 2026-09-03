import numpy as np
import pandas as pd

from numba import njit


# ============================================================
# EMA STRATEGIES
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


# ============================================================
# EXIT CONFIGURATION
# ============================================================

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
# BUILD 25 COMBINATIONS
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
# EMA CROSSOVER
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
            or
            np.isnan(
                slow_ema[i - 1]
            )
            or
            np.isnan(
                fast_ema[i]
            )
            or
            np.isnan(
                slow_ema[i]
            )
        ):
            continue

        # Fast EMA crosses ABOVE slow EMA.
        if (
            fast_ema[i - 1]
            <=
            slow_ema[i - 1]
            and
            fast_ema[i]
            >
            slow_ema[i]
        ):

            signals[i] = 1

    return signals


# ============================================================
# SIMULATE ALL 25 EXIT CONFIGURATIONS
# ============================================================

@njit
def simulate_all_rr_numba(
    close,
    high,
    low,
    signal_indices,
    stop_percentages,
    risk_rewards,
):

    signal_count = len(
        signal_indices
    )

    combination_count = len(
        stop_percentages
    )

    # Maximum number of trades per
    # configuration is number of signals.
    entry_indices = np.full(
        (
            combination_count,
            signal_count + 1,
        ),
        -1,
        dtype=np.int64,
    )

    exit_indices = np.full(
        (
            combination_count,
            signal_count + 1,
        ),
        -1,
        dtype=np.int64,
    )

    entry_prices = np.zeros(
        (
            combination_count,
            signal_count + 1,
        ),
        dtype=np.float64,
    )

    exit_prices = np.zeros(
        (
            combination_count,
            signal_count + 1,
        ),
        dtype=np.float64,
    )

    returns = np.zeros(
        (
            combination_count,
            signal_count + 1,
        ),
        dtype=np.float64,
    )

    trade_counts = np.zeros(
        combination_count,
        dtype=np.int64,
    )

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

    n = len(close)

    signal_pointer = 0

    for i in range(n):

        # ----------------------------------------------------
        # CHECK ACTIVE TRADES
        # ----------------------------------------------------

        for c in range(
            combination_count
        ):

            if active[c] == 0:
                continue

            stop_hit = (
                low[i]
                <=
                stop_prices[c]
            )

            target_hit = (
                high[i]
                >=
                target_prices[c]
            )

            # Stop loss gets priority if
            # both occur on same candle.
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
                    *
                    risk_rewards[c]
                    *
                    100.0
                )

                trade_counts[c] += 1

                active[c] = 0

        # ----------------------------------------------------
        # NEW EMA CROSSOVER
        # ----------------------------------------------------

        if (
            signal_pointer
            <
            signal_count
        ):

            if (
                signal_indices[
                    signal_pointer
                ]
                ==
                i
            ):

                entry_price = (
                    close[i]
                )

                for c in range(
                    combination_count
                ):

                    # One active trade per
                    # strategy / RR combination.
                    if active[c] != 0:
                        continue

                    stop_pct = (
                        stop_percentages[c]
                    )

                    rr = (
                        risk_rewards[c]
                    )

                    active[c] = 1

                    active_entry_index[
                        c
                    ] = i

                    active_entry_price[
                        c
                    ] = entry_price

                    stop_prices[c] = (
                        entry_price
                        *
                        (
                            1.0
                            -
                            stop_pct
                        )
                    )

                    target_prices[c] = (
                        entry_price
                        *
                        (
                            1.0
                            +
                            stop_pct
                            *
                            rr
                        )
                    )

                signal_pointer += 1

    # --------------------------------------------------------
    # CLOSE ANY REMAINING TRADES
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
            /
            entry_price
            -
            1.0
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
# BUILD EQUITY CURVES
# ============================================================

def build_equity_curves(
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
    # FIND CROSSOVER SIGNALS
    # --------------------------------------------------------

    signals = (
        find_crossovers_numba(
            fast_ema,
            slow_ema,
        )
    )

    signal_indices = (
        np.flatnonzero(
            signals
        ).astype(
            np.int64
        )
    )

    if len(signal_indices) == 0:

        return pd.DataFrame(
            columns=[
                "timestamp"
            ]
        )

    # --------------------------------------------------------
    # SIMULATE ALL 25 COMBINATIONS
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
        signal_indices,
        STOP_ARRAY,
        RR_ARRAY,
    )

    timestamps = df[
        "timestamp"
    ].to_numpy()

    # --------------------------------------------------------
    # CREATE ONE EQUITY SERIES PER RR CONFIGURATION
    # --------------------------------------------------------

    curve_series = []

    for c in range(25):

        count = (
            trade_counts[c]
        )

        if count == 0:
            continue

        exits = exit_indices[
            c,
            :count
        ]

        trade_returns = returns[
            c,
            :count
        ]

        equity_values = []

        equity = 100.000

        for r in trade_returns:

            equity *= (
                1.0
                +
                r / 100.0
            )

            equity = round(
                equity,
                3,
            )

            equity_values.append(
                equity
            )

        column_name = (
            f"equity_sl_"
            f"{STOP_ARRAY[c] * 100.0:g}"
            f"_rr_"
            f"{RR_ARRAY[c]:g}"
        )

        curve = pd.DataFrame(
            {
                "timestamp":
                    timestamps[exits],

                column_name:
                    np.asarray(
                        equity_values,
                        dtype=np.float64,
                    ),
            }
        )

        curve_series.append(
            curve
        )

    # --------------------------------------------------------
    # MERGE ALL 25 CURVES BY EXIT TIMESTAMP
    # --------------------------------------------------------

    result = None

    for curve in curve_series:

        if result is None:

            result = curve

        else:

            result = result.merge(
                curve,
                on="timestamp",
                how="outer",
            )

    # --------------------------------------------------------
    # SORT BY TIME
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return result


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

        equity = (
            build_equity_curves(
                df,
                strategy_number,
                fast_period,
                slow_period,
            )
        )

        results[
            f"strategy_{strategy_number}"
        ] = equity

    return results