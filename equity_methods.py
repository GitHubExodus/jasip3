import numpy as np
import pandas as pd

from numba import njit


STARTING_EQUITY = 100.000
MA_PERIOD = 5
PROGRESS_INTERVAL = 10000


@njit
def calculate_raw_equity_numba(
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


@njit
def calculate_equity_methods_numba(
    raw_equity,
    trade_returns,
):
    n = len(raw_equity)

    method_1_equity = np.empty(
        n,
        dtype=np.float64,
    )

    method_2_equity = np.empty(
        n,
        dtype=np.float64,
    )

    method_1_power = np.zeros(
        n,
        dtype=np.float64,
    )

    method_2_power = np.zeros(
        n,
        dtype=np.float64,
    )

    method_1_effective = np.zeros(
        n,
        dtype=np.float64,
    )

    method_2_effective = np.zeros(
        n,
        dtype=np.float64,
    )

    method_1_value = STARTING_EQUITY
    method_2_value = STARTING_EQUITY

    # Running information needed for Method 2.
    distance_sum = 0.0
    distance_count = 0

    for i in range(n):

        # --------------------------------------------------------
        # Calculate the 5-bar MA using ONLY completed raw
        # equity bars available before the current trade.
        # --------------------------------------------------------

        if i < MA_PERIOD:

            method_1_equity[i] = (
                method_1_value
            )

            method_2_equity[i] = (
                method_2_value
            )

            continue

        ma_start = i - MA_PERIOD
        ma_end = i

        ma = np.mean(
            raw_equity[
                ma_start:ma_end
            ]
        )

        current_equity = raw_equity[i - 1]

        if ma <= 0:

            method_1_equity[i] = (
                method_1_value
            )

            method_2_equity[i] = (
                method_2_value
            )

            continue

        # --------------------------------------------------------
        # Current percentage distance from the MA.
        # --------------------------------------------------------

        current_distance = (
            (
                current_equity - ma
            )
            / ma
        ) * 100.0

        distance = abs(
            current_distance
        )

        # --------------------------------------------------------
        # METHOD 1
        #
        # Above MA  -> 100%
        # At/below  -> 0%
        # --------------------------------------------------------

        if current_equity > ma:

            power_1 = 1.0

            effective_1 = (
                trade_returns[i]
                * power_1
            )

            method_1_value *= (
                1.0
                + effective_1 / 100.0
            )

            method_1_value = np.round(
                method_1_value,
                3,
            )

            method_1_power[i] = (
                power_1
            )

            method_1_effective[i] = (
                effective_1
            )

        method_1_equity[i] = (
            method_1_value
        )

        # --------------------------------------------------------
        # METHOD 2
        #
        # Above MA -> current distance /
        #              average historical distance
        #
        # At/below MA -> 0%
        # --------------------------------------------------------

        if current_equity > ma:

            if distance_count == 0:

                power_2 = 1.0

            else:

                average_distance = (
                    distance_sum
                    / distance_count
                )

                if average_distance <= 0:

                    power_2 = 1.0

                else:

                    power_2 = (
                        distance
                        / average_distance
                    )

                    if power_2 > 1.0:
                        power_2 = 1.0

                    elif power_2 < 0.0:
                        power_2 = 0.0

            effective_2 = (
                trade_returns[i]
                * power_2
            )

            method_2_value *= (
                1.0
                + effective_2 / 100.0
            )

            method_2_value = np.round(
                method_2_value,
                3,
            )

            method_2_power[i] = (
                power_2
            )

            method_2_effective[i] = (
                effective_2
            )

        method_2_equity[i] = (
            method_2_value
        )

        # --------------------------------------------------------
        # Add the CURRENT distance to the historical pool
        # only AFTER the current trade's decision has been made.
        #
        # This prevents the current bar from influencing its
        # own average distance.
        # --------------------------------------------------------

        distance_sum += distance
        distance_count += 1

    return (
        method_1_equity,
        method_2_equity,
        method_1_power,
        method_2_power,
        method_1_effective,
        method_2_effective,
    )


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
    ).reset_index(
        drop=True
    )

    returns = trades[
        "return_pct"
    ].to_numpy(
        dtype=np.float64
    )

    print(
        f"    Calculating raw equity: "
        f"{len(returns):,} trades",
        flush=True,
    )

    equity = calculate_raw_equity_numba(
        returns
    )

    print(
        f"    Raw equity finished: "
        f"{len(returns):,} / "
        f"{len(returns):,} "
        f"(100.0%)",
        flush=True,
    )

    return pd.DataFrame(
        {
            "timestamp":
                trades[
                    "exit_time"
                ].to_numpy(),

            "equity":
                equity,

            "return_pct":
                returns,
        }
    )


def build_equity_methods(
    trades,
    raw_equity,
    strategy_name,
):
    if trades.empty:

        empty_columns = [
            "timestamp",
            "equity",
            "trade_return_pct",
            "buying_power",
            "effective_return_pct",
        ]

        return (
            pd.DataFrame(
                columns=empty_columns
            ),
            pd.DataFrame(
                columns=empty_columns
            ),
        )

    trades = trades.sort_values(
        "exit_time"
    ).reset_index(
        drop=True
    )

    raw_values = raw_equity[
        "equity"
    ].to_numpy(
        dtype=np.float64
    )

    returns = trades[
        "return_pct"
    ].to_numpy(
        dtype=np.float64
    )

    timestamps = trades[
        "exit_time"
    ].to_numpy()

    total = len(returns)

    print(
        f"    {strategy_name}: "
        f"Calculating Method 1 + Method 2 "
        f"from raw equity...",
        flush=True,
    )

    (
        method_1_equity,
        method_2_equity,
        method_1_power,
        method_2_power,
        method_1_effective,
        method_2_effective,
    ) = calculate_equity_methods_numba(
        raw_values,
        returns,
    )

    # ------------------------------------------------------------
    # Progress output.
    #
    # The actual calculation is already finished because Numba
    # processes the entire array. This prints progress afterward
    # so large runs still provide confirmation of the result.
    # ------------------------------------------------------------

    for start in range(
        PROGRESS_INTERVAL,
        total + PROGRESS_INTERVAL,
        PROGRESS_INTERVAL,
    ):

        completed = min(
            start,
            total,
        )

        percent = (
            completed
            / total
        ) * 100.0

        print(
            f"      Methods: "
            f"{completed:,} / "
            f"{total:,} "
            f"({percent:.1f}%)",
            flush=True,
        )

        if completed == total:
            break

    method_1 = pd.DataFrame(
        {
            "timestamp":
                timestamps,

            "equity":
                method_1_equity,

            "trade_return_pct":
                returns,

            "buying_power":
                method_1_power,

            "effective_return_pct":
                method_1_effective,
        }
    )

    method_2 = pd.DataFrame(
        {
            "timestamp":
                timestamps,

            "equity":
                method_2_equity,

            "trade_return_pct":
                returns,

            "buying_power":
                method_2_power,

            "effective_return_pct":
                method_2_effective,
        }
    )

    print(
        f"    Methods finished: "
        f"{total:,} / "
        f"{total:,} (100.0%)",
        flush=True,
    )

    return (
        method_1,
        method_2,
    )


def build_all_equity_curves(
    trades,
    strategy_name="strategy",
):
    # ------------------------------------------------------------
    # STEP 1
    # Calculate raw equity exactly once.
    # ------------------------------------------------------------

    raw = calculate_raw_equity(
        trades
    )

    # ------------------------------------------------------------
    # STEP 2
    # Filter the raw equity using Method 1 and Method 2.
    # ------------------------------------------------------------

    (
        equity_1,
        equity_2,
    ) = build_equity_methods(
        trades,
        raw,
        strategy_name,
    )

    return {
        "raw":
            raw,

        "equity_method_1":
            equity_1,

        "equity_method_2":
            equity_2,
    }
