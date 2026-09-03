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
        returns
    )

    print(
        f"    Raw equity: "
        f"{len(returns):,} / {len(returns):,} trades",
        flush=True,
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


def build_equity_method(
    trades,
    raw_equity,
    method,
    strategy_name="strategy",
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
        ].to_numpy(
            dtype=np.float64
        )
    )

    total = len(trades)

    output_times = []
    output_equity = []
    output_returns = []
    output_power = []
    output_effective = []

    equity = STARTING_EQUITY

    method_name = (
        "Equity Method 1"
        if method == 1
        else "Equity Method 2"
    )

    print(
        f"    {strategy_name}: "
        f"{method_name}",
        flush=True,
    )

    processed = 0
    last_logged = 0

    for i in range(total):

        entry_time = trade_entries[i]

        available = np.searchsorted(
            equity_times,
            entry_time,
            side="left",
        )

        if available < MA_PERIOD:
            processed += 1
            continue

        window_start = (
            available - MA_PERIOD
        )

        window = equity_values[
            window_start:available
        ]

        ma = np.mean(window)

        if ma <= 0:
            processed += 1
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

                historical_ma = np.mean(
                    historical_window
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
                    abs(historical_distance)
                )

            if len(distances) == 0:

                power = 1.0

            else:

                average_distance = np.mean(
                    np.asarray(
                        distances,
                        dtype=np.float64,
                    )
                )

                if average_distance <= 0:

                    power = 1.0

                else:

                    power = (
                        abs(distance)
                        / average_distance
                    )

                    power = max(
                        0.0,
                        min(
                            1.0,
                            power,
                        )
                    )

        if power > 0:

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

        processed += 1

        if (
            processed - last_logged
            >= PROGRESS_INTERVAL
            or processed == total
        ):

            percent = (
                processed
                / total
            ) * 100.0

            print(
                f"      "
                f"{processed:,} / "
                f"{total:,} trades "
                f"({percent:.1f}%)",
                flush=True,
            )

            last_logged = processed

    print(
        f"    {method_name} finished: "
        f"{processed:,} / {total:,}",
        flush=True,
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
    strategy_name="strategy",
):
    print(
        f"    Building raw equity...",
        flush=True,
    )

    raw = calculate_raw_equity(
        trades
    )

    equity_1 = build_equity_method(
        trades,
        raw,
        method=1,
        strategy_name=strategy_name,
    )

    equity_2 = build_equity_method(
        trades,
        raw,
        method=2,
        strategy_name=strategy_name,
    )

    return {
        "raw": raw,
        "equity_method_1": equity_1,
        "equity_method_2": equity_2,
    }
