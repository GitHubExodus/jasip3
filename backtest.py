import numpy as np
import pandas as pd

from numba import njit


# ============================================================
# CONFIGURATION
# ============================================================

STARTING_EQUITY = 100.000


# ============================================================
# EQUITY CALCULATION
# ============================================================

@njit
def calculate_equity_numba(
    returns,
):
    n = len(
        returns
    )

    equity = np.empty(
        n,
        dtype=np.float64,
    )

    value = STARTING_EQUITY

    for i in range(n):

        value *= (
            1.0
            + returns[i]
            / 100.0
        )

        # Keep equity at 3 decimals.
        value = np.round(
            value,
            3,
        )

        equity[i] = value

    return equity


# ============================================================
# BUILD ONE EQUITY CURVE
# ============================================================

def build_equity_curve(
    trades,
):
    if trades.empty:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "return_pct",
                "equity",
            ]
        )

    trades = (
        trades
        .sort_values(
            "exit_time"
        )
        .reset_index(
            drop=True
        )
    )

    returns = np.asarray(
        trades["return_pct"],
        dtype=np.float64,
    )

    equity = (
        calculate_equity_numba(
            returns
        )
    )

    result = pd.DataFrame(
        {
            "timestamp":
                trades[
                    "exit_time"
                ].to_numpy(),

            "entry_time":
                trades[
                    "entry_time"
                ].to_numpy(),

            "exit_time":
                trades[
                    "exit_time"
                ].to_numpy(),

            "entry_price":
                trades[
                    "entry_price"
                ].to_numpy(),

            "exit_price":
                trades[
                    "exit_price"
                ].to_numpy(),

            "return_pct":
                returns,

            "equity":
                equity,
        }
    )

    return result


# ============================================================
# BUILD ALL EQUITY CURVES
# ============================================================

def build_all_equity_curves(
    strategy_results,
):
    results = {}

    for (
        combination_number,
        trades,
    ) in strategy_results.items():

        stop_pct = (
            trades[
                "stop_loss_pct"
            ].iloc[0]
            if not trades.empty
            else 0.0
        )

        rr = (
            trades[
                "risk_reward"
            ].iloc[0]
            if not trades.empty
            else 0.0
        )

        curve = (
            build_equity_curve(
                trades
            )
        )

        results[
            combination_number
        ] = {
            "trades": trades,
            "equity": curve,
            "stop_loss_pct":
                stop_pct,
            "risk_reward":
                rr,
        }

    return results