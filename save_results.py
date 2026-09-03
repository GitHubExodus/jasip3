from cloud_access import save_dataframe


# ============================================================
# SAVE ONE TIMEFRAME
# ============================================================

def save_timeframe_results(
    symbol,
    timeframe,
    results,
):
    for strategy_name, strategy_results in (
        results.items()
    ):

        all_trades = []
        all_equity = []

        for (
            combination_number,
            combination_results,
        ) in strategy_results.items():

            trades = (
                combination_results[
                    "trades"
                ]
            )

            equity = (
                combination_results[
                    "equity"
                ]
            )

            if not trades.empty:
                all_trades.append(
                    trades
                )

            if not equity.empty:

                all_equity.append(
                    equity.assign(
                        stop_loss_pct=(
                            combination_results[
                                "stop_loss_pct"
                            ]
                        ),
                        risk_reward=(
                            combination_results[
                                "risk_reward"
                            ]
                        ),
                    )
                )

        # ----------------------------------------------------
        # Combine all 25 combinations.
        # ----------------------------------------------------

        if all_trades:

            trades_df = (
                __import__(
                    "pandas"
                ).concat(
                    all_trades,
                    ignore_index=True,
                )
            )

        else:

            trades_df = (
                __import__(
                    "pandas"
                ).DataFrame()
            )

        if all_equity:

            equity_df = (
                __import__(
                    "pandas"
                ).concat(
                    all_equity,
                    ignore_index=True,
                )
            )

        else:

            equity_df = (
                __import__(
                    "pandas"
                ).DataFrame()
            )

        # ----------------------------------------------------
        # Save trades.
        # ----------------------------------------------------

        trades_key = (
            f"equity_test/"
            f"{symbol}/"
            f"{timeframe}/"
            f"{strategy_name}/"
            f"trades.parquet"
        )

        save_dataframe(
            trades_df,
            trades_key,
        )

        # ----------------------------------------------------
        # Save equity curves.
        # ----------------------------------------------------

        equity_key = (
            f"equity_test/"
            f"{symbol}/"
            f"{timeframe}/"
            f"{strategy_name}/"
            f"equity.parquet"
        )

        save_dataframe(
            equity_df,
            equity_key,
        )


# ============================================================
# SAVE ALL RESULTS
# ============================================================

def save_all_results(
    symbol,
    results,
):
    for timeframe, timeframe_results in (
        results.items()
    ):

        save_timeframe_results(
            symbol,
            timeframe,
            timeframe_results,
        )