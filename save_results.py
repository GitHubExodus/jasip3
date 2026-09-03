from cloud_access import save_dataframe


def save_all_results(
    symbol,
    results,
):
    for timeframe, timeframe_results in (
        results.items()
    ):

        for strategy_name, strategy_results in (
            timeframe_results.items()
        ):

            trades = strategy_results[
                "trades"
            ]

            trade_key = (
                f"equity_test/"
                f"{symbol}/"
                f"{timeframe}/"
                f"trades/"
                f"{strategy_name}.parquet"
            )

            save_dataframe(
                trades,
                trade_key,
            )

            curves = strategy_results[
                "equity"
            ]

            for curve_name, curve in (
                curves.items()
            ):

                key = (
                    f"equity_test/"
                    f"{symbol}/"
                    f"{timeframe}/"
                    f"{strategy_name}/"
                    f"{curve_name}.parquet"
                )

                save_dataframe(
                    curve,
                    key,
                )
