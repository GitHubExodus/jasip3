from strategies import generate_strategies

from equity_methods import (
    build_all_equity_curves,
)


def run_timeframe(
    df,
    timeframe,
):
    strategies = generate_strategies(
        df,
        timeframe,
    )

    results = {}

    for strategy_name, trades in (
        strategies.items()
    ):

        print(
            f"  {strategy_name}: "
            f"{len(trades):,} trades",
            flush=True,
        )

        curves = build_all_equity_curves(
            trades,
            df,
        )

        results[strategy_name] = {
            "trades": trades,
            "equity": curves,
        }

    return results


def run_all_timeframes(
    timeframe_data,
):
    results = {}

    for timeframe, df in (
        timeframe_data.items()
    ):

        print(
            f"\nRunning {timeframe}...",
            flush=True,
        )

        results[timeframe] = (
            run_timeframe(
                df,
                timeframe,
            )
        )

    return results