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

    total_strategies = len(strategies)

    for strategy_number, (
        strategy_name,
        trades,
    ) in enumerate(
        strategies.items(),
        start=1,
    ):

        print(
            f"\n  [{strategy_number}/{total_strategies}] "
            f"{strategy_name}",
            flush=True,
        )

        print(
            f"    Trades: {len(trades):,}",
            flush=True,
        )

        curves = build_all_equity_curves(
            trades,
            strategy_name,
        )

        results[strategy_name] = {
            "trades": trades,
            "equity": curves,
        }

        print(
            f"    {strategy_name} complete",
            flush=True,
        )

    return results


def run_all_timeframes(
    timeframe_data,
):
    results = {}

    total_timeframes = len(
        timeframe_data
    )

    for timeframe_number, (
        timeframe,
        df,
    ) in enumerate(
        timeframe_data.items(),
        start=1,
    ):

        print(
            f"\n"
            f"{'=' * 70}\n"
            f"TIMEFRAME "
            f"{timeframe_number}/{total_timeframes}: "
            f"{timeframe}\n"
            f"{'=' * 70}",
            flush=True,
        )

        results[timeframe] = (
            run_timeframe(
                df,
                timeframe,
            )
        )

        print(
            f"\nFinished timeframe: "
            f"{timeframe}",
            flush=True,
        )

    return results
