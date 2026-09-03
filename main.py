import sys
import traceback

from cloud_access import (
    get_stock_symbols,
    download_raw_stock,
    save_dataframe,
    log,
)

from indicators import (
    build_all_timeframes,
)

from strategies import (
    generate_strategies,
)

from backtest import (
    build_all_equity_curves,
)

from save_results import (
    save_timeframe_results,
)


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def run_stock(
    symbol,
):
    symbol = symbol.upper()

    log("")
    log("=" * 70)
    log(
        f"EQUITY CURVE TEST: {symbol}"
    )
    log("=" * 70)

    # --------------------------------------------------------
    # Download raw data.
    # --------------------------------------------------------

    raw = download_raw_stock(
        symbol
    )

    log(
        f"Loaded "
        f"{len(raw):,} "
        f"1-minute rows"
    )

    # --------------------------------------------------------
    # Create all five timeframes.
    # --------------------------------------------------------

    timeframe_data = (
        build_all_timeframes(
            raw
        )
    )

    # --------------------------------------------------------
    # Process every timeframe.
    # --------------------------------------------------------

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

        log("")
        log("=" * 70)
        log(
            f"TIMEFRAME "
            f"{timeframe_number}/"
            f"{total_timeframes}: "
            f"{timeframe}"
        )
        log("=" * 70)

        # ----------------------------------------------------
        # Save indicators.
        # ----------------------------------------------------

        indicator_key = (
            f"equity_test/"
            f"{symbol}/"
            f"indicators/"
            f"{timeframe}.parquet"
        )

        save_dataframe(
            df,
            indicator_key,
        )

        # ----------------------------------------------------
        # Generate all 8 strategies.
        # ----------------------------------------------------

        strategy_results = (
            generate_strategies(
                df
            )
        )

        # ----------------------------------------------------
        # Build the 25 equity curves
        # for every strategy.
        # ----------------------------------------------------

        final_results = {}

        for strategy_name, (
            strategy_combinations
        ) in strategy_results.items():

            print(
                f"    Building equity curves "
                f"for {strategy_name}...",
                flush=True,
            )

            final_results[
                strategy_name
            ] = (
                build_all_equity_curves(
                    strategy_combinations
                )
            )

        # ----------------------------------------------------
        # Save timeframe.
        # ----------------------------------------------------

        save_timeframe_results(
            symbol,
            timeframe,
            final_results,
        )

        log("")
        log(
            f"Finished timeframe: "
            f"{timeframe}"
        )

    log("")
    log("=" * 70)
    log(
        f"FINISHED STOCK: {symbol}"
    )
    log("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # If a symbol was supplied:
    #
    #     python3.11 main.py AAPL
    #
    # process only that symbol.
    #
    # Otherwise process symbols.txt.
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        symbols = [
            sys.argv[1]
            .upper()
        ]

    else:

        symbols = (
            get_stock_symbols()
        )

    log(
        f"Stocks to process: "
        f"{len(symbols):,}"
    )

    for number, symbol in enumerate(
        symbols,
        start=1,
    ):

        log("")
        log(
            f"STOCK "
            f"{number}/{len(symbols)}: "
            f"{symbol}"
        )

        try:

            run_stock(
                symbol
            )

        except Exception:

            log(
                f"ERROR processing "
                f"{symbol}"
            )

            traceback.print_exc()

            # Continue to the next stock.
            continue


if __name__ == "__main__":
    main()