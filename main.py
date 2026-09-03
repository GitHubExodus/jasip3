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

from save_results import (
    save_strategy_equity,
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
    # DOWNLOAD RAW DATA
    # --------------------------------------------------------

    raw = download_raw_stock(
        symbol
    )

    log(
        f"Loaded "
        f"{len(raw):,} "
        f"raw rows"
    )

    # --------------------------------------------------------
    # BUILD TIMEFRAMES
    # --------------------------------------------------------

    timeframe_data = (
        build_all_timeframes(
            raw
        )
    )

    total_timeframes = len(
        timeframe_data
    )

    # --------------------------------------------------------
    # PROCESS EACH TIMEFRAME
    # --------------------------------------------------------

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
        # SAVE INDICATOR DATA
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
        # BUILD ALL 8 STRATEGIES
        # ----------------------------------------------------

        strategy_results = (
            generate_strategies(
                df
            )
        )

        # ----------------------------------------------------
        # SAVE EACH STRATEGY
        # ----------------------------------------------------

        for strategy_name, equity_df in (
            strategy_results.items()
        ):

            log(
                f"    Saving "
                f"{strategy_name}..."
            )

            save_strategy_equity(
                symbol,
                timeframe,
                strategy_name,
                equity_df,
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
    # ONE STOCK IF PROVIDED
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        symbols = [
            sys.argv[1].upper()
        ]

    # --------------------------------------------------------
    # OTHERWISE ALL STOCKS FROM symbols.txt
    # --------------------------------------------------------

    else:

        symbols = (
            get_stock_symbols()
        )

    log(
        f"Stocks to process: "
        f"{len(symbols):,}"
    )

    # --------------------------------------------------------
    # PROCESS STOCKS
    # --------------------------------------------------------

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

            continue


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()