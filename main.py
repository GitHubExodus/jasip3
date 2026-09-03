import sys
import traceback

from cloud_access import (
    download_raw_stock,
    save_dataframe,
    log,
)

from indicators import (
    build_all_timeframes,
)

from backtest import (
    run_all_timeframes,
)

from save_results import (
    save_all_results,
)


SYMBOL = "AAPL"


def run_stock(symbol):

    symbol = symbol.upper()

    log("")
    log("=" * 70)
    log(
        f"EQUITY CURVE TEST: {symbol}"
    )
    log("=" * 70)

    raw = download_raw_stock(
        symbol
    )

    log(
        f"Loaded {len(raw):,} "
        f"1-minute rows"
    )

    timeframe_data = (
        build_all_timeframes(
            raw
        )
    )

    for timeframe, df in (
        timeframe_data.items()
    ):

        key = (
            f"equity_test/"
            f"{symbol}/"
            f"indicators/"
            f"{timeframe}.parquet"
        )

        save_dataframe(
            df,
            key,
        )

    results = run_all_timeframes(
        timeframe_data
    )

    save_all_results(
        symbol,
        results,
    )

    log("")
    log("=" * 70)
    log(
        f"FINISHED: {symbol}"
    )
    log("=" * 70)


def main():

    symbol = (
        sys.argv[1]
        if len(sys.argv) > 1
        else SYMBOL
    )

    try:

        run_stock(
            symbol
        )

    except Exception:

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()