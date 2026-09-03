import io
import os
import sys
import traceback

import boto3
import numpy as np
import pandas as pd


# ============================================================
# R2 CONFIGURATION
# ============================================================

R2_ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"

R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"

R2_SECRET_ACCESS_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"

R2_BUCKET = "stocks-data"


# ============================================================
# GLOBAL EQUITY CONFIGURATION
# ============================================================

STARTING_EQUITY = 100.0

EQUITY_ROOT = "equity_test"

OUTPUT_ROOT = "global_equity"


# ============================================================
# R2 CLIENT
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
)


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)


# ============================================================
# GET STOCK SYMBOLS
# ============================================================

def get_stock_symbols():

    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key="misc/symbols.txt",
    )

    text = response["Body"].read().decode(
        "utf-8"
    )

    symbols = [
        line.strip().upper()
        for line in text.splitlines()
        if line.strip()
    ]

    return symbols


# ============================================================
# FIND ALL EQUITY FILES
# ============================================================

def get_equity_files():

    files = []

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    pages = paginator.paginate(
        Bucket=R2_BUCKET,
        Prefix=f"{EQUITY_ROOT}/",
    )

    for page in pages:

        for obj in page.get(
            "Contents",
            [],
        ):

            key = obj["Key"]

            if not key.endswith(
                "_equity.parquet"
            ):
                continue

            files.append(key)

    files.sort()

    return files


# ============================================================
# DOWNLOAD ONE EQUITY FILE
# ============================================================

def download_equity_file(
    key,
):

    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key=key,
    )

    data = response["Body"].read()

    df = pd.read_parquet(
        io.BytesIO(data)
    )

    return df


# ============================================================
# SAVE DATAFRAME
# ============================================================

def save_dataframe(
    df,
    key,
):

    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
    )

    log(
        f"Saved: {key}"
    )


# ============================================================
# EXTRACT METADATA FROM EQUITY FILE
# ============================================================

def parse_equity_file_key(
    key,
):

    # Expected:
    #
    # equity_test/AAPL/5m/strategy_1_equity.parquet

    parts = key.split("/")

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected equity key: {key}"
        )

    symbol = parts[1]

    timeframe = parts[2]

    strategy_file = parts[3]

    strategy = (
        strategy_file
        .replace(
            "_equity.parquet",
            "",
        )
    )

    return (
        symbol,
        timeframe,
        strategy,
    )


# ============================================================
# LOAD ALL RAW EQUITY CURVES
# ============================================================

def load_all_equity_curves(
    equity_files,
):

    curves = []

    for number, key in enumerate(
        equity_files,
        start=1,
    ):

        log(
            f"Loading equity file "
            f"{number:,}/"
            f"{len(equity_files):,}: "
            f"{key}"
        )

        df = download_equity_file(
            key
        )

        if df.empty:
            continue

        if "timestamp" not in df.columns:
            raise ValueError(
                f"No timestamp column: {key}"
            )

        (
            symbol,
            timeframe,
            strategy,
        ) = parse_equity_file_key(
            key
        )

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                utc=True,
            )
        )

        df = (
            df
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        equity_columns = [
            column
            for column in df.columns
            if column.startswith(
                "equity_sl_"
            )
        ]

        for column in equity_columns:

            curves.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "curve": column,
                    "data": df[
                        [
                            "timestamp",
                            column,
                        ]
                    ].copy(),
                }
            )

    return curves


# ============================================================
# BUILD GLOBAL TIMELINE
# ============================================================

def build_global_timeline(
    curves,
):

    timestamps = []

    for curve in curves:

        data = curve["data"]

        timestamps.append(
            data["timestamp"]
        )

    if not timestamps:

        return pd.DatetimeIndex(
            [],
            tz="UTC",
        )

    combined = pd.concat(
        timestamps,
        ignore_index=True,
    )

    combined = (
        pd.DatetimeIndex(
            combined
            .drop_duplicates()
            .sort_values()
        )
    )

    return combined


# ============================================================
# PREPARE CURVE DATA
# ============================================================

def prepare_curve(
    curve,
):

    data = curve["data"].copy()

    column = curve["curve"]

    data = data[
        [
            "timestamp",
            column,
        ]
    ]

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            column
        ]
    )

    data = (
        data
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return data


# ============================================================
# CREATE AS-OF VALUE ARRAYS
# ============================================================

def align_curve_to_global_time(
    curve_data,
    timeline,
):

    values = np.asarray(
        curve_data.iloc[:, 1],
        dtype=np.float64,
    )

    curve_times = (
        curve_data["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    global_times = (
        timeline
        .astype("int64")
        .to_numpy()
    )

    # For every global timestamp, find
    # the latest raw curve timestamp
    # that is <= global timestamp.
    positions = np.searchsorted(
        curve_times,
        global_times,
        side="right",
    ) - 1

    aligned = np.full(
        len(global_times),
        np.nan,
        dtype=np.float64,
    )

    valid = positions >= 0

    aligned[valid] = (
        values[
            positions[valid]
        ]
    )

    return aligned


# ============================================================
# CALCULATE GLOBAL EQUITY
# ============================================================

def calculate_global_equity(
    curves,
    timeline,
):

    curve_count = len(
        curves
    )

    timestamp_count = len(
        timeline
    )

    log("")
    log(
        f"Raw equity curves: "
        f"{curve_count:,}"
    )

    log(
        f"Global timestamps: "
        f"{timestamp_count:,}"
    )

    # --------------------------------------------------------
    # ALIGN EVERY RAW CURVE TO THE GLOBAL TIMELINE
    # --------------------------------------------------------

    log("")
    log(
        "Aligning raw equity curves..."
    )

    aligned_equities = np.full(
        (
            curve_count,
            timestamp_count,
        ),
        np.nan,
        dtype=np.float64,
    )

    for i, curve in enumerate(
        curves
    ):

        data = prepare_curve(
            curve
        )

        aligned_equities[i] = (
            align_curve_to_global_time(
                data,
                timeline,
            )
        )

        if (
            (i + 1) % 1000 == 0
            or
            i + 1 == curve_count
        ):

            log(
                f"  Aligned "
                f"{i + 1:,}/"
                f"{curve_count:,}"
            )

    # --------------------------------------------------------
    # GLOBAL EQUITY
    # --------------------------------------------------------

    global_equity = np.empty(
        timestamp_count,
        dtype=np.float64,
    )

    global_equity[0] = (
        STARTING_EQUITY
    )

    # --------------------------------------------------------
    # PROCESS EACH TIMESTAMP
    # --------------------------------------------------------

    log("")
    log(
        "Calculating global equity..."
    )

    previous_values = (
        aligned_equities[:, 0]
    )

    main_equity = (
        STARTING_EQUITY
    )

    for t in range(
        1,
        timestamp_count,
    ):

        current_values = (
            aligned_equities[:, t]
        )

        # ----------------------------------------------------
        # FIND VALID CURVES
        # ----------------------------------------------------

        valid_current = (
            ~np.isnan(
                current_values
            )
        )

        valid_previous = (
            ~np.isnan(
                previous_values
            )
        )

        valid = (
            valid_current
            &
            valid_previous
        )

        if not np.any(valid):

            global_equity[t] = (
                main_equity
            )

            previous_values = (
                current_values
            )

            continue

        # ----------------------------------------------------
        # CURRENT PROFITS
        #
        # Every raw equity starts at 100.
        #
        # A raw equity of:
        #
        # 120 -> +20 profit
        # 100 ->  0 profit
        #  80  -> -20 profit
        #
        # Only positive profits participate.
        # ----------------------------------------------------

        profits = (
            current_values[valid]
            - STARTING_EQUITY
        )

        positive_profits = np.maximum(
            profits,
            0.0,
        )

        total_positive_profit = (
            np.sum(
                positive_profits
            )
        )

        # ----------------------------------------------------
        # NO PROFITABLE CURVES
        # ----------------------------------------------------

        if (
            total_positive_profit
            <= 0.0
        ):

            global_equity[t] = (
                main_equity
            )

            previous_values = (
                current_values
            )

            continue

        # ----------------------------------------------------
        # CONTRIBUTION WEIGHTS
        #
        # Example:
        #
        # A profit = 20
        # B profit = 40
        #
        # total = 60
        #
        # A = 20/60
        # B = 40/60
        # ----------------------------------------------------

        weights = (
            positive_profits
            /
            total_positive_profit
        )

        # ----------------------------------------------------
        # RAW EQUITY GROWTH
        #
        # If:
        #
        # previous = 1
        # current  = 2
        #
        # growth = 2x
        #
        # growth return = +100%
        # ----------------------------------------------------

        current_valid = (
            current_values[valid]
        )

        previous_valid = (
            previous_values[valid]
        )

        growth = (
            current_valid
            /
            previous_valid
        )

        raw_returns = (
            growth - 1.0
        )

        # ----------------------------------------------------
        # WEIGHT EACH RAW RETURN
        #
        # If:
        #
        # raw return = +100%
        # contribution = 10%
        #
        # contribution to main equity =
        #
        # 10% × 100%
        # = +10%
        # ----------------------------------------------------

        weighted_returns = (
            weights
            *
            raw_returns
        )

        total_return = (
            np.sum(
                weighted_returns
            )
        )

        # ----------------------------------------------------
        # UPDATE MAIN EQUITY
        # ----------------------------------------------------

        main_equity *= (
            1.0
            +
            total_return
        )

        main_equity = round(
            main_equity,
            6,
        )

        global_equity[t] = (
            main_equity
        )

        previous_values = (
            current_values
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (t + 1) % 1000 == 0
            or
            t + 1 == timestamp_count
        ):

            log(
                f"  Timestamp "
                f"{t + 1:,}/"
                f"{timestamp_count:,} "
                f"| Equity: "
                f"{main_equity:.6f}"
            )

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "timestamp": timeline,
            "equity": global_equity,
        }
    )

    return result


# ============================================================
# SAVE GLOBAL EQUITY
# ============================================================

def save_global_equity(
    result,
):

    key = (
        f"{OUTPUT_ROOT}/"
        f"global_equity.parquet"
    )

    save_dataframe(
        result,
        key,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("")
    log("=" * 70)
    log("GLOBAL EQUITY CURVE")
    log("=" * 70)

    # --------------------------------------------------------
    # FIND EQUITY FILES
    # --------------------------------------------------------

    log("")
    log(
        "Finding equity files in R2..."
    )

    equity_files = (
        get_equity_files()
    )

    log(
        f"Found "
        f"{len(equity_files):,} "
        f"equity files"
    )

    if not equity_files:

        raise RuntimeError(
            "No equity files found."
        )

    # --------------------------------------------------------
    # LOAD CURVES
    # --------------------------------------------------------

    curves = (
        load_all_equity_curves(
            equity_files
        )
    )

    log("")
    log(
        f"Total raw equity curves: "
        f"{len(curves):,}"
    )

    # --------------------------------------------------------
    # BUILD GLOBAL TIMELINE
    # --------------------------------------------------------

    log("")
    log(
        "Building global timeline..."
    )

    timeline = (
        build_global_timeline(
            curves
        )
    )

    log(
        f"Global timestamps: "
        f"{len(timeline):,}"
    )

    if len(timeline) == 0:

        raise RuntimeError(
            "No timestamps found."
        )

    # --------------------------------------------------------
    # CALCULATE GLOBAL EQUITY
    # --------------------------------------------------------

    result = (
        calculate_global_equity(
            curves,
            timeline,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_global_equity(
        result
    )

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    starting = (
        result["equity"].iloc[0]
    )

    ending = (
        result["equity"].iloc[-1]
    )

    total_return = (
        ending
        /
        starting
        -
        1.0
    ) * 100.0

    log("")
    log("=" * 70)
    log("GLOBAL EQUITY COMPLETE")
    log("=" * 70)

    log(
        f"Starting equity: "
        f"{starting:.6f}"
    )

    log(
        f"Ending equity:   "
        f"{ending:.6f}"
    )

    log(
        f"Total return:    "
        f"{total_return:.6f}%"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        traceback.print_exc()

        sys.exit(1)