import io
import os
import sys
import traceback
import heapq

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
# PATHS
# ============================================================

EQUITY_ROOT = "equity_test"

OUTPUT_KEY = (
    "global_equity/"
    "global_equity.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

STARTING_EQUITY = 100.0


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

    print(
        message,
        flush=True,
    )


# ============================================================
# GET ALL EQUITY FILES
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
# DOWNLOAD EQUITY FILE
# ============================================================

def download_equity_file(
    key,
):

    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key=key,
    )

    data = response["Body"].read()

    return pd.read_parquet(
        io.BytesIO(data)
    )


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
# LOAD ALL CURVES
# ============================================================
#
# We do NOT create a giant matrix.
#
# Instead, each raw curve is converted into a small record:
#
# {
#     timestamps,
#     values
# }
#
# The global calculation later walks through timestamps
# and advances each curve only when that curve has a new
# observation.
#
# ============================================================

def load_all_curves(
    equity_files,
):

    curves = []

    total_files = len(
        equity_files
    )

    for file_number, key in enumerate(
        equity_files,
        start=1,
    ):

        if (
            file_number % 100 == 0
            or
            file_number == total_files
        ):

            log(
                f"Loading files "
                f"{file_number:,}/"
                f"{total_files:,}"
            )

        df = download_equity_file(
            key
        )

        if df.empty:
            continue

        if "timestamp" not in df.columns:
            continue

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

            curve_df = df[
                [
                    "timestamp",
                    column,
                ]
            ].dropna(
                subset=[
                    column
                ]
            )

            if curve_df.empty:
                continue

            timestamps = (
                curve_df[
                    "timestamp"
                ]
                .astype("int64")
                .to_numpy()
            )

            values = np.asarray(
                curve_df[column],
                dtype=np.float64,
            )

            curves.append(
                (
                    timestamps,
                    values,
                )
            )

    return curves


# ============================================================
# BUILD GLOBAL TIMELINE
# ============================================================
#
# We still need the timestamps, but only as a 1-dimensional
# array. This is tiny compared with the old 530 GiB matrix.
#
# ============================================================

def build_global_timeline(
    curves,
):

    log("")
    log(
        "Building global timeline..."
    )

    timestamp_chunks = []

    for timestamps, values in curves:

        timestamp_chunks.append(
            timestamps
        )

    if not timestamp_chunks:

        return np.empty(
            0,
            dtype=np.int64,
        )

    all_timestamps = np.concatenate(
        timestamp_chunks
    )

    all_timestamps = np.unique(
        all_timestamps
    )

    all_timestamps.sort()

    return all_timestamps


# ============================================================
# CALCULATE GLOBAL EQUITY
# ============================================================
#
# IMPORTANT:
#
# We do NOT create:
#
#     96,650 x 735,826
#
# Instead we maintain:
#
#     current equity for each raw curve
#     previous equity for each raw curve
#
# That is only:
#
#     96,650 values
#
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

    if curve_count == 0:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "equity",
            ]
        )

    # --------------------------------------------------------
    # CURRENT VALUE OF EVERY CURVE
    # --------------------------------------------------------

    current_values = np.full(
        curve_count,
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # PREVIOUS VALUE OF EVERY CURVE
    # --------------------------------------------------------

    previous_values = np.full(
        curve_count,
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # CURRENT POSITION INSIDE EACH CURVE
    # --------------------------------------------------------

    positions = np.zeros(
        curve_count,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # HEAP
    #
    # Each entry:
    #
    # (next_timestamp, curve_number)
    #
    # This lets us efficiently find which raw curves have
    # a new value at the current global timestamp.
    # --------------------------------------------------------

    heap = []

    for curve_number, (
        timestamps,
        values,
    ) in enumerate(curves):

        if len(timestamps) == 0:
            continue

        heapq.heappush(
            heap,
            (
                timestamps[0],
                curve_number,
            )
        )

    # --------------------------------------------------------
    # RESULT ARRAYS
    # --------------------------------------------------------

    global_equity_values = np.empty(
        timestamp_count,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # MAIN EQUITY
    # --------------------------------------------------------

    main_equity = (
        STARTING_EQUITY
    )

    # --------------------------------------------------------
    # FIRST TIMESTAMP
    # --------------------------------------------------------
    #
    # There is no previous bar yet, so we initialize every
    # curve that has its first observation at the first
    # global timestamp.
    #
    # Main equity remains 100 here.
    #
    # --------------------------------------------------------

    first_timestamp = timeline[0]

    while (
        heap
        and
        heap[0][0]
        <= first_timestamp
    ):

        timestamp, curve_number = (
            heapq.heappop(heap)
        )

        timestamps, values = (
            curves[curve_number]
        )

        position = (
            positions[curve_number]
        )

        current_values[
            curve_number
        ] = values[position]

        positions[
            curve_number
        ] += 1

        if (
            positions[curve_number]
            <
            len(timestamps)
        ):

            heapq.heappush(
                heap,
                (
                    timestamps[
                        positions[
                            curve_number
                        ]
                    ],
                    curve_number,
                )
            )

    global_equity_values[0] = (
        main_equity
    )

    # --------------------------------------------------------
    # PROCESS REMAINING TIMESTAMPS
    # --------------------------------------------------------

    log("")
    log(
        "Calculating global equity..."
    )

    for t in range(
        1,
        timestamp_count,
    ):

        current_timestamp = (
            timeline[t]
        )

        # ----------------------------------------------------
        # MOVE EVERY CURVE THAT HAS A NEW VALUE
        # ----------------------------------------------------

        while (
            heap
            and
            heap[0][0]
            <= current_timestamp
        ):

            timestamp, curve_number = (
                heapq.heappop(heap)
            )

            timestamps, values = (
                curves[curve_number]
            )

            position = (
                positions[
                    curve_number
                ]
            )

            # ------------------------------------------------
            # Move current value to previous value.
            # ------------------------------------------------

            current_value = (
                values[position]
            )

            previous_values[
                curve_number
            ] = current_values[
                curve_number
            ]

            current_values[
                curve_number
            ] = current_value

            positions[
                curve_number
            ] += 1

            # ------------------------------------------------
            # Add next observation for this curve to heap.
            # ------------------------------------------------

            if (
                positions[
                    curve_number
                ]
                <
                len(timestamps)
            ):

                next_position = (
                    positions[
                        curve_number
                    ]
                )

                heapq.heappush(
                    heap,
                    (
                        timestamps[
                            next_position
                        ],
                        curve_number,
                    )
                )

        # ----------------------------------------------------
        # DETERMINE WHICH CURVES CAN CONTRIBUTE
        # ----------------------------------------------------
        #
        # A curve must:
        #
        # 1. Have a current value.
        # 2. Have a previous value.
        # 3. Be profitable now.
        #
        # Profitable means:
        #
        # current equity > 100
        #
        # ----------------------------------------------------

        valid = (
            ~np.isnan(
                current_values
            )
            &
            ~np.isnan(
                previous_values
            )
        )

        profitable = (
            current_values
            >
            STARTING_EQUITY
        )

        contributors = (
            valid
            &
            profitable
        )

        # ----------------------------------------------------
        # IF NO CURVE IS PROFITABLE
        # ----------------------------------------------------

        if not np.any(
            contributors
        ):

            global_equity_values[t] = (
                main_equity
            )

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

            continue

        # ----------------------------------------------------
        # POSITIVE PROFITS
        # ----------------------------------------------------

        positive_profits = (
            current_values[
                contributors
            ]
            -
            STARTING_EQUITY
        )

        total_positive_profit = (
            np.sum(
                positive_profits
            )
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if (
            total_positive_profit
            <= 0.0
        ):

            global_equity_values[t] = (
                main_equity
            )

            continue

        # ----------------------------------------------------
        # CONTRIBUTION WEIGHTS
        # ----------------------------------------------------
        #
        # Example:
        #
        # Curve A = 120
        # Curve B = 140
        #
        # Profits:
        #
        # A = 20
        # B = 40
        #
        # Total = 60
        #
        # A contribution = 33.33%
        # B contribution = 66.67%
        #
        # ----------------------------------------------------

        weights = (
            positive_profits
            /
            total_positive_profit
        )

        # ----------------------------------------------------
        # GET PREVIOUS AND CURRENT VALUES
        # ----------------------------------------------------

        current = (
            current_values[
                contributors
            ]
        )

        previous = (
            previous_values[
                contributors
            ]
        )

        # ----------------------------------------------------
        # RAW CURVE GROWTH
        # ----------------------------------------------------
        #
        # Example:
        #
        # Previous = 1
        # Current  = 2
        #
        # Growth = 2.0
        # Return = +100%
        #
        # ----------------------------------------------------

        valid_previous = (
            previous
            >
            0.0
        )

        if not np.any(
            valid_previous
        ):

            global_equity_values[t] = (
                main_equity
            )

            continue

        current = current[
            valid_previous
        ]

        previous = previous[
            valid_previous
        ]

        weights = weights[
            valid_previous
        ]

        growth = (
            current
            /
            previous
        )

        raw_returns = (
            growth
            -
            1.0
        )

        # ----------------------------------------------------
        # WEIGHTED RETURN
        # ----------------------------------------------------
        #
        # Contribution weight determines how much of the
        # raw curve's movement is used by the global strategy.
        #
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

        # Keep precision high enough to avoid unnecessary
        # rounding accumulation over hundreds of thousands
        # of timestamps.
        main_equity = round(
            main_equity,
            6,
        )

        global_equity_values[t] = (
            main_equity
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
            "timestamp": pd.to_datetime(
                timeline,
                utc=True,
            ),
            "equity": (
                global_equity_values
            ),
        }
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    log("")
    log("=" * 70)
    log("GLOBAL EQUITY CURVE")
    log("=" * 70)

    # --------------------------------------------------------
    # FIND FILES
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

    log("")
    log(
        "Loading raw equity curves..."
    )

    curves = (
        load_all_curves(
            equity_files
        )
    )

    log("")
    log(
        f"Total raw equity curves: "
        f"{len(curves):,}"
    )

    # --------------------------------------------------------
    # BUILD TIMELINE
    # --------------------------------------------------------

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
    # CALCULATE
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

    save_dataframe(
        result,
        OUTPUT_KEY,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    starting = (
        result[
            "equity"
        ].iloc[0]
    )

    ending = (
        result[
            "equity"
        ].iloc[-1]
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
        f"Raw curves: "
        f"{len(curves):,}"
    )

    log(
        f"Timestamps: "
        f"{len(result):,}"
    )

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
