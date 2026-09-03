import io
import heapq
import numpy as np
import pandas as pd
import boto3


# ============================================================
# R2 CONFIGURATION
# ============================================================

R2_ENDPOINT_URL = (
    "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
)

R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"

R2_SECRET_ACCESS_KEY = (
    "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
)

R2_BUCKET = "stocks-data"


# ============================================================
# PATHS
# ============================================================

EQUITY_TEST_PATH = "equity_test"
GLOBAL_EQUITY_PATH = "global_equity/global_equity.parquet"


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
    print(message, flush=True)


# ============================================================
# GET STOCK SYMBOLS
# ============================================================

def get_stock_symbols():
    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key="misc/symbols.txt",
    )

    text = response["Body"].read().decode("utf-8")

    symbols = [
        line.strip().upper()
        for line in text.splitlines()
        if line.strip()
    ]

    return symbols


# ============================================================
# LIST EQUITY FILES
# ============================================================

def get_equity_files():
    """
    Find every strategy equity parquet file under:

        equity_test/<SYMBOL>/<TIMEFRAME>/

    Returns a list of R2 keys.
    """

    files = []

    paginator = s3.get_paginator("list_objects_v2")

    pages = paginator.paginate(
        Bucket=R2_BUCKET,
        Prefix=f"{EQUITY_TEST_PATH}/",
    )

    for page in pages:
        contents = page.get("Contents", [])

        for obj in contents:
            key = obj["Key"]

            if not key.endswith("_equity.parquet"):
                continue

            files.append(key)

    files.sort()

    return files


# ============================================================
# DOWNLOAD PARQUET
# ============================================================

def download_parquet(key):
    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key=key,
    )

    data = response["Body"].read()

    return pd.read_parquet(
        io.BytesIO(data)
    )


# ============================================================
# LOAD ALL RAW EQUITY CURVES
# ============================================================

def load_raw_curves():
    """
    Loads every equity column as an individual raw equity curve.

    Each curve is stored as:

        {
            "name": curve name,
            "timestamps": int64 numpy array,
            "values": float64 numpy array,
        }

    A single strategy file contains 25 equity columns.
    """

    equity_files = get_equity_files()

    log("")
    log(f"Equity files found: {len(equity_files)}")

    curves = []

    for file_number, key in enumerate(equity_files, start=1):

        log(
            f"Loading file "
            f"{file_number}/{len(equity_files)}: "
            f"{key}"
        )

        df = download_parquet(key)

        if "timestamp" not in df.columns:
            log(f"WARNING: timestamp missing: {key}")
            continue

        df = df.sort_values("timestamp")

        timestamps = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

        timestamp_values = (
            timestamps.astype("int64").to_numpy()
        )

        equity_columns = [
            column
            for column in df.columns
            if column.startswith("equity_sl_")
        ]

        for column in equity_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            valid = (
                ~np.isnan(timestamp_values)
                &
                ~np.isnan(values)
            )

            if not np.any(valid):
                continue

            curve_timestamps = timestamp_values[valid]
            curve_values = values[valid]

            curves.append(
                (
                    curve_timestamps,
                    curve_values,
                )
            )

    log("")
    log(f"Total raw equity curves: {len(curves)}")

    return curves


# ============================================================
# BUILD GLOBAL TIMELINE
# ============================================================

def build_global_timeline(curves):
    """
    Build one sorted array containing every timestamp
    at which at least one raw equity curve has an observation.
    """

    log("")
    log("Building global timeline...")

    timestamp_chunks = []

    for timestamps, values in curves:
        timestamp_chunks.append(timestamps)

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

    log(
        f"Global timestamps: "
        f"{len(all_timestamps):,}"
    )

    return all_timestamps


# ============================================================
# INITIALIZE CURVE STATE
# ============================================================

def initialize_curve_state(curves):
    """
    State maintained for every raw equity curve.
    """

    curve_count = len(curves)

    current_values = np.full(
        curve_count,
        np.nan,
        dtype=np.float64,
    )

    previous_values = np.full(
        curve_count,
        np.nan,
        dtype=np.float64,
    )

    positions = np.zeros(
        curve_count,
        dtype=np.int64,
    )

    return (
        current_values,
        previous_values,
        positions,
    )


# ============================================================
# BUILD INITIAL HEAP
# ============================================================

def build_initial_heap(curves):
    """
    Heap contains:

        (next timestamp, curve number)

    for every curve that has at least one observation.
    """

    heap = []

    for curve_number, (timestamps, values) in enumerate(curves):

        if len(timestamps) == 0:
            continue

        first_timestamp = timestamps[0]

        heapq.heappush(
            heap,
            (
                int(first_timestamp),
                curve_number,
            ),
        )

    return heap


# ============================================================
# PROCESS GLOBAL EQUITY
# ============================================================

def build_global_equity(
    curves,
    global_timestamps,
):
    """
    Build the global equity curve.

    IMPORTANT LOGIC
    ---------------

    At each global timestamp:

    1. Find every raw curve that has a NEW observation
       at this timestamp.

    2. Update those curves:

           previous = old current
           current  = new observation

    3. Curves that did not receive a new observation keep
       their latest completed equity value.

    4. Calculate contribution using CURRENT equity:

           profit = current equity - 100

       Only positive profits contribute.

    5. Calculate the return for ONLY the curves that
       received a new observation:

           return = current / previous - 1

    6. Weight each NEW curve's return by its current
       contribution.

    7. Apply the combined return to the global equity ONCE.

    This means a daily curve sitting at its Day 2 value
    during Day 3 contributes its Day 2 equity to the
    weighting calculation, but its Day 2 movement is NOT
    applied again.
    """

    curve_count = len(curves)

    (
        current_values,
        previous_values,
        positions,
    ) = initialize_curve_state(curves)

    heap = build_initial_heap(curves)

    log("")
    log("Building global equity...")
    log(f"Curves: {curve_count:,}")
    log(
        f"Global timestamps: "
        f"{len(global_timestamps):,}"
    )

    # --------------------------------------------------------
    # GLOBAL EQUITY
    # --------------------------------------------------------

    global_equity = STARTING_EQUITY

    output_timestamps = []
    output_equity = []

    # --------------------------------------------------------
    # PROCESS EACH GLOBAL TIMESTAMP
    # --------------------------------------------------------

    for timestamp_number, global_timestamp in enumerate(
        global_timestamps
    ):

        current_timestamp = int(
            global_timestamp
        )

        # ----------------------------------------------------
        # FIND ALL CURVES WITH A NEW BAR AT THIS TIMESTAMP
        # ----------------------------------------------------

        changed_curves = []

        while heap and heap[0][0] <= current_timestamp:

            next_timestamp, curve_number = heapq.heappop(
                heap
            )

            timestamps, values = curves[curve_number]

            position = positions[curve_number]

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if position >= len(timestamps):
                continue

            # ------------------------------------------------
            # There may be multiple observations at the same
            # timestamp. Process them until we reach a future
            # timestamp.
            # ------------------------------------------------

            while (
                position < len(timestamps)
                and timestamps[position] <= current_timestamp
            ):

                new_value = values[position]

                old_value = current_values[
                    curve_number
                ]

                previous_values[
                    curve_number
                ] = old_value

                current_values[
                    curve_number
                ] = new_value

                position += 1

            positions[curve_number] = position

            changed_curves.append(
                curve_number
            )

            # ------------------------------------------------
            # Schedule next observation
            # ------------------------------------------------

            if position < len(timestamps):

                next_curve_timestamp = int(
                    timestamps[position]
                )

                heapq.heappush(
                    heap,
                    (
                        next_curve_timestamp,
                        curve_number,
                    ),
                )

        # ----------------------------------------------------
        # No curve changed at this timestamp.
        #
        # There is no new equity movement to apply.
        # ----------------------------------------------------

        if not changed_curves:

            output_timestamps.append(
                current_timestamp
            )

            output_equity.append(
                global_equity
            )

            continue

        # ----------------------------------------------------
        # STEP 1
        #
        # Calculate TOTAL positive profit using the CURRENT
        # equity of every curve.
        #
        # This is NOT based on current - previous.
        # ----------------------------------------------------

        valid_current = (
            ~np.isnan(current_values)
        )

        positive_profit_mask = (
            valid_current
            &
            (
                current_values
                >
                STARTING_EQUITY
            )
        )

        total_positive_profit = np.sum(
            current_values[
                positive_profit_mask
            ]
            -
            STARTING_EQUITY
        )

        # ----------------------------------------------------
        # STEP 2
        #
        # If nobody is profitable, there is no contribution.
        # Therefore global equity does not move.
        # ----------------------------------------------------

        if (
            not np.isfinite(total_positive_profit)
            or total_positive_profit <= 0.0
        ):

            output_timestamps.append(
                current_timestamp
            )

            output_equity.append(
                global_equity
            )

            continue

        # ----------------------------------------------------
        # STEP 3
        #
        # Only changed curves can have a non-zero return
        # during this event.
        #
        # Calculate their contribution using their CURRENT
        # equity value.
        # ----------------------------------------------------

        total_global_return = 0.0

        for curve_number in changed_curves:

            current_value = current_values[
                curve_number
            ]

            previous_value = previous_values[
                curve_number
            ]

            # ----------------------------------------------
            # Need both values to calculate movement.
            #
            # A curve's first observation establishes its
            # current equity but produces no return yet.
            # ----------------------------------------------

            if not np.isfinite(
                current_value
            ):
                continue

            if not np.isfinite(
                previous_value
            ):
                continue

            if previous_value <= 0.0:
                continue

            # ----------------------------------------------
            # CURRENT-EQUITY CONTRIBUTION
            #
            # Example:
            #
            # current = 140
            #
            # contribution profit = 40
            # ----------------------------------------------

            current_profit = (
                current_value
                -
                STARTING_EQUITY
            )

            if current_profit <= 0.0:
                continue

            contribution = (
                current_profit
                /
                total_positive_profit
            )

            # ----------------------------------------------
            # RAW CURVE MOVEMENT
            #
            # Example:
            #
            # previous = 130
            # current  = 140
            #
            # return = +7.6923%
            # ----------------------------------------------

            raw_return = (
                current_value
                /
                previous_value
                -
                1.0
            )

            if not np.isfinite(
                raw_return
            ):
                continue

            # ----------------------------------------------
            # CONTRIBUTION × MOVEMENT
            # ----------------------------------------------

            weighted_return = (
                contribution
                *
                raw_return
            )

            if not np.isfinite(
                weighted_return
            ):
                continue

            total_global_return += (
                weighted_return
            )

        # ----------------------------------------------------
        # STEP 4
        #
        # Apply the combined weighted return ONCE.
        # ----------------------------------------------------

        if np.isfinite(
            total_global_return
        ):

            global_equity *= (
                1.0
                +
                total_global_return
            )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if not np.isfinite(
            global_equity
        ):

            raise RuntimeError(
                "\n"
                "GLOBAL EQUITY BECAME NON-FINITE\n"
                f"timestamp = {current_timestamp}\n"
                f"global_return = {total_global_return}\n"
                f"global_equity = {global_equity}\n"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_timestamps.append(
            current_timestamp
        )

        output_equity.append(
            global_equity
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            timestamp_number % 10_000 == 0
            or
            timestamp_number
            ==
            len(global_timestamps) - 1
        ):

            percent = (
                (
                    timestamp_number + 1
                )
                /
                len(global_timestamps)
                *
                100.0
            )

            log(
                f"Progress: "
                f"{percent:6.2f}% | "
                f"Time: "
                f"{pd.to_datetime(current_timestamp, unit='ns', utc=True)} | "
                f"Global equity: "
                f"{global_equity:.3f}"
            )

    # ========================================================
    # BUILD OUTPUT DATAFRAME
    # ========================================================

    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                np.asarray(
                    output_timestamps,
                    dtype=np.int64,
                ),
                unit="ns",
                utc=True,
            ),
            "equity": np.asarray(
                output_equity,
                dtype=np.float64,
            ),
        }
    )

    return result


# ============================================================
# SAVE GLOBAL EQUITY
# ============================================================

def save_global_equity(df):
    log("")
    log("Saving global equity curve...")
    log(
        f"  {GLOBAL_EQUITY_PATH}"
    )

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
        Key=GLOBAL_EQUITY_PATH,
        Body=buffer.getvalue(),
    )

    log("Global equity saved.")


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("GLOBAL EQUITY CURVE")
    log("=" * 70)

    # --------------------------------------------------------
    # LOAD CURVES
    # --------------------------------------------------------

    curves = load_raw_curves()

    if not curves:

        raise RuntimeError(
            "No raw equity curves found."
        )

    # --------------------------------------------------------
    # GLOBAL TIMELINE
    # --------------------------------------------------------

    global_timestamps = build_global_timeline(
        curves
    )

    if len(global_timestamps) == 0:

        raise RuntimeError(
            "No timestamps found."
        )

    # --------------------------------------------------------
    # BUILD GLOBAL EQUITY
    # --------------------------------------------------------

    global_equity_df = build_global_equity(
        curves,
        global_timestamps,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_global_equity(
        global_equity_df
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    final_equity = global_equity_df[
        "equity"
    ].iloc[-1]

    max_equity = global_equity_df[
        "equity"
    ].max()

    min_equity = global_equity_df[
        "equity"
    ].min()

    total_return = (
        final_equity
        /
        STARTING_EQUITY
        -
        1.0
    ) * 100.0

    log("")
    log("=" * 70)
    log("COMPLETE")
    log("=" * 70)

    log(
        f"Starting equity: "
        f"{STARTING_EQUITY:.3f}"
    )

    log(
        f"Final equity: "
        f"{final_equity:.3f}"
    )

    log(
        f"Minimum equity: "
        f"{min_equity:.3f}"
    )

    log(
        f"Maximum equity: "
        f"{max_equity:.3f}"
    )

    log(
        f"Total return: "
        f"{total_return:.2f}%"
    )

    log(
        f"Output: "
        f"{GLOBAL_EQUITY_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
