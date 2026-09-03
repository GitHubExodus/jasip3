import io
import boto3
import numpy as np
import pandas as pd


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

OUTPUT_PATH = (
    "global_equity/"
    "raw_equity_rankings.parquet"
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
    print(message, flush=True)


# ============================================================
# GET EQUITY FILES
# ============================================================

def get_equity_files():
    """
    Find every raw strategy equity file.

    Example:

        equity_test/AAPL/5m/strategy_1_equity.parquet
    """

    files = []

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    pages = paginator.paginate(
        Bucket=R2_BUCKET,
        Prefix=f"{EQUITY_TEST_PATH}/",
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
# DOWNLOAD PARQUET
# ============================================================

def download_parquet(key):

    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key=key,
    )

    data = response[
        "Body"
    ].read()

    return pd.read_parquet(
        io.BytesIO(data)
    )


# ============================================================
# PARSE FILE INFORMATION
# ============================================================

def parse_file_information(key):
    """
    Extract:

        symbol
        timeframe
        strategy

    from:

        equity_test/AAPL/5m/strategy_1_equity.parquet
    """

    parts = key.split("/")

    symbol = parts[1]
    timeframe = parts[2]

    filename = parts[3]

    strategy_name = filename.replace(
        "_equity.parquet",
        "",
    )

    return (
        symbol,
        timeframe,
        strategy_name,
    )


# ============================================================
# PARSE SL / RR FROM COLUMN NAME
# ============================================================

def parse_sl_rr(column):

    # Expected:

    # equity_sl_0.5_rr_1
    # equity_sl_1_rr_1.5
    # equity_sl_5_rr_5

    parts = column.split("_")

    stop_loss = float(
        parts[2]
    )

    risk_reward = float(
        parts[4]
    )

    return (
        stop_loss,
        risk_reward,
    )


# ============================================================
# CALCULATE RAW CURVE STATISTICS
# ============================================================

def calculate_curve_statistics(
    values
):
    """
    Calculate statistics for one raw equity curve.

    Returns:

        average_equity
        average_bar_change
        final_equity
        total_return
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # --------------------------------------------------------
    # Average equity
    # --------------------------------------------------------

    average_equity = np.mean(
        values
    )

    # --------------------------------------------------------
    # Bar-to-bar percentage changes
    # --------------------------------------------------------

    if len(values) < 2:

        average_bar_change = np.nan

    else:

        previous = values[:-1]
        current = values[1:]

        valid = (
            np.isfinite(previous)
            &
            np.isfinite(current)
            &
            (previous != 0.0)
        )

        if np.any(valid):

            bar_changes = (
                current[valid]
                /
                previous[valid]
                -
                1.0
            )

            average_bar_change = (
                np.mean(bar_changes)
                *
                100.0
            )

        else:

            average_bar_change = np.nan

    # --------------------------------------------------------
    # Final equity
    # --------------------------------------------------------

    final_equity = values[-1]

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    total_return = (
        final_equity
        /
        STARTING_EQUITY
        -
        1.0
    ) * 100.0

    return (
        average_equity,
        average_bar_change,
        final_equity,
        total_return,
    )


# ============================================================
# BUILD RAW EQUITY RANKING
# ============================================================

def build_rankings():

    equity_files = get_equity_files()

    log("")
    log(
        f"Equity files found: "
        f"{len(equity_files):,}"
    )

    rows = []

    # ========================================================
    # PROCESS EVERY EQUITY FILE
    # ========================================================

    for file_number, key in enumerate(
        equity_files,
        start=1,
    ):

        log(
            f"Processing "
            f"{file_number:,}/"
            f"{len(equity_files):,}: "
            f"{key}"
        )

        df = download_parquet(
            key
        )

        if "timestamp" not in df.columns:

            log(
                f"WARNING: timestamp missing: "
                f"{key}"
            )

            continue

        (
            symbol,
            timeframe,
            strategy_name,
        ) = parse_file_information(
            key
        )

        equity_columns = [
            column
            for column in df.columns
            if column.startswith(
                "equity_sl_"
            )
        ]

        # ====================================================
        # PROCESS EACH RAW CURVE
        # ====================================================

        for column in equity_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            (
                average_equity,
                average_bar_change,
                final_equity,
                total_return,
            ) = calculate_curve_statistics(
                values
            )

            (
                stop_loss,
                risk_reward,
            ) = parse_sl_rr(
                column
            )

            rows.append(
                {
                    "curve": (
                        f"{symbol}_"
                        f"{timeframe}_"
                        f"{strategy_name}_"
                        f"sl_{stop_loss:g}_"
                        f"rr_{risk_reward:g}"
                    ),

                    "symbol": symbol,

                    "timeframe": timeframe,

                    "strategy": strategy_name,

                    "stop_loss_percent": (
                        stop_loss
                    ),

                    "risk_reward": (
                        risk_reward
                    ),

                    "average_equity": (
                        average_equity
                    ),

                    "average_bar_change_percent": (
                        average_bar_change
                    ),

                    "final_equity": (
                        final_equity
                    ),

                    "total_return_percent": (
                        total_return
                    ),
                }
            )

    # ========================================================
    # BUILD DATAFRAME
    # ========================================================

    rankings = pd.DataFrame(
        rows
    )

    if rankings.empty:

        raise RuntimeError(
            "No raw equity curves found."
        )

    log("")
    log(
        f"Total raw equity curves: "
        f"{len(rankings):,}"
    )

    # ========================================================
    # CALCULATE CONTRIBUTION
    # ========================================================

    rankings[
        "profit_from_average_equity"
    ] = (
        rankings[
            "average_equity"
        ]
        -
        STARTING_EQUITY
    )

    # --------------------------------------------------------
    # Only positive average-equity profit contributes.
    # --------------------------------------------------------

    positive_profit = (
        rankings[
            "profit_from_average_equity"
        ]
        .clip(lower=0.0)
    )

    total_positive_profit = (
        positive_profit.sum()
    )

    if (
        not np.isfinite(
            total_positive_profit
        )
        or
        total_positive_profit <= 0.0
    ):

        rankings[
            "contribution_percent"
        ] = 0.0

    else:

        rankings[
            "contribution_percent"
        ] = (
            positive_profit
            /
            total_positive_profit
            *
            100.0
        )

    # ========================================================
    # RANK
    # ========================================================

    rankings[
        "rank"
    ] = (
        rankings[
            "contribution_percent"
        ]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    # ========================================================
    # ORDER BY CONTRIBUTION
    # ========================================================

    rankings = rankings.sort_values(
        "contribution_percent",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # PUT RANK FIRST
    # ========================================================

    rankings = rankings[
        [
            "rank",
            "curve",
            "symbol",
            "timeframe",
            "strategy",
            "stop_loss_percent",
            "risk_reward",
            "average_equity",
            "average_bar_change_percent",
            "final_equity",
            "total_return_percent",
            "profit_from_average_equity",
            "contribution_percent",
        ]
    ]

    return rankings


# ============================================================
# SAVE RESULTS
# ============================================================

def save_rankings(
    rankings
):

    log("")
    log(
        "Saving raw equity rankings..."
    )

    buffer = io.BytesIO()

    rankings.to_parquet(
        buffer,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    buffer.seek(0)

    s3.put_object(
        Bucket=R2_BUCKET,
        Key=OUTPUT_PATH,
        Body=buffer.getvalue(),
    )

    log(
        f"Saved: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("RAW EQUITY CURVE RANKINGS")
    log("=" * 70)

    rankings = build_rankings()

    save_rankings(
        rankings
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    log("")
    log("=" * 70)
    log("COMPLETE")
    log("=" * 70)

    log(
        f"Total curves: "
        f"{len(rankings):,}"
    )

    log("")
    log("Top 20 curves:")

    print(
        rankings[
            [
                "rank",
                "curve",
                "average_equity",
                "average_bar_change_percent",
                "contribution_percent",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        ),
        flush=True,
    )

    log("")
    log(
        "Contribution total: "
        f"{rankings['contribution_percent'].sum():.6f}%"
    )

    log("")
    log(
        f"Output: {OUTPUT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()