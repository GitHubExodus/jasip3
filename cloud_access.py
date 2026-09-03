import io
# import os

import boto3
import pandas as pd


R2_ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_ACCESS_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_BUCKET = "stocks-data"

import io
import os

import boto3
import pandas as pd


# ============================================================
# R2 CONFIGURATION
# ============================================================

R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]


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
# DOWNLOAD SYMBOL LIST
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
# DOWNLOAD RAW STOCK DATA
# ============================================================

def download_raw_stock(symbol):

    key = f"{symbol}.parquet"

    log("")
    log("Downloading input data:")
    log(f"  {key}")

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

    log(f"Saved: {key}")