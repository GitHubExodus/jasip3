import io
# import os

import boto3
import pandas as pd


R2_ENDPOINT_URL = "https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = "00e18b0c16ecb3395cd6f7c8e0eb3554"
R2_SECRET_ACCESS_KEY = "33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0"
R2_BUCKET = "stocks-data"




s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
)


def log(message):
    print(message, flush=True)


def download_raw_stock(symbol):
    """
    Raw 1-minute stock data is stored directly
    in the bucket root.

    Example:
        AAPL.parquet
        DELL.parquet
    """

    key = f"{symbol}.parquet"

    log(f"Downloading raw data:")
    log(f"  {key}")

    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key=key,
    )

    data = response["Body"].read()

    return pd.read_parquet(
        io.BytesIO(data)
    )


def save_dataframe(df, key):
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


def get_stock_symbols():
    response = s3.get_object(
        Bucket=R2_BUCKET,
        Key="misc/symbols.txt",
    )

    text = response["Body"].read().decode(
        "utf-8"
    )

    return [
        x.strip().upper()
        for x in text.splitlines()
        if x.strip()
    ]