"""Data ingestion: S3 upload/download and local file reading for pipe-delimited data.

When ``USE_LOCAL_STORAGE`` is True (default for dev), all S3 operations fall back
to the local filesystem under ``LOCAL_DATA_DIR``.
"""

import logging
from pathlib import Path

import boto3
import pandas as pd

from dfast.config import settings

logger = logging.getLogger(__name__)

# Column definitions for Fannie Mae Multifamily files
ACQUISITION_COLUMNS: list[str] = [
    "loan_id",
    "origination_date",
    "original_upb",
    "original_ltv",
    "original_dscr",
    "note_rate",
    "property_type",
    "state",
    "loan_purpose",
    "number_of_units",
    "maturity_date",
]

PERFORMANCE_COLUMNS: list[str] = [
    "loan_id",
    "reporting_period",
    "current_upb",
    "delinquency_status",
    "zero_balance_code",
]

ACQUISITION_DTYPES: dict[str, str] = {
    "loan_id": "str",
    "original_upb": "float64",
    "original_ltv": "float64",
    "original_dscr": "float64",
    "note_rate": "float64",
    "property_type": "str",
    "state": "str",
    "loan_purpose": "str",
    "number_of_units": "int64",
}

PERFORMANCE_DTYPES: dict[str, str] = {
    "loan_id": "str",
    "current_upb": "float64",
    "delinquency_status": "str",
    "zero_balance_code": "str",
}


def _get_s3_client() -> "boto3.client":
    """Return a boto3 S3 client."""
    return boto3.client("s3")


def upload_to_s3(local_path: str | Path, bucket: str, key: str) -> None:
    """Upload a local file to an S3 bucket.

    Args:
        local_path: Path to the file on the local filesystem.
        bucket: Target S3 bucket name.
        key: Object key (path) within the bucket.
    """
    if settings.USE_LOCAL_STORAGE:
        dest = Path(settings.LOCAL_DATA_DIR) / bucket / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(str(local_path), str(dest))
        logger.info("Local copy: %s -> %s", local_path, dest)
        return

    client = _get_s3_client()
    client.upload_file(str(local_path), bucket, key)
    logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, key)


def download_from_s3(bucket: str, key: str, local_path: str | Path) -> Path:
    """Download a file from S3 to the local filesystem.

    Args:
        bucket: Source S3 bucket name.
        key: Object key within the bucket.
        local_path: Destination path on the local filesystem.

    Returns:
        The local path where the file was saved.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.USE_LOCAL_STORAGE:
        src = Path(settings.LOCAL_DATA_DIR) / bucket / key
        import shutil

        shutil.copy2(str(src), str(local_path))
        logger.info("Local copy: %s -> %s", src, local_path)
        return local_path

    client = _get_s3_client()
    client.download_file(bucket, key, str(local_path))
    logger.info("Downloaded s3://%s/%s to %s", bucket, key, local_path)
    return local_path


def read_raw_file(
    filepath: str | Path,
    file_type: str = "acquisition",
) -> pd.DataFrame:
    """Read a pipe-delimited Fannie Mae Multifamily raw data file.

    Args:
        filepath: Path to the pipe-delimited text/CSV file.
        file_type: Either ``"acquisition"`` or ``"performance"`` to determine
            column names and dtypes.

    Returns:
        A pandas DataFrame with properly named and typed columns.

    Raises:
        ValueError: If ``file_type`` is not recognized.
    """
    filepath = Path(filepath)

    if file_type == "acquisition":
        columns = ACQUISITION_COLUMNS
        dtypes = ACQUISITION_DTYPES
        parse_dates = ["origination_date", "maturity_date"]
    elif file_type == "performance":
        columns = PERFORMANCE_COLUMNS
        dtypes = PERFORMANCE_DTYPES
        parse_dates = ["reporting_period"]
    elif file_type == "macro":
        # Macro data has its own simple format
        df = pd.read_csv(filepath, sep="|", parse_dates=["date"])
        logger.info("Read macro file %s: %d rows", filepath, len(df))
        return df
    else:
        raise ValueError(f"Unknown file_type: {file_type!r}. Use 'acquisition' or 'performance'.")

    df = pd.read_csv(
        filepath,
        sep="|",
        names=columns if _file_has_no_header(filepath) else None,
        header=0 if not _file_has_no_header(filepath) else None,
        dtype=dtypes,
        parse_dates=parse_dates,
    )

    logger.info("Read %s file %s: %d rows x %d cols", file_type, filepath, *df.shape)
    return df


def _file_has_no_header(filepath: Path) -> bool:
    """Heuristic: if the first cell is numeric or looks like a loan_id, there's no header."""
    with open(filepath) as f:
        first_line = f.readline().strip()
    first_field = first_line.split("|")[0]
    # If the first field matches a known header name, file has headers
    known_headers = {"loan_id", "date", "reporting_period"}
    return first_field.lower() not in known_headers
