"""Metaflow data pipeline: ingest → clean → engineer features → validate → store.

Run locally with:
    python -m dfast.pipelines.data_pipeline run

Or via Makefile:
    make pipeline-data
"""

import logging
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, step

from dfast.config import settings

logger = logging.getLogger(__name__)


class DataPipeline(FlowSpec):
    """End-to-end data pipeline for DFAST feature table construction.

    Steps:
        start → ingest → clean → engineer_features → validate → store → end
    """

    data_dir = Parameter(
        "data-dir",
        help="Local directory containing raw data files",
        default=settings.LOCAL_DATA_DIR,
    )

    output_bucket = Parameter(
        "output-bucket",
        help="S3 bucket for storing feature table (or local dir name)",
        default=settings.S3_BUCKET_FEATURES,
    )

    @step
    def start(self):
        """Initialize pipeline, log configuration."""
        logger.info("DataPipeline starting — data_dir=%s", self.data_dir)
        self.next(self.ingest)

    @step
    def ingest(self):
        """Read raw pipe-delimited files into DataFrames."""
        from dfast.data.ingestion import read_raw_file

        data_path = Path(self.data_dir)

        self.acquisition_df = read_raw_file(data_path / "acquisition.csv", file_type="acquisition")
        self.performance_df = read_raw_file(data_path / "performance.csv", file_type="performance")
        self.macro_df = read_raw_file(data_path / "macro.csv", file_type="macro")

        logger.info(
            "Ingested: %d acquisitions, %d performance records, %d macro rows",
            len(self.acquisition_df),
            len(self.performance_df),
            len(self.macro_df),
        )
        self.next(self.clean)

    @step
    def clean(self):
        """Apply cleaning transformations and create default flag."""
        from dfast.data.cleaning import (
            clean_acquisition,
            clean_performance,
            create_default_flag,
            remove_duplicates,
        )

        self.acquisition_df = clean_acquisition(self.acquisition_df)
        self.performance_df = clean_performance(self.performance_df)
        self.performance_df = remove_duplicates(self.performance_df)
        self.performance_df = create_default_flag(self.performance_df)

        logger.info(
            "Cleaned: %d acquisitions, %d performance records",
            len(self.acquisition_df),
            len(self.performance_df),
        )
        self.next(self.engineer_features)

    @step
    def engineer_features(self):
        """Build the modeling-ready feature table."""
        from dfast.data.feature_engineering import build_feature_table

        self.features_df = build_feature_table(
            self.acquisition_df, self.performance_df, self.macro_df
        )
        logger.info("Feature table: %d rows x %d cols", *self.features_df.shape)
        self.next(self.validate)

    @step
    def validate(self):
        """Run Pandera schema validation on the final feature table."""
        from dfast.data.schemas import FeatureSchema

        self.features_df = FeatureSchema.validate(self.features_df)
        logger.info("Schema validation passed")
        self.next(self.store)

    @step
    def store(self):
        """Persist the feature table as Parquet."""
        if settings.USE_LOCAL_STORAGE:
            out_dir = Path(self.data_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "features.parquet"
            self.features_df.to_parquet(out_path, index=False)
            logger.info("Feature table saved to %s", out_path)
        else:
            from dfast.data.ingestion import upload_to_s3
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                self.features_df.to_parquet(tmp.name, index=False)
                upload_to_s3(tmp.name, self.output_bucket, "features/features.parquet")
                logger.info("Feature table uploaded to s3://%s/features/features.parquet", self.output_bucket)

        self.next(self.end)

    @step
    def end(self):
        """Log summary statistics."""
        n_loans = self.features_df["loan_id"].nunique()
        n_rows = len(self.features_df)
        default_rate = self.features_df["is_default"].mean()

        logger.info("=" * 60)
        logger.info("DataPipeline complete")
        logger.info("  Unique loans : %d", n_loans)
        logger.info("  Total rows   : %d", n_rows)
        logger.info("  Default rate : %.2f%%", default_rate * 100)
        logger.info("=" * 60)


if __name__ == "__main__":
    DataPipeline()
