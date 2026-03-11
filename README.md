# DFAST Stress Testing — Fannie Mae Multifamily Loan Portfolio

A production-grade ML pipeline replicating the **Dodd-Frank Act Stress Testing (DFAST)** framework applied to Fannie Mae's publicly available Multifamily Loan Performance data. The project models portfolio default rates under three supervisory macro-economic stress scenarios consistent with FHFA / Federal Reserve methodology.

## Project Overview

Section 165 of Dodd-Frank requires covered institutions to conduct annual forward-looking stress tests. While DFAST was designed for bank holding companies, the FHFA extended analogous requirements to the GSEs (Fannie Mae, Freddie Mac). This project builds a **toy replica** of that exercise:

1. **Data Engineering** (Phase 1) — Ingest, clean, and feature-engineer Fannie Mae MF data
2. **Model Training** (Phase 2) — Train default prediction models under three stress scenarios
3. **Deployment** (Phase 3) — Serve models via FastAPI on AWS with monitoring

### Stress Scenarios

| Scenario | GDP Growth | Unemployment | HPI Change |
|---|---|---|---|
| **Baseline** | +2.0% | 4.5% | 0% |
| **Adverse** | −2.0% | 7.0% | −15% |
| **Severely Adverse** | −6.0% | 10.0% | −30% |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone and install
git clone <repo-url> && cd dfast-stress-testing
make install

# Generate synthetic data (5,000 loans)
make generate-data

# Run the full data pipeline
make pipeline-data

# Run tests
make test

# Launch EDA notebook
make eda
```

## Repository Structure

```
dfast-stress-testing/
├── src/dfast/
│   ├── config.py                  # Pydantic settings (S3, DB, env toggles)
│   ├── data/
│   │   ├── schemas.py             # Pandera schemas (Acquisition, Performance, Feature)
│   │   ├── ingestion.py           # S3 upload/download, pipe-delimited file reader
│   │   ├── cleaning.py            # Null handling, delinquency normalization, default flag
│   │   ├── feature_engineering.py # Loan-level, behavioral, and macro-linked features
│   │   └── macro_scenarios.py     # DFAST scenario definitions and overlay application
│   └── pipelines/
│       └── data_pipeline.py       # Metaflow: ingest → clean → engineer → validate → store
├── scripts/
│   └── generate_sample_data.py    # Synthetic data generator (5K loans, realistic distributions)
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory data analysis
├── tests/
│   ├── conftest.py                # Shared fixtures
│   └── unit/                      # Unit tests for cleaning, features, scenarios
├── pyproject.toml                 # Dependencies (uv-compatible)
├── Makefile                       # Build targets
└── .github/workflows/ci.yml      # Lint → Test → Build
```

## Key Design Decisions

- **D90 Default Definition**: A loan is in default when delinquency status ≥ 90 days past due, consistent with industry standard and Fannie Mae's own loss-severity methodology.
- **Current LTV Estimation**: Uses simplified straight-line amortization and HPI-adjusted property values — a common approximation when detailed amortization schedules are unavailable.
- **Pandera Schema Validation**: Applied at every pipeline boundary to catch data drift early.
- **Local-First Development**: All S3 interactions gracefully fall back to the local filesystem via `USE_LOCAL_STORAGE=True`.

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Install dependencies and pre-commit hooks |
| `make generate-data` | Generate synthetic dataset |
| `make pipeline-data` | Run Metaflow data pipeline |
| `make test` | Run full test suite with coverage |
| `make test-unit` | Run unit tests only |
| `make lint` | Run ruff and mypy |
| `make format` | Auto-format with ruff |
| `make clean` | Remove caches and generated files |

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `USE_LOCAL_STORAGE` | `True` | Use local filesystem instead of S3 |
| `LOCAL_DATA_DIR` | `./data` | Path to local data directory |
| `S3_BUCKET_RAW` | `dfast-raw` | S3 bucket for raw data |
| `S3_BUCKET_FEATURES` | `dfast-features` | S3 bucket for feature tables |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## License

MIT
