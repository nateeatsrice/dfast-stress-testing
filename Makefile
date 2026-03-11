.PHONY: install lint format test test-unit test-integration docker-build docker-run pipeline-data pipeline-train clean generate-data eda

# ── Environment ──────────────────────────────────────────────
install:
	uv sync --all-extras
	uv run pre-commit install

# ── Code Quality ─────────────────────────────────────────────
lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# ── Testing ──────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/ -m integration -v

# ── Data ─────────────────────────────────────────────────────
generate-data:
	uv run python scripts/generate_sample_data.py

eda:
	uv run jupyter notebook notebooks/01_eda.ipynb

# ── Pipelines ────────────────────────────────────────────────
pipeline-data:
	uv run python -m dfast.pipelines.data_pipeline run

pipeline-train:
	uv run python -m dfast.pipelines.train_pipeline run

# ── Docker ───────────────────────────────────────────────────
docker-build:
	docker build -t dfast-stress-testing .

docker-run:
	docker run --rm -p 8000:8000 dfast-stress-testing

# ── Cleanup ──────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache __pycache__ .mypy_cache .ruff_cache
	rm -rf src/dfast/__pycache__ tests/__pycache__
	rm -rf data/*.csv data/*.parquet
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
