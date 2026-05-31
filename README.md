# Pipeline Doctor

![CI](https://github.com/valtrof/pipeline-doctor/actions/workflows/ci.yml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.12-blue)

LLM-powered data quality system for BigQuery: statistical analysis detects anomalies; Claude generates natural-language diagnosis and executable pandas fix code. The LLM is used for reasoning — never for detection.

Built with Python and the Anthropic Claude API (`claude-haiku-4-5`). Uses only BigQuery public datasets — no proprietary data, no GCP billing required.

## What it does

1. Connects to BigQuery public datasets
2. Scans for data anomalies (nulls, zero values, outliers) using statistical analysis
3. Sends confirmed anomalies to Claude for natural-language diagnosis
4. Returns executable pandas fix code for each issue
5. Exposes all of the above via a REST API

## Architecture

```
POST /analyze
      │
      ▼
┌─────────────────────────────────────┐
│  anomaly_detector.py                │
│                                     │
│  detect_anomalies()                 │
│  → pandas only, deterministic       │
│  → no LLM involved                  │
└──────────────┬──────────────────────┘
               │ confirmed anomalies
               ▼
┌─────────────────────────────────────┐
│  diagnose() + suggest_fixes()       │
│  → Anthropic Claude API             │
│  → explains causes                  │
│  → generates pandas fix code        │
└──────────────┬──────────────────────┘
               │
               ▼
         FastAPI response
```

## Key technical decisions

**LLM as diagnosis layer, not detection layer**
The LLM is never asked "are there anomalies?" — only "given these confirmed anomalies, explain causes and suggest fixes." This prevents hallucination on the detection step and keeps LLM output focused on reasoning rather than data scanning.

**Statistical detection without dependencies**
Anomaly detection uses pandas only — no Great Expectations or external libraries. This keeps the detection stage fast, deterministic, and independently testable. The LLM is only invoked after anomalies have been confirmed statistically.

**Dependency injection over global state**
The Anthropic client is created once at startup and passed into functions as a parameter (`diagnose(df, name, client)`), rather than instantiated inside each function. Unit tests pass a mock client without patching global state or making real API calls.

**FastAPI lifespan for shared resources**
The BigQuery client and LLM client are initialised once in the FastAPI lifespan context and stored on `app.state`. Avoids creating a new connection on every request; surfaces startup failures immediately.

## Quick start (Docker)

```bash
docker build -t pipeline-doctor .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key_here \
  -e GCP_PROJECT_ID=your_gcp_project_id \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  pipeline-doctor
```

Then open `http://localhost:8000/docs` for the interactive API.

## Quick start (local)

```bash
pip install -r requirements.txt
```

Create `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

Authenticate with GCP:
```bash
gcloud auth application-default login
```

Run:
```bash
uvicorn api:app --reload
```

Call the API:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"dataset": "bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022", "limit": 1000}'
```

Or open `http://localhost:8000/docs` for the interactive UI.

## Run tests

```bash
pytest tests/ -v
```

18 tests. No live API or BigQuery calls — mock client injected throughout.

## Project structure

```
api.py              # FastAPI service — HTTP layer, calls anomaly_detector
anomaly_detector.py # Core logic — detection, LLM diagnosis, fix suggestions
main.ipynb          # Notebook — interactive exploration of datasets
tests/              # Unit tests (pytest, no network calls)
requirements.txt
Dockerfile
```

## Sample output

**Dataset:** `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022`

**Anomalies detected:**
```
passenger_count:   156 nulls (15.6%)
rate_code:         156 nulls (15.6%)
store_and_fwd_flag: 156 nulls (15.6%)
airport_fee:       156 nulls (15.6%)
passenger_count:    21 zero values
```

**LLM diagnosis (trimmed):**
> **Anomaly 1: Null values in `passenger_count`, `rate_code`, `store_and_fwd_flag`, `airport_fee`**
> Likely caused by incomplete records at trip completion or data ingestion failures from the source system.
> Recommendation: impute `passenger_count` with column median; fill `store_and_fwd_flag` nulls with `'N'`.
>
> **Anomaly 2: Zero values in `passenger_count`**
> Likely incorrect data entry or system error — a completed trip should have at least one passenger.
> Recommendation: filter out zero-passenger records before analysis; add upstream validation.

**Suggested pandas fixes:**
```python
df['passenger_count'].fillna(df['passenger_count'].median(), inplace=True)
df['rate_code'].fillna(df['rate_code'].mode()[0], inplace=True)
df['store_and_fwd_flag'].fillna('N', inplace=True)
df['airport_fee'].fillna(0, inplace=True)
df['passenger_count'].replace(0, df['passenger_count'].median(), inplace=True)
```

## Datasets used

- `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022`
- `bigquery-public-data.chicago_taxi_trips.taxi_trips`
