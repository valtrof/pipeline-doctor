# Pipeline Doctor

An LLM-powered system that scans BigQuery public datasets for data anomalies, generates natural-language diagnosis of each problem, and recommends pipeline fixes.

Built with Python, LangChain, and the OpenAI API. Uses only BigQuery public datasets — no proprietary data.

## What it does

1. Connects to BigQuery public datasets
2. Scans for data anomalies (nulls, zero values, outliers)
3. Generates a natural-language diagnosis of each anomaly via LLM
4. Recommends concrete pandas fix code for each issue
5. Exposes all of the above via a REST API

## Quick start (Docker)

```bash
docker build -t pipeline-doctor .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e GCP_PROJECT_ID=your_gcp_project_id \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  pipeline-doctor
```

Then open `http://localhost:8000/docs` for the interactive API.

## Quick start (local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API key

Create a `.env` file in this directory:

```
OPENAI_API_KEY=your_key_here
```

### 3. Set up GCP credentials

```bash
gcloud auth application-default login
```

If you don't have the GCP CLI installed: https://cloud.google.com/sdk/docs/install

No GCP billing required — this project uses only BigQuery public datasets.

### 4. Run the API

```bash
uvicorn api:app --reload
```

### 5. Call the API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"dataset": "bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022", "limit": 1000}'
```

Or open `http://localhost:8000/docs` for the interactive UI.

### 6. Run the notebook

Open `main.ipynb` in VS Code or JupyterLab and run all cells.

### 7. Run tests

```bash
pytest tests/ -v
```

## Project structure

```
api.py              # FastAPI service — HTTP layer, calls anomaly_detector
anomaly_detector.py # Core logic — anomaly detection, LLM diagnosis, fix suggestions
main.ipynb          # Notebook — interactive exploration of datasets
tests/              # Unit tests (pytest, no network calls)
requirements.txt    # Python dependencies
Dockerfile          # Container build
.env                # Your API keys (not committed to version control)
```

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   BigQuery      │────>│  anomaly_detector.py │────>│   FastAPI       │
│ Public Datasets │     │                      │     │   api.py        │
└─────────────────┘     │  detect_anomalies()  │     └─────────────────┘
                        │  diagnose()          │
                        │  suggest_fixes()     │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   OpenAI API         │
                        │   (via LangChain)    │
                        └──────────────────────┘
```

### Design decisions

**Separation of concerns — three distinct pipeline stages**

Detection, diagnosis, and fix suggestion are three separate functions rather than one combined function. This makes each stage independently testable and replaceable. Swapping OpenAI for a local Ollama model, for example, requires changing only the LLM instantiation — not the detection logic.

**Dependency injection over global state**

The LLM client is created once at startup and passed into functions as a parameter (`diagnose(df, name, llm)`), rather than instantiated inside each function. This makes unit tests straightforward — tests pass a mock LLM without patching global state or making real API calls.

**FastAPI lifespan for shared resources**

The BigQuery client and LLM are initialised once in the FastAPI lifespan context (`@asynccontextmanager`) and stored on `app.state`. This avoids the cost of creating a new connection on every request, and makes startup failures visible immediately rather than on the first request.

**Statistical detection without dependencies**

Anomaly detection uses pandas only — no external libraries like Great Expectations. This keeps the detection stage fast, deterministic, and independently testable. The LLM is only invoked after anomalies have been confirmed statistically.

**LLM as diagnosis layer, not detection layer**

The LLM is never asked "are there anomalies?" — it is only asked "given these confirmed anomalies, explain causes and suggest fixes." This prevents hallucination on the detection step and keeps the LLM output focused on reasoning rather than data scanning.

## Sample output

### NYC Taxi Trips 2022 — Null and Zero Anomalies

**Anomalies detected:**
```
- passenger_count: 156 nulls (15.6%)
- rate_code: 156 nulls (15.6%)
- store_and_fwd_flag: 156 nulls (15.6%)
- airport_fee: 156 nulls (15.6%)
- passenger_count: 21 zero values
```

**LLM diagnosis (trimmed):**

> ### Anomaly 1: Null Values in `passenger_count`, `rate_code`, `store_and_fwd_flag`, and `airport_fee`
> **Likely Cause**: These null values can arise from data entry errors, incomplete records at the time of trip completion, or issues during data ingestion from the source system.
>
> **Pipeline Recommendation**:
> 1. **Data Imputation**: Consider replacing null values with a default value. For `passenger_count`, replacing nulls with the median or mode of the column is appropriate.
> 2. **Data Validation**: Implement strict validation at the data entry stage to prevent future null entries.
>
> ### Anomaly 2: Zero Values in `passenger_count`
> **Likely Cause**: Zero values likely indicate incorrect data entry or system errors, as a trip should have at least one passenger.
>
> **Pipeline Recommendation**:
> 1. **Filtering**: Set up a validation rule to exclude records with passenger_count = 0 from analysis.
> 2. **Alerts**: Implement alerting mechanisms whenever zero values are encountered.

**Suggested fixes (LLM-generated pandas code):**
```
1. passenger_count nulls → df['passenger_count'].fillna(df['passenger_count'].median(), inplace=True)
2. rate_code nulls → df['rate_code'].fillna(df['rate_code'].mode()[0], inplace=True)
3. store_and_fwd_flag nulls → df['store_and_fwd_flag'].fillna('N', inplace=True)
4. airport_fee nulls → df['airport_fee'].fillna(0, inplace=True)
5. passenger_count zeros → df['passenger_count'].replace(0, df['passenger_count'].median(), inplace=True)
```

---

## Datasets used

- `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022`
- `bigquery-public-data.chicago_taxi_trips.taxi_trips`
