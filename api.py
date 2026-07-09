import logging
import os
import re
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from google.cloud import bigquery

from agent import run_remediation
from anomaly_detector import diagnose, get_client, suggest_fixes

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# validated in lifespan (not at import) so the module stays importable in tests
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
SAMPLE_SIZE_DEFAULT = 1000
SAMPLE_SIZE_MAX = 5000
BQ_TABLE_RE = re.compile(r'^[\w\-]+\.[\w\-]+\.[\w\-]+$')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared clients once at startup, release at shutdown."""
    logger.info("Starting up — initialising BigQuery and Anthropic clients.")
    if not PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is not set. Add it to your .env file.")
    app.state.bq_client = bigquery.Client(project=PROJECT_ID)
    app.state.client = get_client()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Pipeline Doctor",
    description="LLM-powered data quality analysis for BigQuery datasets.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_bq_client(request: Request) -> bigquery.Client:
    """Dependency provider for the shared BigQuery client."""
    return request.app.state.bq_client


def get_llm_client(request: Request):
    """Dependency provider for the shared Anthropic client."""
    return request.app.state.client


class AnalyzeRequest(BaseModel):
    dataset: str
    limit: int = SAMPLE_SIZE_DEFAULT


class AnalyzeResponse(BaseModel):
    dataset: str
    anomalies: str
    diagnosis: str
    fixes: str


class RemediateRequest(BaseModel):
    dataset: str
    limit: int = SAMPLE_SIZE_DEFAULT


class RemediateResponse(BaseModel):
    status: str
    dataset: str
    summary: str
    baseline: list[dict]
    remaining: list[dict]
    fixes: list[dict]
    verified_fix_count: int
    rows_original: int
    rows_final: int
    turns_used: int
    transcript: list[dict]


def _fetch_sample(bq_client: bigquery.Client, dataset: str, limit: int):
    """Validate the request and fetch a row sample from BigQuery; raise HTTPException on failure."""
    if limit > SAMPLE_SIZE_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be {SAMPLE_SIZE_MAX} or fewer.",
        )

    if not BQ_TABLE_RE.match(dataset):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset format. Expected: project.dataset.table",
        )

    try:
        query = f"SELECT * FROM `{dataset}` LIMIT {limit}"
        df = bq_client.query(query).to_dataframe()
    except Exception as e:
        logger.error("BigQuery fetch failed for %s: %s", dataset, e)
        raise HTTPException(status_code=502, detail=f"BigQuery error: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset returned 0 rows. Check the table name.")

    return df


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    bq_client: bigquery.Client = Depends(get_bq_client),
    client=Depends(get_llm_client),
) -> AnalyzeResponse:
    """Fetch a sample from a BigQuery public dataset, detect anomalies, and return diagnosis and fixes."""
    logger.info("Analyzing dataset: %s (limit=%d)", request.dataset, request.limit)
    df = _fetch_sample(bq_client, request.dataset, request.limit)

    try:
        anomalies, diagnosis = diagnose(df, request.dataset, client)
        fixes = suggest_fixes(anomalies, client)
    except Exception as e:
        logger.error("Analysis failed for %s: %s", request.dataset, e)
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    return AnalyzeResponse(
        dataset=request.dataset,
        anomalies=anomalies,
        diagnosis=diagnosis,
        fixes=fixes,
    )


@app.post("/remediate", response_model=RemediateResponse)
def remediate(
    request: RemediateRequest,
    bq_client: bigquery.Client = Depends(get_bq_client),
    client=Depends(get_llm_client),
) -> RemediateResponse:
    """Run the v2 agentic remediation loop (see DESIGN.md) on a BigQuery dataset sample.

    Fixes are applied to a sandboxed copy only — the source data is never touched.
    The response is a remediation report for human review, including the full
    agent transcript.
    """
    logger.info("Remediating dataset: %s (limit=%d)", request.dataset, request.limit)
    df = _fetch_sample(bq_client, request.dataset, request.limit)

    try:
        report = run_remediation(df, request.dataset, client)
    except Exception as e:
        logger.error("Remediation failed for %s: %s", request.dataset, e)
        raise HTTPException(status_code=500, detail=f"Remediation error: {e}")

    return RemediateResponse(**report)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
