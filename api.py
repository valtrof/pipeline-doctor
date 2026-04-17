import logging
import os
import re
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from google.cloud import bigquery

from anomaly_detector import diagnose, get_llm, suggest_fixes

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
SAMPLE_SIZE_DEFAULT = 1000
SAMPLE_SIZE_MAX = 5000
BQ_TABLE_RE = re.compile(r'^[\w\-]+\.[\w\-]+\.[\w\-]+$')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared clients once at startup, release at shutdown."""
    logger.info("Starting up — initialising BigQuery client and LLM.")
    app.state.bq_client = bigquery.Client(project=PROJECT_ID)
    app.state.llm = get_llm()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Pipeline Doctor",
    description="LLM-powered data quality analysis for BigQuery datasets.",
    version="0.1.0",
    lifespan=lifespan,
)


class AnalyzeRequest(BaseModel):
    dataset: str
    limit: int = SAMPLE_SIZE_DEFAULT


class AnalyzeResponse(BaseModel):
    dataset: str
    anomalies: str
    diagnosis: str
    fixes: str


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, app_request: Request) -> AnalyzeResponse:
    """Fetch a sample from a BigQuery public dataset, detect anomalies, and return diagnosis and fixes."""
    if request.limit > SAMPLE_SIZE_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be {SAMPLE_SIZE_MAX} or fewer.",
        )

    if not BQ_TABLE_RE.match(request.dataset):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset format. Expected: project.dataset.table",
        )

    bq_client = app_request.app.state.bq_client
    llm = app_request.app.state.llm

    logger.info("Analyzing dataset: %s (limit=%d)", request.dataset, request.limit)

    try:
        query = f"SELECT * FROM `{request.dataset}` LIMIT {request.limit}"
        df = bq_client.query(query).to_dataframe()
    except Exception as e:
        logger.error("BigQuery fetch failed for %s: %s", request.dataset, e)
        raise HTTPException(status_code=502, detail=f"BigQuery error: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset returned 0 rows. Check the table name.")

    try:
        anomalies, diagnosis = diagnose(df, request.dataset, llm)
        fixes = suggest_fixes(anomalies, llm)
    except Exception as e:
        logger.error("Analysis failed for %s: %s", request.dataset, e)
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    return AnalyzeResponse(
        dataset=request.dataset,
        anomalies=anomalies,
        diagnosis=diagnosis,
        fixes=fixes,
    )


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
