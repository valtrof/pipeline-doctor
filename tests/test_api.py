from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app, get_bq_client, get_llm_client
from test_agent import ScriptedClient, finalize_call, fix_call, response, text, verify_call


class FakeBigQuery:
    """Stands in for google.cloud.bigquery.Client: returns a fixed DataFrame."""

    def __init__(self, df):
        self._df = df

    def query(self, _query):
        return SimpleNamespace(to_dataframe=lambda: self._df)


DF_WITH_NULLS = pd.DataFrame({"a": [1.0, None, 2.0, 1.5], "b": [1, 2, 3, 4]})
VALID_DATASET = "some-project.some_dataset.some_table"


@pytest.fixture
def make_client():
    """Build a TestClient with overridden BigQuery/LLM dependencies.

    TestClient is used without a `with` block on purpose: that skips the lifespan,
    so no real BigQuery or Anthropic client is ever constructed.
    """
    def _make(df=None, llm=None):
        df = DF_WITH_NULLS if df is None else df
        app.dependency_overrides[get_bq_client] = lambda: FakeBigQuery(df)
        app.dependency_overrides[get_llm_client] = lambda: llm if llm is not None else ScriptedClient([])
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /remediate
# ---------------------------------------------------------------------------

def test_remediate_returns_completed_report(make_client):
    llm = ScriptedClient([
        response(fix_call()),
        response(verify_call()),
        response(finalize_call("fixed the nulls")),
    ])
    client = make_client(llm=llm)

    resp = client.post("/remediate", json={"dataset": VALID_DATASET, "limit": 100})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["summary"] == "fixed the nulls"
    assert body["verified_fix_count"] == 1
    assert body["remaining"] == []
    assert body["rows_original"] == 4
    assert body["rows_final"] == 4
    assert len(body["transcript"]) == 3


def test_remediate_clean_data_skips_llm(make_client):
    clean = pd.DataFrame({"a": [1, 2, 3]})
    llm = ScriptedClient([])  # any API call would raise: script is empty
    client = make_client(df=clean, llm=llm)

    resp = client.post("/remediate", json={"dataset": VALID_DATASET})

    assert resp.status_code == 200
    assert resp.json()["status"] == "clean"
    assert llm.calls == []


def test_remediate_rejects_bad_dataset_format(make_client):
    client = make_client()
    resp = client.post("/remediate", json={"dataset": "not a table name"})
    assert resp.status_code == 400
    assert "Invalid dataset format" in resp.json()["detail"]


def test_remediate_rejects_oversized_limit(make_client):
    client = make_client()
    resp = client.post("/remediate", json={"dataset": VALID_DATASET, "limit": 999999})
    assert resp.status_code == 400


def test_remediate_returns_404_for_empty_dataset(make_client):
    client = make_client(df=pd.DataFrame())
    resp = client.post("/remediate", json={"dataset": VALID_DATASET})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /analyze — regression coverage after the shared-fetch refactor
# ---------------------------------------------------------------------------

def test_analyze_still_works(make_client):
    llm = ScriptedClient([
        response(text("diagnosis text")),
        response(text("1. fix code")),
    ])
    client = make_client(llm=llm)

    resp = client.post("/analyze", json={"dataset": VALID_DATASET, "limit": 100})

    assert resp.status_code == 200
    body = resp.json()
    assert "nulls" in body["anomalies"]
    assert body["diagnosis"] == "diagnosis text"
    assert body["fixes"] == "1. fix code"


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
