import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

import anthropic
from anomaly_detector import detect_anomalies, detect_outliers, diagnose, get_client, suggest_fixes


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

def test_detect_anomalies_finds_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    report = detect_anomalies(df)
    assert "a: 1 nulls" in report


def test_detect_anomalies_finds_zeros():
    df = pd.DataFrame({"a": [0, 1, 2]})
    report = detect_anomalies(df)
    assert "a: 1 zero values" in report


def test_detect_anomalies_finds_both_nulls_and_zeros():
    df = pd.DataFrame({"a": [None, 0, 1]})
    report = detect_anomalies(df)
    assert "nulls" in report
    assert "zero values" in report


def test_detect_anomalies_returns_empty_string_when_clean():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    report = detect_anomalies(df)
    assert report == ""


def test_detect_anomalies_null_percentage_is_correct():
    df = pd.DataFrame({"a": [None, None, 1, 1]})
    report = detect_anomalies(df)
    assert "50.0%" in report


def test_detect_anomalies_ignores_zeros_in_non_numeric_columns():
    df = pd.DataFrame({"a": ["0", "1", "2"]})  # string column
    report = detect_anomalies(df)
    assert "zero values" not in report


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------

def test_get_client_raises_when_api_key_missing():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_client()


def test_get_client_returns_instance_when_key_present():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        client = get_client()
    assert isinstance(client, anthropic.Anthropic)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_client_mock(response_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_message
    return mock_client


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

def test_diagnose_returns_anomalies_and_diagnosis():
    df = pd.DataFrame({"a": [None, 1, 2]})
    mock_client = _make_client_mock("Fix the nulls.")
    anomalies, diagnosis = diagnose(df, "Test Dataset", mock_client)
    assert "nulls" in anomalies
    assert diagnosis == "Fix the nulls."


def test_diagnose_skips_llm_when_no_anomalies():
    df = pd.DataFrame({"a": [1, 2, 3]})
    anomalies, diagnosis = diagnose(df, "Clean Dataset", MagicMock())
    assert anomalies == ""
    assert diagnosis == "No anomalies detected."


def test_diagnose_passes_dataset_name_to_llm():
    df = pd.DataFrame({"a": [None, 1]})
    mock_client = _make_client_mock("ok")
    diagnose(df, "My Dataset", mock_client)
    call_kwargs = mock_client.messages.create.call_args
    # dataset name appears in the formatted prompt passed as messages content
    prompt_text = call_kwargs[1]["messages"][0]["content"]
    assert "My Dataset" in prompt_text


def test_diagnose_raises_on_llm_failure():
    df = pd.DataFrame({"a": [None, 1]})
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API timeout")
    with pytest.raises(RuntimeError, match="API timeout"):
        diagnose(df, "Test Dataset", mock_client)


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------

def test_detect_anomalies_finds_outliers():
    df = pd.DataFrame({"a": [1, 2, 3, 1, 2, 999]})
    report = detect_anomalies(df)
    assert "outliers" in report
    assert "IQR method" in report


def test_detect_anomalies_no_outliers_when_clean():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    report = detect_anomalies(df)
    assert "outliers" not in report


def test_detect_anomalies_ignores_outliers_in_non_numeric():
    df = pd.DataFrame({"a": ["a", "b", "c", "z"]})
    report = detect_anomalies(df)
    assert "outliers" not in report


# ---------------------------------------------------------------------------
# suggest_fixes
# ---------------------------------------------------------------------------

def test_suggest_fixes_returns_fix_string():
    anomalies = "- passenger_count: 10 nulls (10.0%)"
    mock_client = _make_client_mock("1. df['passenger_count'].fillna(df['passenger_count'].median(), inplace=True)")
    result = suggest_fixes(anomalies, mock_client)
    assert "passenger_count" in result


def test_suggest_fixes_returns_no_fixes_when_empty():
    result = suggest_fixes("", MagicMock())
    assert result == "No fixes needed."


def test_suggest_fixes_raises_on_llm_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API timeout")
    with pytest.raises(RuntimeError, match="API timeout"):
        suggest_fixes("- col: 1 nulls (10.0%)", mock_client)
