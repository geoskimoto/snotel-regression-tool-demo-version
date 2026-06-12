# tests/test_outage_detector.py
import sqlite3
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import create_tables, get_conn
from services.outage_detector import OutageDetector, STALENESS_DAYS


def _make_meta_df(triplet="301:CA:SNTL"):
    return pd.DataFrame([{
        "triplet": triplet, "name": "Test Station",
        "latitude": 41.0, "longitude": -120.0,
    }])


def _make_values(days_ago=0, value=10.0):
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return [{"date": d, "value": value}]


def _mock_awdb_response(triplet, values):
    return [{
        "stationTriplet": triplet,
        "data": [{"values": values}],
    }]


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    create_tables(p)
    return p


@pytest.fixture
def detector(db_path):
    meta = _make_meta_df()
    return OutageDetector(db_path, meta)


def test_null_value_flagged_as_outage(detector, db_path):
    values = _make_values(days_ago=0, value=None)
    with patch("services.outage_detector.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _mock_awdb_response("301:CA:SNTL", values),
        )
        result = detector.check_station_param("301:CA:SNTL", "WTEQ")
    assert result == "null_value"


def test_stale_data_flagged_as_outage(detector):
    values = _make_values(days_ago=STALENESS_DAYS + 1, value=5.0)
    with patch("services.outage_detector.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _mock_awdb_response("301:CA:SNTL", values),
        )
        result = detector.check_station_param("301:CA:SNTL", "WTEQ")
    assert result == "stale"


def test_fresh_data_returns_none(detector):
    values = _make_values(days_ago=0, value=12.0)
    with patch("services.outage_detector.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _mock_awdb_response("301:CA:SNTL", values),
        )
        result = detector.check_station_param("301:CA:SNTL", "WTEQ")
    assert result is None


def test_run_inserts_new_outage(detector, db_path):
    online_status = {"301:CA:SNTL": {"WTEQ": "null_value", "PREC": None}}
    new_outages, resolved = detector.run(online_status)
    assert len(new_outages) == 1
    assert new_outages[0]["triplet"] == "301:CA:SNTL"
    assert new_outages[0]["parameter"] == "WTEQ"
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM outages").fetchall()
    assert len(rows) == 1
    assert rows[0]["detection_method"] == "null_value"
    assert rows[0]["resolved_date"] is None


def test_run_resolves_existing_outage(detector, db_path):
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO outages (triplet, parameter, detected_date, detection_method) VALUES (?,?,?,?)",
            ("301:CA:SNTL", "WTEQ", "2026-06-01", "stale"),
        )
    online_status = {"301:CA:SNTL": {"WTEQ": None, "PREC": None}}
    new_outages, resolved = detector.run(online_status)
    assert len(resolved) == 1
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM outages WHERE id=1").fetchone()
    assert row["resolved_date"] == date.today().isoformat()


def test_run_skips_already_open_outage(detector, db_path):
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO outages (triplet, parameter, detected_date, detection_method) VALUES (?,?,?,?)",
            ("301:CA:SNTL", "WTEQ", "2026-06-01", "stale"),
        )
    online_status = {"301:CA:SNTL": {"WTEQ": "stale", "PREC": None}}
    new_outages, _ = detector.run(online_status)
    assert len(new_outages) == 0
    with get_conn(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM outages").fetchone()[0]
    assert count == 1
