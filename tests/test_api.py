# tests/test_api.py
import hashlib
import json
import pickle
import pytest
import sys
from datetime import date
from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import create_tables, get_conn


TRIPLET = "301:CA:SNTL"
RAW_KEY = "test-secret-api-key-xyz"
KEY_HASH = hashlib.sha256(RAW_KEY.encode()).hexdigest()


def _dummy_model():
    m = LinearRegression()
    m.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    return m


@pytest.fixture(scope="module")
def test_db(tmp_path_factory):
    p = tmp_path_factory.mktemp("api_dbs") / "regr_models.db"
    create_tables(p)
    with get_conn(p) as conn:
        # Insert API key
        conn.execute(
            "INSERT INTO api_keys (key_hash, label, created_date, active) VALUES (?,?,?,1)",
            (KEY_HASH, "test", "2026-01-01"),
        )
        # Insert an open outage
        conn.execute(
            "INSERT INTO outages (triplet, parameter, detected_date, detection_method) "
            "VALUES (?,?,?,?)",
            (TRIPLET, "WTEQ", "2026-06-01", "stale"),
        )
        outage_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        # Insert a model for that outage
        pairs = str([(TRIPLET, "WTEQ"), ("391:CA:SNTL", "WTEQ")])
        conn.execute(
            """INSERT INTO auto_models
            (outage_id, rank, stationparameters, regressor_type,
             RMSE_train, RMSE_test, model, trained_date, water_year)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (outage_id, 1, pairs, "Lasso", 0.5, 0.6,
             pickle.dumps(_dummy_model()), "2026-06-01", 2026),
        )
        model_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        # Insert an estimate
        conn.execute(
            """INSERT INTO estimates
            (outage_id, auto_model_id, triplet, parameter,
             date, estimated_value, generated_date)
            VALUES (?,?,?,?,?,?,?)""",
            (outage_id, model_id, TRIPLET, "WTEQ",
             "2026-06-10", 12.5, "2026-06-10"),
        )
    return p


@pytest.fixture(scope="module")
def client(test_db):
    import api.v1.estimates as estimates_module
    estimates_module.DB_PATH = test_db

    from dash import html
    import dbs
    from api.v1.estimates import estimates_bp

    # Dash validates layout in a before_request hook; set a stub so API
    # requests don't trigger NoLayoutException before our decorator runs.
    if dbs.app.layout is None:
        dbs.app.layout = html.Div()

    # Guard against blueprint-already-registered if fixture runs more than once
    if "estimates_v1" not in dbs.app.server.blueprints:
        dbs.app.server.register_blueprint(estimates_bp)

    with dbs.app.server.app_context():
        yield dbs.app.server.test_client()


def _headers(key=RAW_KEY):
    return {"X-API-Key": key}


def test_estimates_no_key_returns_401(client):
    r = client.get(f"/api/v1/estimates/{TRIPLET}/WTEQ")
    assert r.status_code == 401


def test_estimates_bad_key_returns_401(client):
    r = client.get(f"/api/v1/estimates/{TRIPLET}/WTEQ",
                   headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401


def test_estimates_valid_key_returns_200(client):
    r = client.get(f"/api/v1/estimates/{TRIPLET}/WTEQ", headers=_headers())
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["station"] == TRIPLET
    assert data["parameter"] == "WTEQ"
    assert data["unit"] == "inches"
    assert isinstance(data["estimates"], list)
    assert len(data["estimates"]) == 1
    assert abs(data["estimates"][0]["estimated_value"] - 12.5) < 0.01


def test_estimates_date_filter(client):
    r = client.get(
        f"/api/v1/estimates/{TRIPLET}/WTEQ?start=2026-06-11",
        headers=_headers(),
    )
    assert r.status_code == 200
    data = json.loads(r.data)
    assert len(data["estimates"]) == 0  # estimate is before start


def test_outages_no_key_returns_401(client):
    r = client.get("/api/v1/outages")
    assert r.status_code == 401


def test_outages_returns_open_outages(client):
    r = client.get("/api/v1/outages", headers=_headers())
    assert r.status_code == 200
    data = json.loads(r.data)
    assert any(o["triplet"] == TRIPLET for o in data["outages"])


def test_outages_parameter_filter(client):
    r = client.get("/api/v1/outages?parameter=PREC", headers=_headers())
    assert r.status_code == 200
    data = json.loads(r.data)
    assert all(o["parameter"] == "PREC" for o in data["outages"])


def test_station_outages_endpoint(client):
    r = client.get(f"/api/v1/outages/{TRIPLET}", headers=_headers())
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["station"] == TRIPLET
    assert isinstance(data["outages"], list)


def test_status_endpoint(client):
    r = client.get("/api/v1/status", headers=_headers())
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data["status"] == "ok"
    assert "last_job_run" in data
    assert "current_date" in data
