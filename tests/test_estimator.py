# tests/test_estimator.py
import pickle
import ast
import pytest
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import create_tables, get_conn
from services.estimator import Estimator

TRIPLET_A = "301:CA:SNTL"
TRIPLET_B = "391:CA:SNTL"


def _dummy_model():
    """A real scikit-learn model trained on trivial data for pickle testing."""
    from sklearn.linear_model import LinearRegression
    import numpy as np
    m = LinearRegression()
    m.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
    return m


def _insert_outage(conn, triplet=TRIPLET_A, parameter="WTEQ"):
    conn.execute(
        "INSERT INTO outages (triplet, parameter, detected_date, detection_method) "
        "VALUES (?,?,?,?)",
        (triplet, parameter, "2026-06-01", "stale"),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_auto_model(conn, outage_id, rank, triplet_b=TRIPLET_B):
    pairs = str([(TRIPLET_A, "WTEQ"), (triplet_b, "WTEQ")])
    conn.execute(
        """INSERT INTO auto_models
        (outage_id, rank, stationparameters, regressor_type,
         RMSE_train, RMSE_test, model, trained_date, water_year)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (outage_id, rank, pairs, "Lasso", 1.0, 1.0,
         pickle.dumps(_dummy_model()), "2026-06-01", 2026),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    create_tables(p)
    return p


def _online_all():
    return {
        TRIPLET_A: {"WTEQ": None, "PREC": None, "SNWD": None},
        TRIPLET_B: {"WTEQ": None, "PREC": None, "SNWD": None},
    }


def _online_b_down():
    status = _online_all()
    status[TRIPLET_B]["WTEQ"] = "stale"
    return status


def test_estimate_stored_when_predictors_online(db_path):
    with get_conn(db_path) as conn:
        outage_id = _insert_outage(conn)
        _insert_auto_model(conn, outage_id, rank=1)

    estimator = Estimator(db_path)
    today = date.today().isoformat()

    # Mock ModelPredictor.predict to avoid real API call
    mock_preds = np.array([14.5])
    with patch("services.estimator.ModelPredictor.predict") as mock_predict:
        mock_predict.return_value = (mock_preds, None, None, None)
        estimator.run(_online_all())

    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM estimates").fetchone()
    assert row is not None
    assert abs(row["estimated_value"] - 14.5) < 0.01
    assert row["triplet"] == TRIPLET_A
    assert row["date"] == today


def test_falls_back_to_rank2_when_rank1_predictors_offline(db_path):
    with get_conn(db_path) as conn:
        outage_id = _insert_outage(conn)
        _insert_auto_model(conn, outage_id, rank=1, triplet_b=TRIPLET_B)
        # Rank 2 uses TRIPLET_A's SNWD (same station, always "online" from status)
        pairs2 = str([(TRIPLET_A, "WTEQ"), (TRIPLET_A, "SNWD")])
        conn.execute(
            """INSERT INTO auto_models
            (outage_id, rank, stationparameters, regressor_type,
             RMSE_train, RMSE_test, model, trained_date, water_year)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (outage_id, 2, pairs2, "Lasso", 1.2, 1.2,
             pickle.dumps(_dummy_model()), "2026-06-01", 2026),
        )
        rank2_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    estimator = Estimator(db_path)
    mock_preds = np.array([11.0])
    with patch("services.estimator.ModelPredictor.predict") as mock_predict:
        mock_predict.return_value = (mock_preds, None, None, None)
        # TRIPLET_B WTEQ is down — rank 1 model should be skipped
        estimator.run(_online_b_down())

    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM estimates").fetchone()
    assert row is not None
    assert row["auto_model_id"] == rank2_id  # rank 2 was used


def test_no_estimate_when_all_predictors_offline(db_path):
    with get_conn(db_path) as conn:
        outage_id = _insert_outage(conn)
        _insert_auto_model(conn, outage_id, rank=1, triplet_b=TRIPLET_B)

    estimator = Estimator(db_path)
    # Mark TRIPLET_B offline — the only predictor
    all_offline = {
        TRIPLET_A: {"WTEQ": "stale", "PREC": None, "SNWD": "stale"},
        TRIPLET_B: {"WTEQ": "stale", "PREC": "stale", "SNWD": "stale"},
    }
    with patch("services.estimator.ModelPredictor.predict") as mock_predict:
        estimator.run(all_offline)
        mock_predict.assert_not_called()

    with get_conn(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM estimates").fetchone()[0]
    assert count == 0
