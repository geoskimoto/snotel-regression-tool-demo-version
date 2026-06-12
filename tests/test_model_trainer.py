# tests/test_model_trainer.py
import pickle
import ast
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import create_tables, get_conn
from services.model_trainer import AutoModelTrainer


TRIPLET_A = "301:CA:SNTL"
TRIPLET_B = "391:CA:SNTL"
TRIPLET_C = "428:CA:SNTL"


def _make_meta_df():
    return pd.DataFrame([
        {"triplet": TRIPLET_A, "name": "Adin Mtn",    "latitude": 41.2, "longitude": -120.9},
        {"triplet": TRIPLET_B, "name": "Blue Canyon", "latitude": 39.2, "longitude": -120.7},
        {"triplet": TRIPLET_C, "name": "Caples Lake", "latitude": 38.7, "longitude": -120.0},
    ])


def _online_all():
    """All stations fully online — all values are None (no outage)."""
    return {
        TRIPLET_A: {"WTEQ": None, "PREC": None, "SNWD": None},
        TRIPLET_B: {"WTEQ": None, "PREC": None, "SNWD": None},
        TRIPLET_C: {"WTEQ": None, "PREC": None, "SNWD": None},
    }


def _online_no_same_station_snwd():
    """SNWD at TRIPLET_A is also down."""
    status = _online_all()
    status[TRIPLET_A]["SNWD"] = "null_value"
    return status


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    create_tables(p)
    return p


@pytest.fixture
def trainer(db_path):
    return AutoModelTrainer(db_path, _make_meta_df())


def test_get_nearby_triplets_excludes_self(trainer):
    nearby = trainer._get_nearby_triplets(TRIPLET_A, n=5)
    assert TRIPLET_A not in nearby


def test_get_nearby_triplets_returns_sorted_by_distance(trainer):
    nearby = trainer._get_nearby_triplets(TRIPLET_A, n=2)
    # TRIPLET_B is closer to TRIPLET_A than TRIPLET_C based on lat/lon in fixture
    assert nearby[0] == TRIPLET_B


def test_build_predictor_candidates_includes_same_station_snwd(trainer):
    """When SNWD is online at the outage station, it should be in candidate set 0."""
    candidates = trainer.build_predictor_candidates(TRIPLET_A, "WTEQ", _online_all())
    assert len(candidates) > 0
    flat = [item for cset in candidates for item in cset]
    assert (TRIPLET_A, "SNWD") in flat


def test_build_predictor_candidates_falls_back_when_snwd_down(trainer):
    """When same-station SNWD is down, fallback to nearby stations."""
    candidates = trainer.build_predictor_candidates(
        TRIPLET_A, "WTEQ", _online_no_same_station_snwd()
    )
    assert len(candidates) > 0
    # Same-station SNWD must not appear since it's down
    flat = [item for cset in candidates for item in cset]
    assert (TRIPLET_A, "SNWD") not in flat


def test_build_predictor_candidates_max_three_sets(trainer):
    candidates = trainer.build_predictor_candidates(TRIPLET_A, "WTEQ", _online_all())
    assert len(candidates) <= 3


@pytest.mark.integration
def test_train_for_outage_stores_ranked_models(db_path):
    """Integration: actually trains models against AWDB. Requires network."""
    pytest.importorskip("sklearn")
    meta = _make_meta_df()
    trainer = AutoModelTrainer(db_path, meta)

    # Insert a fake outage
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO outages (triplet, parameter, detected_date, detection_method) "
            "VALUES (?,?,?,?)",
            (TRIPLET_A, "WTEQ", "2026-06-01", "stale"),
        )
        outage_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    trainer.train_for_outage(outage_id, TRIPLET_A, "WTEQ", _online_all())

    with get_conn(db_path) as conn:
        models = conn.execute(
            "SELECT * FROM auto_models WHERE outage_id=? ORDER BY rank",
            (outage_id,),
        ).fetchall()

    assert len(models) >= 1
    ranks = [m["rank"] for m in models]
    assert ranks == sorted(ranks)  # ranks are ordered 1, 2, 3
    rmse_tests = [m["RMSE_test"] for m in models]
    assert rmse_tests == sorted(rmse_tests)  # rank 1 has lowest RMSE_test
    # Verify model blob is a valid pickle
    m = pickle.loads(models[0]["model"])
    assert hasattr(m, "predict")


@pytest.mark.integration
def test_train_for_outage_stationparameters_format(db_path):
    """stationparameters must be parseable as a list of tuples."""
    meta = _make_meta_df()
    trainer = AutoModelTrainer(db_path, meta)
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO outages (triplet, parameter, detected_date, detection_method) "
            "VALUES (?,?,?,?)",
            (TRIPLET_A, "WTEQ", "2026-06-01", "stale"),
        )
        outage_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    trainer.train_for_outage(outage_id, TRIPLET_A, "WTEQ", _online_all())
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT stationparameters FROM auto_models WHERE outage_id=?",
            (outage_id,),
        ).fetchone()
    pairs = ast.literal_eval(row["stationparameters"])
    assert isinstance(pairs, list)
    assert pairs[0] == (TRIPLET_A, "WTEQ")  # response is always first
    assert all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pairs)
