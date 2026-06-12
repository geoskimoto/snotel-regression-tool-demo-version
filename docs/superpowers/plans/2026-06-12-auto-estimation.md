# Auto-Estimation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated gap-filling system that detects WTEQ/PREC sensor outages at SNOTEL stations, trains regression models on-demand, generates daily estimates, and serves them via a versioned REST API.

**Architecture:** A standalone `services/daily_job.py` runs as a systemd timer at 15:30 UTC daily. It orchestrates three steps: outage detection (sqlite3), model training (reuses `RegressionFun`), and estimation (reuses `ModelPredictor`). A Flask blueprint mounted on the existing Dash app serves estimates at `/api/v1/`. All new tables live in the existing `dbs/regr_models.db`. All files owned by `snotel:snotel`.

**Tech Stack:** Python 3.12, sqlite3, scikit-learn (via existing `views/regression.py`), Flask Blueprint, pytest, systemd timer

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `services/__init__.py` | package marker |
| Create | `services/db.py` | sqlite3 helpers, CREATE TABLE SQL, shared by services and API |
| Create | `services/outage_detector.py` | Step 1 — detect and record outages |
| Create | `services/model_trainer.py` | Step 2 — train 3 ranked models for new outages |
| Create | `services/estimator.py` | Step 3 — daily gap-filling via stored models |
| Create | `services/daily_job.py` | orchestrator entry point |
| Create | `api/__init__.py` | package marker |
| Create | `api/v1/__init__.py` | package marker |
| Create | `api/v1/estimates.py` | Flask blueprint + 4 endpoints + API key auth |
| Create | `scripts/manage_api_keys.py` | CLI for key generation, listing, deactivation |
| Create | `tests/test_db.py` | table creation tests |
| Create | `tests/test_outage_detector.py` | outage detection unit tests |
| Create | `tests/test_model_trainer.py` | predictor candidate + training unit tests |
| Create | `tests/test_estimator.py` | estimation unit tests |
| Create | `tests/test_api.py` | API endpoint + auth tests |
| Modify | `app.py` | register blueprint, call create_tables() at startup |

---

## Task 1: Database Helper (`services/db.py`)

**Files:**
- Create: `services/__init__.py`
- Create: `services/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import sqlite3
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import get_conn, create_tables


def test_create_tables_creates_all_four_tables(tmp_path):
    db_path = tmp_path / "test.db"
    create_tables(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert {"outages", "auto_models", "estimates", "api_keys"} <= tables


def test_get_conn_row_factory_enables_column_access(tmp_path):
    db_path = tmp_path / "test.db"
    create_tables(db_path)
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, label, created_date, active) VALUES (?,?,?,1)",
            ("abc123", "test-key", "2026-01-01"),
        )
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM api_keys").fetchone()
    assert row["label"] == "test-key"
    assert row["active"] == 1


def test_create_tables_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    create_tables(db_path)
    create_tables(db_path)  # second call must not raise
    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    assert count >= 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/snotel/htdocs/app
sudo -u snotel .venv/bin/pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'services'`

- [ ] **Step 3: Create package markers and implement `services/db.py`**

```python
# services/__init__.py
```

```python
# services/db.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path

MODELS_DB_PATH = Path(__file__).parent.parent / "dbs" / "regr_models.db"

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS outages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triplet VARCHAR(50) NOT NULL,
    parameter VARCHAR(20) NOT NULL,
    detected_date DATE NOT NULL,
    resolved_date DATE,
    detection_method VARCHAR(20) NOT NULL
);
CREATE TABLE IF NOT EXISTS auto_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outage_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    stationparameters TEXT NOT NULL,
    regressor_type VARCHAR(50) NOT NULL,
    RMSE_train REAL NOT NULL,
    RMSE_test REAL NOT NULL,
    model BLOB NOT NULL,
    trained_date DATE NOT NULL,
    water_year INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    outage_id INTEGER NOT NULL,
    auto_model_id INTEGER NOT NULL,
    triplet VARCHAR(50) NOT NULL,
    parameter VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    estimated_value REAL NOT NULL,
    generated_date DATE NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    label VARCHAR(100) NOT NULL,
    created_date DATE NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
"""


@contextmanager
def get_conn(db_path=None):
    path = str(db_path or MODELS_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_tables(db_path=None):
    with get_conn(db_path) as conn:
        conn.executescript(CREATE_TABLES_SQL)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
sudo -u snotel .venv/bin/pytest tests/test_db.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Fix ownership and commit**

```bash
sudo chown -R snotel:snotel /home/snotel/htdocs/app/services /home/snotel/htdocs/app/tests/test_db.py
sudo -u snotel git add services/__init__.py services/db.py tests/test_db.py
sudo -u snotel git commit -m "feat: add services/db.py with sqlite3 helpers and auto-estimation table schema"
```

---

## Task 2: OutageDetector (`services/outage_detector.py`)

**Files:**
- Create: `services/outage_detector.py`
- Create: `tests/test_outage_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    """Return a AWDB-style values list with one entry N days ago."""
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
    # Insert a pre-existing open outage
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO outages (triplet, parameter, detected_date, detection_method) VALUES (?,?,?,?)",
            ("301:CA:SNTL", "WTEQ", "2026-06-01", "stale"),
        )
    # Station is now back online
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
    assert count == 1  # no duplicate inserted
```

- [ ] **Step 2: Run test to verify it fails**

```bash
sudo -u snotel .venv/bin/pytest tests/test_outage_detector.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.outage_detector'`

- [ ] **Step 3: Implement `services/outage_detector.py`**

```python
# services/outage_detector.py
import logging
import requests
from datetime import date, timedelta

from services.db import get_conn

logger = logging.getLogger(__name__)

PARAMETERS = ("WTEQ", "PREC")
CHECK_PARAMS = ("WTEQ", "PREC", "SNWD")
STALENESS_DAYS = 3
LOOKBACK_DAYS = 5
API_SERVER = "https://wcc.sc.egov.usda.gov/awdbRestApi"


class OutageDetector:
    def __init__(self, db_path, meta_df):
        self.db_path = db_path
        self.meta_df = meta_df

    def _fetch_recent(self, triplet, parameter):
        """Fetch last LOOKBACK_DAYS values from AWDB. No cache — always fresh."""
        end = date.today()
        start = end - timedelta(days=LOOKBACK_DAYS)
        params = {
            "stationTriplets": triplet,
            "elements": parameter,
            "duration": "DAILY",
            "beginDate": start.isoformat(),
            "endDate": end.isoformat(),
        }
        try:
            r = requests.get(
                f"{API_SERVER}/services/v1/data", params=params, timeout=30
            )
            if r.ok:
                data = r.json()
                if data and data[0].get("data"):
                    return data[0]["data"][0].get("values", [])
        except Exception as e:
            logger.warning(f"Failed to fetch {triplet}/{parameter}: {e}")
        return []

    def check_station_param(self, triplet, parameter):
        """
        Returns 'null_value', 'stale', or None (online).
        None means the station/parameter is reporting good data.
        """
        values = self._fetch_recent(triplet, parameter)
        if not values:
            return "stale"
        latest = values[-1]
        if latest.get("value") is None:
            return "null_value"
        try:
            latest_date = date.fromisoformat(latest["date"])
        except (KeyError, ValueError):
            return "stale"
        if (date.today() - latest_date).days >= STALENESS_DAYS:
            return "stale"
        return None

    def build_online_status(self):
        """
        Check WTEQ, PREC, and SNWD for all stations.
        Returns {triplet: {param: None|'null_value'|'stale'}}.
        None means online and healthy.
        """
        status = {}
        for _, row in self.meta_df.iterrows():
            triplet = row["triplet"]
            status[triplet] = {}
            for param in CHECK_PARAMS:
                status[triplet][param] = self.check_station_param(triplet, param)
        return status

    def run(self, online_status):
        """
        Reconcile online_status against the outages table.
        Returns (new_outages, resolved_outages) as lists of dicts.
        """
        new_outages = []
        resolved = []
        today = date.today().isoformat()

        with get_conn(self.db_path) as conn:
            for _, row in self.meta_df.iterrows():
                triplet = row["triplet"]
                for parameter in PARAMETERS:
                    method = online_status.get(triplet, {}).get(parameter)
                    is_down = method is not None

                    existing = conn.execute(
                        "SELECT * FROM outages WHERE triplet=? AND parameter=? "
                        "AND resolved_date IS NULL",
                        (triplet, parameter),
                    ).fetchone()

                    if is_down and not existing:
                        conn.execute(
                            "INSERT INTO outages "
                            "(triplet, parameter, detected_date, detection_method) "
                            "VALUES (?,?,?,?)",
                            (triplet, parameter, today, method),
                        )
                        outage_id = conn.execute(
                            "SELECT last_insert_rowid()"
                        ).fetchone()[0]
                        new_outages.append({
                            "id": outage_id,
                            "triplet": triplet,
                            "parameter": parameter,
                        })
                    elif not is_down and existing:
                        conn.execute(
                            "UPDATE outages SET resolved_date=? WHERE id=?",
                            (today, existing["id"]),
                        )
                        resolved.append(dict(existing))

        return new_outages, resolved
```

- [ ] **Step 4: Run test to verify it passes**

```bash
sudo -u snotel .venv/bin/pytest tests/test_outage_detector.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Fix ownership and commit**

```bash
sudo chown snotel:snotel /home/snotel/htdocs/app/services/outage_detector.py \
  /home/snotel/htdocs/app/tests/test_outage_detector.py
sudo -u snotel git add services/outage_detector.py tests/test_outage_detector.py
sudo -u snotel git commit -m "feat: add OutageDetector — detects null/stale WTEQ and PREC sensor failures"
```

---

## Task 3: AutoModelTrainer (`services/model_trainer.py`)

**Files:**
- Create: `services/model_trainer.py`
- Create: `tests/test_model_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
sudo -u snotel .venv/bin/pytest tests/test_model_trainer.py -v -k "not train_for_outage"
```

Expected: `ModuleNotFoundError: No module named 'services.model_trainer'`

- [ ] **Step 3: Implement `services/model_trainer.py`**

```python
# services/model_trainer.py
import ast
import logging
import pickle
from datetime import date, timedelta

from dateutil.relativedelta import relativedelta

from services.db import get_conn
from views.regression import RegressionFun

logger = logging.getLogger(__name__)

CORRELATED_PARAMS = {
    "WTEQ": ["SNWD", "PREC"],
    "PREC": ["WTEQ", "SNWD"],
}


def _water_year(d=None):
    d = d or date.today()
    return d.year + 1 if d.month >= 10 else d.year


class AutoModelTrainer:
    def __init__(self, db_path, meta_df):
        self.db_path = db_path
        self.meta_df = meta_df

    def _get_nearby_triplets(self, triplet, n=5):
        row = self.meta_df[self.meta_df["triplet"] == triplet]
        if row.empty:
            return []
        lat = row.iloc[0]["latitude"]
        lon = row.iloc[0]["longitude"]
        others = self.meta_df[self.meta_df["triplet"] != triplet].copy()
        others["_dist"] = (others["latitude"] - lat) ** 2 + (others["longitude"] - lon) ** 2
        return others.nsmallest(n, "_dist")["triplet"].tolist()

    def build_predictor_candidates(self, triplet, parameter, online_status):
        """
        Build up to 3 predictor candidate sets from the prioritized pool.
        Returns list of lists of (triplet, param) tuples.
        online_status value of None means the sensor is healthy/online.
        """
        correlates = CORRELATED_PARAMS.get(parameter, [])
        nearby = self._get_nearby_triplets(triplet, n=5)

        pool = []
        # Priority 1: same-station correlated sensors that are online
        for p in correlates:
            if online_status.get(triplet, {}).get(p) is None:
                pool.append((triplet, p))
        # Priority 2: nearest stations, same parameter, online
        for t in nearby:
            if online_status.get(t, {}).get(parameter) is None:
                pool.append((t, parameter))
        # Priority 3: nearest stations, correlated params, online
        for t in nearby:
            for p in correlates:
                cand = (t, p)
                if cand not in pool and online_status.get(t, {}).get(p) is None:
                    pool.append(cand)

        if not pool:
            return []

        # Sliding window of 3 predictors each, up to 3 candidate sets
        candidates = []
        for i in range(min(3, len(pool))):
            cset = pool[i: i + 3]
            if cset and cset not in candidates:
                candidates.append(cset)

        return candidates

    def train_for_outage(self, outage_id, triplet, parameter, online_status):
        """Train up to 3 ranked models for one outage. Stores results to auto_models."""
        candidates = self.build_predictor_candidates(triplet, parameter, online_status)
        if not candidates:
            logger.warning(f"No predictor candidates available for {triplet}/{parameter}")
            return

        today = date.today()
        train_end = (today - timedelta(days=1)).isoformat()
        train_start = (today - relativedelta(years=10)).isoformat()
        wy = _water_year(today)

        trained = []
        for cset in candidates:
            pairs = [(triplet, parameter)] + cset
            try:
                model = RegressionFun(pairs, train_start, train_end)
                model.train_model("Lasso", 0.3)
                trained.append({
                    "stationparameters": str(pairs),
                    "regressor_type": "Lasso",
                    "RMSE_train": float(model.RMSE_train),
                    "RMSE_test": float(model.RMSE_test),
                    "model_blob": pickle.dumps(model.regr),
                })
            except Exception as e:
                logger.warning(f"Training failed for candidate {cset}: {e}")

        if not trained:
            logger.warning(f"All candidate sets failed for outage {outage_id}")
            return

        trained.sort(key=lambda x: x["RMSE_test"])

        with get_conn(self.db_path) as conn:
            for rank, t in enumerate(trained[:3], start=1):
                conn.execute(
                    """INSERT INTO auto_models
                    (outage_id, rank, stationparameters, regressor_type,
                     RMSE_train, RMSE_test, model, trained_date, water_year)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        outage_id, rank, t["stationparameters"], t["regressor_type"],
                        t["RMSE_train"], t["RMSE_test"], t["model_blob"],
                        today.isoformat(), wy,
                    ),
                )

    def run(self, new_outages, online_status):
        for outage in new_outages:
            logger.info(f"Training models for new outage: {outage['triplet']}/{outage['parameter']}")
            self.train_for_outage(
                outage["id"], outage["triplet"], outage["parameter"], online_status
            )
```

- [ ] **Step 4: Run unit tests (skip network tests)**

```bash
sudo -u snotel .venv/bin/pytest tests/test_model_trainer.py -v -k "not train_for_outage"
```

Expected: `5 passed`

- [ ] **Step 5: Run integration tests (requires network)**

```bash
sudo -u snotel .venv/bin/pytest tests/test_model_trainer.py -v -k "train_for_outage"
```

Expected: `2 passed` (makes real AWDB calls — may take ~30 seconds)

- [ ] **Step 6: Fix ownership and commit**

```bash
sudo chown snotel:snotel /home/snotel/htdocs/app/services/model_trainer.py \
  /home/snotel/htdocs/app/tests/test_model_trainer.py
sudo -u snotel git add services/model_trainer.py tests/test_model_trainer.py
sudo -u snotel git commit -m "feat: add AutoModelTrainer — builds predictor candidates and trains ranked Lasso models on outage detection"
```

---

## Task 4: Estimator (`services/estimator.py`)

**Files:**
- Create: `services/estimator.py`
- Create: `tests/test_estimator.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
sudo -u snotel .venv/bin/pytest tests/test_estimator.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.estimator'`

- [ ] **Step 3: Implement `services/estimator.py`**

```python
# services/estimator.py
import ast
import logging
import pickle
from datetime import date

from services.db import get_conn
from views.regression import ModelPredictor

logger = logging.getLogger(__name__)


class Estimator:
    def __init__(self, db_path):
        self.db_path = db_path

    def estimate_for_outage(self, outage, online_status):
        """
        Select the highest-ranked model whose predictor stations are all online,
        run prediction for today, and store the result in estimates.
        online_status value of None means the sensor is healthy/online.
        """
        with get_conn(self.db_path) as conn:
            models = conn.execute(
                "SELECT * FROM auto_models WHERE outage_id=? ORDER BY rank",
                (outage["id"],),
            ).fetchall()
            models = [dict(m) for m in models]

        today = date.today().isoformat()

        for am in models:
            pairs = ast.literal_eval(am["stationparameters"])
            predictor_pairs = pairs[1:]  # pairs[0] is the response (outage station)
            all_online = all(
                online_status.get(t, {}).get(p) is None
                for t, p in predictor_pairs
            )
            if not all_online:
                continue

            try:
                model_obj = pickle.loads(am["model"])
                predictor = ModelPredictor(model_obj, pairs)
                predictions, _, _, _ = predictor.predict(today, today)
                estimated_value = float(predictions[0])

                with get_conn(self.db_path) as conn:
                    conn.execute(
                        """INSERT INTO estimates
                        (outage_id, auto_model_id, triplet, parameter,
                         date, estimated_value, generated_date)
                        VALUES (?,?,?,?,?,?,?)""",
                        (
                            outage["id"], am["id"],
                            outage["triplet"], outage["parameter"],
                            today, estimated_value, today,
                        ),
                    )
                logger.info(
                    f"Estimated {outage['triplet']}/{outage['parameter']} "
                    f"= {estimated_value:.3f} (model rank {am['rank']})"
                )
                return
            except Exception as e:
                logger.warning(f"Estimation failed with model {am['id']}: {e}")

        logger.warning(
            f"No viable model for outage {outage['id']} "
            f"({outage['triplet']}/{outage['parameter']}) — all predictors offline"
        )

    def run(self, online_status):
        """Run estimation for all currently open outages."""
        with get_conn(self.db_path) as conn:
            open_outages = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM outages WHERE resolved_date IS NULL"
                ).fetchall()
            ]

        for outage in open_outages:
            self.estimate_for_outage(outage, online_status)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
sudo -u snotel .venv/bin/pytest tests/test_estimator.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Fix ownership and commit**

```bash
sudo chown snotel:snotel /home/snotel/htdocs/app/services/estimator.py \
  /home/snotel/htdocs/app/tests/test_estimator.py
sudo -u snotel git add services/estimator.py tests/test_estimator.py
sudo -u snotel git commit -m "feat: add Estimator — daily gap-filling using best available ranked model"
```

---

## Task 5: DailyJob Orchestrator (`services/daily_job.py`)

**Files:**
- Create: `services/daily_job.py`
- Create: `logs/` directory (for log output)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_daily_job.py
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.db import create_tables
from services.daily_job import DailyJob


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    create_tables(p)
    return p


@pytest.fixture
def meta_db_path(tmp_path):
    """Minimal meta.db with two stations."""
    import sqlite3
    import pandas as pd
    p = tmp_path / "meta.db"
    df = pd.DataFrame([
        {"triplet": "301:CA:SNTL", "name": "Adin Mtn",
         "latitude": 41.2, "longitude": -120.9},
        {"triplet": "391:CA:SNTL", "name": "Blue Canyon",
         "latitude": 39.2, "longitude": -120.7},
    ])
    with sqlite3.connect(str(p)) as conn:
        df.to_sql("meta", conn, if_exists="replace", index=False)
    return p


def test_daily_job_runs_three_steps(db_path, meta_db_path):
    """DailyJob.run() must call detector.run, trainer.run, and estimator.run."""
    job = DailyJob(db_path=db_path, meta_db_path=meta_db_path)

    with patch("services.daily_job.OutageDetector") as MockDetector, \
         patch("services.daily_job.AutoModelTrainer") as MockTrainer, \
         patch("services.daily_job.Estimator") as MockEstimator:

        mock_detector = MagicMock()
        mock_detector.build_online_status.return_value = {}
        mock_detector.run.return_value = ([], [])
        MockDetector.return_value = mock_detector

        mock_trainer = MagicMock()
        MockTrainer.return_value = mock_trainer

        mock_estimator = MagicMock()
        MockEstimator.return_value = mock_estimator

        job.run()

    mock_detector.build_online_status.assert_called_once()
    mock_detector.run.assert_called_once()
    mock_trainer.run.assert_called_once()
    mock_estimator.run.assert_called_once()


def test_daily_job_passes_online_status_to_all_steps(db_path, meta_db_path):
    """The same online_status dict must be passed to detector.run, trainer.run, estimator.run."""
    job = DailyJob(db_path=db_path, meta_db_path=meta_db_path)
    sentinel_status = {"301:CA:SNTL": {"WTEQ": None}}

    with patch("services.daily_job.OutageDetector") as MockDetector, \
         patch("services.daily_job.AutoModelTrainer") as MockTrainer, \
         patch("services.daily_job.Estimator") as MockEstimator:

        mock_detector = MagicMock()
        mock_detector.build_online_status.return_value = sentinel_status
        mock_detector.run.return_value = ([], [])
        MockDetector.return_value = mock_detector

        MockTrainer.return_value = MagicMock()
        MockEstimator.return_value = MagicMock()

        job.run()

    mock_detector.run.assert_called_once_with(sentinel_status)
    MockTrainer.return_value.run.assert_called_once_with([], sentinel_status)
    MockEstimator.return_value.run.assert_called_once_with(sentinel_status)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
sudo -u snotel .venv/bin/pytest tests/test_daily_job.py -v
```

Expected: `ModuleNotFoundError: No module named 'services.daily_job'`

- [ ] **Step 3: Create logs directory and implement `services/daily_job.py`**

```bash
sudo -u snotel mkdir -p /home/snotel/htdocs/app/logs
```

```python
# services/daily_job.py
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from services.db import create_tables, MODELS_DB_PATH
from services.outage_detector import OutageDetector
from services.model_trainer import AutoModelTrainer
from services.estimator import Estimator

APP_DIR = Path(__file__).parent.parent
META_DB_PATH = APP_DIR / "dbs" / "meta.db"
LOG_PATH = APP_DIR / "logs" / "daily_job.log"

logger = logging.getLogger(__name__)


class DailyJob:
    def __init__(self, db_path=None, meta_db_path=None):
        self.db_path = db_path or MODELS_DB_PATH
        self.meta_db_path = meta_db_path or META_DB_PATH

    def _load_meta(self):
        with sqlite3.connect(str(self.meta_db_path)) as conn:
            return pd.read_sql("SELECT * FROM meta", conn)

    def run(self):
        create_tables(self.db_path)
        meta_df = self._load_meta()

        logger.info("Step 1: Building online status and detecting outages...")
        detector = OutageDetector(self.db_path, meta_df)
        online_status = detector.build_online_status()
        new_outages, resolved = detector.run(online_status)
        logger.info(f"  New outages: {len(new_outages)}, Resolved: {len(resolved)}")

        logger.info("Step 2: Training models for new outages...")
        trainer = AutoModelTrainer(self.db_path, meta_df)
        trainer.run(new_outages, online_status)

        logger.info("Step 3: Generating daily estimates...")
        estimator = Estimator(self.db_path)
        estimator.run(online_status)

        logger.info("Daily job complete.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_PATH)),
            logging.StreamHandler(sys.stdout),
        ],
    )
    DailyJob().run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
sudo -u snotel .venv/bin/pytest tests/test_daily_job.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Fix ownership and commit**

```bash
sudo chown -R snotel:snotel /home/snotel/htdocs/app/services/daily_job.py \
  /home/snotel/htdocs/app/logs \
  /home/snotel/htdocs/app/tests/test_daily_job.py
sudo -u snotel git add services/daily_job.py tests/test_daily_job.py logs/.gitkeep
sudo -u snotel git commit -m "feat: add DailyJob orchestrator — coordinates outage detection, model training, and estimation"
```

---

## Task 6: API Blueprint (`api/v1/estimates.py`)

**Files:**
- Create: `api/__init__.py`
- Create: `api/v1/__init__.py`
- Create: `api/v1/estimates.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

```python
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

    import dbs
    from api.v1.estimates import estimates_bp
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
sudo -u snotel .venv/bin/pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Create package markers and implement the blueprint**

```python
# api/__init__.py
```

```python
# api/v1/__init__.py
```

```python
# api/v1/estimates.py
import hashlib
from datetime import date
from pathlib import Path

from flask import Blueprint, jsonify, request

from services.db import get_conn

APP_DIR = Path(__file__).parent.parent.parent
DB_PATH = APP_DIR / "dbs" / "regr_models.db"

estimates_bp = Blueprint("estimates_v1", __name__, url_prefix="/api/v1")

UNITS = {"WTEQ": "inches", "PREC": "inches"}


def _require_api_key(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key:
            return jsonify({"error": "Missing X-API-Key header"}), 401
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        with get_conn(DB_PATH) as conn:
            row = conn.execute(
                "SELECT id FROM api_keys WHERE key_hash=? AND active=1",
                (key_hash,),
            ).fetchone()
        if not row:
            return jsonify({"error": "Invalid or inactive API key"}), 401
        return f(*args, **kwargs)

    return decorated


@estimates_bp.route("/estimates/<triplet>/<parameter>")
@_require_api_key
def get_estimates(triplet, parameter):
    start = request.args.get("start")
    end = request.args.get("end")

    query = (
        "SELECT e.date, e.estimated_value, am.rank "
        "FROM estimates e "
        "JOIN auto_models am ON e.auto_model_id = am.id "
        "WHERE e.triplet=? AND e.parameter=?"
    )
    params = [triplet, parameter]
    if start:
        query += " AND e.date >= ?"
        params.append(start)
    if end:
        query += " AND e.date <= ?"
        params.append(end)
    query += " ORDER BY e.date"

    with get_conn(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()

    return jsonify({
        "station": triplet,
        "parameter": parameter,
        "unit": UNITS.get(parameter, "unknown"),
        "estimates": [
            {
                "date": r["date"],
                "estimated_value": r["estimated_value"],
                "model_rank": r["rank"],
            }
            for r in rows
        ],
    })


@estimates_bp.route("/outages")
@_require_api_key
def get_outages():
    parameter = request.args.get("parameter")
    query = "SELECT * FROM outages WHERE resolved_date IS NULL"
    params = []
    if parameter:
        query += " AND parameter=?"
        params.append(parameter)
    with get_conn(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()
    return jsonify({"outages": [dict(r) for r in rows]})


@estimates_bp.route("/outages/<triplet>")
@_require_api_key
def get_station_outages(triplet):
    with get_conn(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT * FROM outages WHERE triplet=? ORDER BY detected_date DESC",
            (triplet,),
        ).fetchall()
    return jsonify({"station": triplet, "outages": [dict(r) for r in rows]})


@estimates_bp.route("/status")
@_require_api_key
def get_status():
    with get_conn(DB_PATH) as conn:
        row = conn.execute(
            "SELECT MAX(generated_date) AS last_run FROM estimates"
        ).fetchone()
    return jsonify({
        "status": "ok",
        "last_job_run": row["last_run"] if row else None,
        "current_date": date.today().isoformat(),
    })
```

- [ ] **Step 4: Run test to verify it passes**

```bash
sudo -u snotel .venv/bin/pytest tests/test_api.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Fix ownership and commit**

```bash
sudo chown -R snotel:snotel /home/snotel/htdocs/app/api \
  /home/snotel/htdocs/app/tests/test_api.py
sudo -u snotel git add api/ tests/test_api.py
sudo -u snotel git commit -m "feat: add /api/v1/ Flask blueprint with estimates, outages, and status endpoints behind API key auth"
```

---

## Task 7: Register Blueprint in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add blueprint registration and table init to `app.py`**

Open `app.py`. After line `server = app.server`, add:

```python
from api.v1.estimates import estimates_bp
from services.db import create_tables as _create_auto_tables

server.register_blueprint(estimates_bp)
_create_auto_tables()
```

The top of `app.py` should look like:

```python
app = dbs.app
server = app.server

from api.v1.estimates import estimates_bp
from services.db import create_tables as _create_auto_tables

server.register_blueprint(estimates_bp)
_create_auto_tables()
```

- [ ] **Step 2: Restart the service and verify the blueprint is live**

```bash
sudo systemctl restart snotel-regression.service && sleep 3
sudo systemctl status snotel-regression.service --no-pager | tail -5
```

Expected: `Active: active (running)`

- [ ] **Step 3: Smoke-test that the status endpoint is reachable (no key → 401)**

```bash
curl -s -o /dev/null -w "%{http_code}" https://snotel-regression-tool.streamflows.org/api/v1/status
```

Expected: `401`

- [ ] **Step 4: Fix ownership and commit**

```bash
sudo chown snotel:snotel /home/snotel/htdocs/app/app.py
sudo -u snotel git add app.py
sudo -u snotel git commit -m "feat: register /api/v1/ blueprint and init auto-estimation tables at app startup"
```

---

## Task 8: API Key Management CLI (`scripts/manage_api_keys.py`)

**Files:**
- Create: `scripts/manage_api_keys.py`

- [ ] **Step 1: Implement the script**

```python
#!/usr/bin/env python
# scripts/manage_api_keys.py
"""CLI for API key generation, listing, and deactivation."""
import hashlib
import secrets
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.db import create_tables, get_conn


def generate():
    label = input("Label for this key (e.g. 'nick-personal'): ").strip()
    if not label:
        print("Label cannot be empty.")
        return
    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    create_tables()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, label, created_date, active) VALUES (?,?,?,1)",
            (key_hash, label, date.today().isoformat()),
        )
    print(f"\nAPI Key — save this, it will not be shown again:\n  {raw_key}\n")
    print(f"Label: {label}")


def list_keys():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, label, created_date, active FROM api_keys ORDER BY id"
        ).fetchall()
    if not rows:
        print("No API keys found.")
        return
    print(f"{'ID':<5} {'Active':<8} {'Created':<12} Label")
    print("-" * 55)
    for r in rows:
        print(f"{r['id']:<5} {'Yes' if r['active'] else 'No':<8} {r['created_date']:<12} {r['label']}")


def deactivate():
    list_keys()
    try:
        key_id = int(input("\nEnter ID to deactivate: "))
    except ValueError:
        print("Invalid ID.")
        return
    with get_conn() as conn:
        conn.execute("UPDATE api_keys SET active=0 WHERE id=?", (key_id,))
    print(f"Key {key_id} deactivated.")


COMMANDS = {"generate": generate, "list": list_keys, "deactivate": deactivate}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python scripts/manage_api_keys.py [{'/'.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
```

- [ ] **Step 2: Make executable and fix ownership**

```bash
sudo chmod +x /home/snotel/htdocs/app/scripts/manage_api_keys.py
sudo chown snotel:snotel /home/snotel/htdocs/app/scripts/manage_api_keys.py
```

- [ ] **Step 3: Generate your first API key**

```bash
sudo -u snotel /home/snotel/htdocs/app/.venv/bin/python \
  /home/snotel/htdocs/app/scripts/manage_api_keys.py generate
```

Enter a label when prompted (e.g. `nick-personal`). **Save the printed key** — it is not stored in plain text.

- [ ] **Step 4: Verify the key works against the live API**

```bash
curl -s -H "X-API-Key: <your-key>" \
  https://snotel-regression-tool.streamflows.org/api/v1/status | python3 -m json.tool
```

Expected:
```json
{
    "status": "ok",
    "last_job_run": null,
    "current_date": "2026-06-12"
}
```

- [ ] **Step 5: Commit**

```bash
sudo -u snotel git add scripts/manage_api_keys.py
sudo -u snotel git commit -m "feat: add manage_api_keys.py CLI for API key generation, listing, and deactivation"
```

---

## Task 9: Systemd Timer Deployment

**Files:**
- Create: `/etc/systemd/system/snotel-daily-job.timer`
- Create: `/etc/systemd/system/snotel-daily-job.service`

- [ ] **Step 1: Create the service unit**

```bash
sudo tee /etc/systemd/system/snotel-daily-job.service > /dev/null << 'EOF'
[Unit]
Description=SNOTEL Auto-Estimation Daily Job
After=network.target

[Service]
Type=oneshot
User=snotel
Group=snotel
WorkingDirectory=/home/snotel/htdocs/app
ExecStart=/home/snotel/htdocs/app/.venv/bin/python services/daily_job.py
StandardOutput=journal
StandardError=journal
EOF
```

- [ ] **Step 2: Create the timer unit**

```bash
sudo tee /etc/systemd/system/snotel-daily-job.timer > /dev/null << 'EOF'
[Unit]
Description=SNOTEL Auto-Estimation Daily Job Timer
Requires=snotel-daily-job.service

[Timer]
OnCalendar=*-*-* 15:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF
```

- [ ] **Step 3: Enable and start the timer**

```bash
sudo systemctl daemon-reload
sudo systemctl enable snotel-daily-job.timer
sudo systemctl start snotel-daily-job.timer
sudo systemctl status snotel-daily-job.timer --no-pager
```

Expected: `Active: active (waiting)` and `Trigger: <next 15:30 UTC>`

- [ ] **Step 4: Run the job once manually to verify end-to-end**

```bash
sudo systemctl start snotel-daily-job.service
sudo journalctl -u snotel-daily-job.service --no-pager | tail -20
```

Expected: Log lines showing Steps 1–3 completing, `Daily job complete.`

- [ ] **Step 5: Commit systemd unit files**

```bash
sudo -u snotel mkdir -p /home/snotel/htdocs/app/deploy
sudo tee /home/snotel/htdocs/app/deploy/snotel-daily-job.service > /dev/null << 'EOF'
[Unit]
Description=SNOTEL Auto-Estimation Daily Job
After=network.target

[Service]
Type=oneshot
User=snotel
Group=snotel
WorkingDirectory=/home/snotel/htdocs/app
ExecStart=/home/snotel/htdocs/app/.venv/bin/python services/daily_job.py
StandardOutput=journal
StandardError=journal
EOF

sudo tee /home/snotel/htdocs/app/deploy/snotel-daily-job.timer > /dev/null << 'EOF'
[Unit]
Description=SNOTEL Auto-Estimation Daily Job Timer
Requires=snotel-daily-job.service

[Timer]
OnCalendar=*-*-* 15:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo chown -R snotel:snotel /home/snotel/htdocs/app/deploy
sudo -u snotel git add deploy/
sudo -u snotel git commit -m "deploy: add systemd timer and service units for daily auto-estimation job at 15:30 UTC"
```

---

## Final Verification

- [ ] **Run the full test suite**

```bash
sudo -u snotel .venv/bin/pytest tests/test_db.py tests/test_outage_detector.py \
  tests/test_estimator.py tests/test_daily_job.py tests/test_api.py -v
```

Expected: All tests pass. `test_model_trainer.py` integration tests require network and make real AWDB calls — run separately.

- [ ] **Confirm timer is scheduled**

```bash
sudo systemctl list-timers snotel-daily-job.timer --no-pager
```

Expected: Shows next trigger at 15:30:00 UTC.

- [ ] **Confirm API key auth works end-to-end**

```bash
curl -s -H "X-API-Key: <your-key>" \
  "https://snotel-regression-tool.streamflows.org/api/v1/outages" | python3 -m json.tool
```

Expected: `{"outages": []}` (empty until the timer first fires and detects outages)
