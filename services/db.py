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
