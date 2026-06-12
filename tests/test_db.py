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
