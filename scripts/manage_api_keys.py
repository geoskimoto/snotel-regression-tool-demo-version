#!/usr/bin/env python
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
