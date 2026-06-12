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
