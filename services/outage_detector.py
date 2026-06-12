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
        """Returns 'null_value', 'stale', or None (online)."""
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
        """Check WTEQ, PREC, and SNWD for all stations.
        Returns {triplet: {param: None|'null_value'|'stale'}}. None means online."""
        status = {}
        for _, row in self.meta_df.iterrows():
            triplet = row["triplet"]
            status[triplet] = {}
            for param in CHECK_PARAMS:
                status[triplet][param] = self.check_station_param(triplet, param)
        return status

    def run(self, online_status):
        """Reconcile online_status against the outages table.
        Returns (new_outages, resolved_outages) as lists of dicts."""
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
