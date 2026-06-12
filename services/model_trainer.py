# services/model_trainer.py
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
        """Return up to 3 predictor candidate sets, prioritized by sensor availability."""
        correlates = CORRELATED_PARAMS.get(parameter, [])
        nearby = self._get_nearby_triplets(triplet, n=5)

        pool = []
        for p in correlates:
            if online_status.get(triplet, {}).get(p) is None:
                pool.append((triplet, p))
        for t in nearby:
            if online_status.get(t, {}).get(parameter) is None:
                pool.append((t, parameter))
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
        """Train up to 3 ranked Lasso models for one outage, stored to auto_models."""
        if parameter not in CORRELATED_PARAMS:
            logger.warning(f"Unsupported parameter {parameter} — no correlated params defined")
            return
        candidates = self.build_predictor_candidates(triplet, parameter, online_status)
        if not candidates:
            logger.warning(f"No predictor candidates available for {triplet}/{parameter}")
            return

        today = date.today()
        if today.month >= 10:
            wy_end_year = today.year
        else:
            wy_end_year = today.year - 1
        train_start = date(wy_end_year - 1, 10, 1).isoformat()
        train_end = date(wy_end_year, 9, 30).isoformat()
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
                logger.warning(f"Training failed for candidate {cset}: {e}", exc_info=True)

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
