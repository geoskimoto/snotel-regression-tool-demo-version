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
        """Select best available model, predict today, store to estimates."""
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
                online_status.get(t, {}).get(p, "missing") is None
                for t, p in predictor_pairs
            )
            if not all_online:
                continue

            try:
                model_obj = pickle.loads(am["model"])
                predictor = ModelPredictor(model_obj, pairs)
                predictions, _, _, _ = predictor.predict(today, today)
            except Exception as e:
                logger.warning(f"Prediction failed with model {am['id']}: {e}", exc_info=True)
                continue

            if len(predictions) == 0:
                logger.warning(
                    f"No prediction data returned for {outage['triplet']}/{outage['parameter']} "
                    f"using model {am['id']}"
                )
                continue

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

        logger.warning(
            f"No viable model for outage {outage['id']} "
            f"({outage['triplet']}/{outage['parameter']}) — all predictors offline"
        )

    def run(self, online_status):
        """Run estimation for all open outages."""
        with get_conn(self.db_path) as conn:
            open_outages = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM outages WHERE resolved_date IS NULL"
                ).fetchall()
            ]

        for outage in open_outages:
            self.estimate_for_outage(outage, online_status)
