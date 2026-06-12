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
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
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
