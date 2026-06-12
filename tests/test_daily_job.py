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
