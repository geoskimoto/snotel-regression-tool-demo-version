# Auto-Estimation System Design
**Date:** 2026-06-12  
**Project:** SNOTEL Regression Tool  
**Author:** Nick Steele

---

## Overview

Extend the SNOTEL Regression Tool with an automated gap-filling system. When a SNOTEL station's WTEQ or PREC sensor goes down, the system detects the outage, trains regression models on-demand using nearby and same-station sensors, generates daily estimates, and serves them via a versioned REST API.

---

## Architecture

**Option B — Separate background service + shared DB:**

- A standalone `services/daily_job.py` runs as a systemd timer owned by `snotel`
- It owns outage detection, model training, and daily estimation
- The existing Dash/Flask app gains a new `/api/v1/` Flask blueprint for serving estimates
- Both share `regr_models.db` (SQLite)
- The web process never blocks on training

```
[systemd timer @ 15:30 UTC daily]
        |
        v
services/daily_job.py
  ├── Step 1: OutageDetector     → writes to `outages` table
  ├── Step 2: ModelTrainer       → writes to `auto_models` table (new outages only)
  └── Step 3: Estimator          → writes to `estimates` table

[Existing Dash/Flask app]
  └── api/v1/ blueprint          → reads `outages`, `auto_models`, `estimates`
```

---

## Scope

- **Parameters tracked:** `WTEQ` and `PREC` only
- **Stations:** All stations in `meta.db` (~148 SNTL/SNTLT stations)
- **Models per outage:** 3 ranked models, trained on-demand at outage detection
- **Retraining:** Annually on October 1 (start of water year)
- **File ownership:** All files and cron entries owned by `snotel:snotel`

---

## Database Schema

All new tables added to `regr_models.db`.

### `outages`
Tracks detected sensor failures.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | auto-increment |
| `triplet` | VARCHAR(50) | e.g. `301:CA:SNTL` |
| `parameter` | VARCHAR(20) | `WTEQ` or `PREC` |
| `detected_date` | DATE | date first flagged |
| `resolved_date` | DATE | NULL if still active |
| `detection_method` | VARCHAR(20) | `null_value` or `stale` |

### `auto_models`
On-demand trained models for active outages.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `outage_id` | INTEGER FK | → `outages.id` |
| `rank` | INTEGER | 1 (best) to 3 |
| `stationparameters` | VARCHAR | predictor set string, same format as existing `regr_models` |
| `regressor_type` | VARCHAR(50) | e.g. `Lasso` |
| `RMSE_train` | FLOAT | |
| `RMSE_test` | FLOAT | |
| `model` | BLOB | pickled model object |
| `trained_date` | DATE | |
| `water_year` | INTEGER | e.g. `2026`, for annual retraining |

### `estimates`
Daily gap-filled values.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `outage_id` | INTEGER FK | → `outages.id` |
| `auto_model_id` | INTEGER FK | → `auto_models.id` — which model produced this |
| `triplet` | VARCHAR(50) | |
| `parameter` | VARCHAR(20) | |
| `date` | DATE | date of the estimated value |
| `estimated_value` | FLOAT | |
| `generated_date` | DATE | when the estimate was computed |

### `api_keys`
API authentication.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `key_hash` | VARCHAR(64) | SHA-256 hash of the key |
| `label` | VARCHAR(100) | human-readable owner label |
| `created_date` | DATE | |
| `active` | BOOLEAN | soft-disable without deleting |

---

## Background Service (`services/daily_job.py`)

Runs as a systemd timer at **15:30 UTC daily** (8:30 AM PDT / 7:30 AM PST). Chosen to avoid known busy windows: 02:00–03:00 UTC (backups/clp-update), 12:00–13:00 UTC (streamflow synopsis), 14:00–14:30 UTC (itemfinder + snow elevation), 23:00 UTC (forecast timers). AWDB data is reliably available by this time.

### Step 1 — Outage Detection

For each station/parameter combination (`WTEQ`, `PREC`) in `meta.db`:
1. Fetch the last 5 days of values from the AWDB API
2. Flag as down if:
   - Latest value is `null`, OR
   - No new value reported in the last 3 days (staleness)
3. Cross-reference against open `outages` records:
   - New outage → insert row with `detected_date = today`
   - Previously open, now reporting → stamp `resolved_date = today`
   - Already open → no action

### Step 2 — Model Training (new outages only)

For each newly inserted outage, build and rank 3 predictor candidate sets:

**Predictor candidate priority:**
1. Same-station sensors currently online (highest priority — e.g., SNWD at the failed station when WTEQ is down; WTEQ when PREC is down)
2. Nearest stations (by lat/lon proximity from `meta.db`) with the same parameter online
3. Nearest stations with correlated parameters online

For each of the top 3 candidate sets:
- Reuse existing `RegressionFun` with `Lasso` as the default regressor (good regularization for correlated predictors)
- Train on the most recent full water year of data (chronological split, no shuffling)
- Rank by `RMSE_test` ascending (lower is better → rank 1)
- Store to `auto_models` with `water_year = current_water_year`

### Step 3 — Daily Estimation (all open outages)

For each open outage:
1. Walk `auto_models` for that outage ordered by `rank`
2. Select the first model whose predictor stations are all currently online (check today's AWDB fetch from Step 1)
3. Run prediction for today using `RegressionFun.make_predictions()`
4. Insert result into `estimates`
5. If no model has all predictors online: log a warning, skip — gap remains unfilled for that day

### Annual Retraining (Oct 1 UTC)

For all open outages, retrain all 3 models incorporating the full previous water year. Replace existing `auto_models` records for that outage (update `trained_date`, `water_year`, `model` blob, RMSE values).

---

## API (`api/v1/` Flask Blueprint)

Mounted on the existing Flask app. All endpoints require `X-API-Key` header. Invalid or missing key returns `401`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/estimates/<triplet>/<parameter>` | Gap-filled estimates. Query params: `start`, `end` (YYYY-MM-DD, optional) |
| `GET` | `/api/v1/outages` | All open outages. Optional: `?parameter=WTEQ` |
| `GET` | `/api/v1/outages/<triplet>` | Outage status for a specific station |
| `GET` | `/api/v1/status` | Health check — service up, last job run time |

### Response Envelope

```json
{
  "station": "301:CA:SNTL",
  "parameter": "WTEQ",
  "unit": "inches",
  "estimates": [
    {"date": "2026-06-11", "estimated_value": 12.3, "model_rank": 1}
  ]
}
```

### Auth

- API keys stored as SHA-256 hashes in `api_keys` table
- Key passed as `X-API-Key` request header
- A management script (`scripts/manage_api_keys.py`) handles key generation, listing, and deactivation
- No rate limiting initially; add when/if the API opens publicly

### Versioning

`/v1/` prefix in all paths allows future `/v2/` without breaking existing consumers.

---

## File Structure

```
/home/snotel/htdocs/app/
├── services/
│   ├── daily_job.py          # orchestrates all 3 steps
│   ├── outage_detector.py    # Step 1 logic
│   ├── model_trainer.py      # Step 2 logic
│   └── estimator.py          # Step 3 logic
├── api/
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py       # blueprint registration
│       └── estimates.py      # endpoint handlers
├── scripts/
│   └── manage_api_keys.py    # CLI key management
└── dbs/
    └── __init__.py           # new SQLAlchemy models added here
```

---

## Deployment

### Systemd Timer (`/etc/systemd/system/snotel-daily-job.timer`)
```ini
[Timer]
OnCalendar=*-*-* 15:30:00 UTC
Persistent=true
```

### Systemd Service (`/etc/systemd/system/snotel-daily-job.service`)
```ini
[Service]
User=snotel
Group=snotel
WorkingDirectory=/home/snotel/htdocs/app
ExecStart=/home/snotel/htdocs/app/.venv/bin/python services/daily_job.py
```

All files created under `/home/snotel/htdocs/app/` owned by `snotel:snotel`.

---

## Testing

- **Unit:** `OutageDetector`, `ModelTrainer`, `Estimator` tested independently with mock AWDB responses
- **Integration:** End-to-end job run against a test SQLite DB with known station data
- **API:** Endpoint tests covering valid keys, invalid keys, missing stations, empty date ranges
- **Model quality:** Assert `RMSE_test` is finite and non-negative for all trained models
- **Temporal integrity:** Assert no shuffling occurs in train/test splits

---

## Out of Scope

- Parameters other than `WTEQ` and `PREC`
- Push notifications when outages are detected
- A UI within the Dash app for browsing auto-estimates (can be added later)
- Rate limiting (deferred until API opens publicly)
