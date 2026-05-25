# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Challenge Overview

Operationalize a DS flight delay prediction model for SCL airport. Four parts: transcribe notebook → model.py, build FastAPI, deploy to cloud, implement CI/CD.

**Strict rule: do not rename files, folders, or existing method signatures.**

## Commands

```bash
# Setup
make venv          # create .venv
make install       # pip install all requirements

# Tests
make model-test    # pytest tests/model/ with coverage
make api-test      # pytest tests/api/ with coverage
make stress-test   # locust stress test against STRESS_URL

# Single test
pytest tests/model/test_model.py::TestModel::test_model_fit
pytest tests/api/test_api.py::TestBatchPipeline::test_should_get_predict
```

## Architecture

```
challenge/
  __init__.py     # exports `app` from api.py — test suite imports `from challenge import app`
  model.py        # DelayModel class (preprocess, fit, predict)
  api.py          # FastAPI app with GET /health and POST /predict
  exploration.ipynb  # DS source notebook — reference for model logic

data/data.csv       # training data (~68k rows)
tests/
  model/          # unit tests for DelayModel
  api/            # integration tests via FastAPI TestClient
  stress/         # locust stress test
workflows/        # ci.yml + cd.yml templates (must be copied to .github/workflows/)
docs/challenge.md # all explanations and design decisions go here
```

## Key Constraints from Tests

**model.py — `preprocess()` must produce exactly these 10 features:**
```python
FEATURES_COLS = [
    "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM",
    "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
]
```
These are one-hot encoded columns selected from the full feature set.

**model.py — fit() performance thresholds (class imbalance must be handled):**
- class `"0"` recall < 0.60, f1 < 0.70
- class `"1"` recall > 0.60, f1 > 0.30

**api.py — POST /predict request/response shape:**
```json
// request
{"flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]}
// response 200
{"predict": [0]}
// response 400 — invalid MES (not 1-12) or invalid TIPOVUELO (not I/N)
```

**Known bug in model.py stub:** `Union(...)` on line 16 is invalid Python — must be `Union[...]`.

## Model Selection

**Must be decided by analyzing `challenge/exploration.ipynb` before implementing `model.py`.** The notebook trains multiple candidate models — evaluate them and pick the one that best satisfies the business constraint below. Justify in `docs/challenge.md`.

**Business constraint — maximize recall, not precision:**
- False Negative (predict on-time, flight delays): airport/airline is caught off-guard — no gate reassignment, no crew reallocation, no passenger notification. Cascading delays across connecting flights. High operational and reputational cost.
- False Positive (predict delay, flight is on-time): unnecessary prep work. Low cost, recoverable.
- **FN is more damaging → maximize recall on class `"1"` (delay).** Choose the model with highest recall on delayed flights, accepting lower precision. Class imbalance handling (e.g. `scale_pos_weight`, `class_weight`) is likely required to achieve this.

Test thresholds enforce this: class `"1"` recall > 0.60 and f1 > 0.30 must pass.

## Deployment

Deploy to **GCP** (Cloud Run recommended). Update `STRESS_URL` in Makefile line 26 to the deployed GCP service URL. Complete the Dockerfile for containerized deployment.

## CI/CD

**GitHub Actions only** — copy `workflows/` → `.github/workflows/`. Complete:
- `ci.yml`: trigger on push/PR, run `make model-test` + `make api-test`
- `cd.yml`: trigger on merge to `main`, build Docker image, push to GCP (Artifact Registry or GCR), deploy to Cloud Run

## Documentation

All design decisions, model choice rationale, and architecture notes go in `docs/challenge.md`.
