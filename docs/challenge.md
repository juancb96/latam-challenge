# Challenge Design Decisions

## Part I — Model Selection

### Business Constraint

False Negatives (predict on-time, flight delays) cascade into unplanned gate reassignments, crew reallocation, and missed passenger notifications. False Positives (predict delay, flight on-time) cause recoverable prep overhead. **Objective: maximize recall on class 1 (delay).**

### Dataset

68,206 flights from SCL. Class distribution: 81.6% on-time (class 0), 18.4% delayed (class 1). Scale factor `n_y0/n_y1 ≈ 4.44`.

### Candidates Evaluated

All models trained on 67% / tested on 33% split (`random_state=42`).

| Model | Features | Balance | Class 1 Recall | Class 1 F1 | Class 0 Recall | Class 0 F1 |
|---|---|---|---|---|---|---|
| XGBoost | All (37) | None | 0.00 | 0.00 | 1.00 | 0.90 |
| LogisticRegression | All (37) | None | 0.03 | 0.06 | 0.99 | 0.90 |
| XGBoost | Top 10 | `scale_pos_weight=4.44` | **0.69** | **0.37** | 0.52 | 0.66 |
| XGBoost | Top 10 | None | 0.01 | 0.01 | 1.00 | 0.90 |
| LogisticRegression | Top 10 | `class_weight={1:0.816, 0:0.184}` | **0.69** | **0.36** | 0.52 | 0.65 |
| LogisticRegression | Top 10 | None | 0.01 | 0.03 | 1.00 | 0.90 |

**Test thresholds:** class 0 recall > 0.60, class 0 f1 > 0.70 · class 1 recall > 0.60, class 1 f1 > 0.30

Only two models pass all thresholds: XGBoost (top 10 + balance) and LogisticRegression (top 10 + balance).

### Decision: Logistic Regression with top 10 features and `class_weight={1: n_y0/n, 0: n_y1/n}`

Both balanced models achieve identical class 1 recall (0.69) — the primary business metric. The F1 delta (0.37 vs 0.36) is within run-to-run noise. Logistic Regression is preferred: simpler inference, smaller memory footprint, and directly interpretable coefficients. No complexity added without meaningful performance gain.

### Top 10 Features

Derived from XGBoost feature importance on the full feature set:

```python
FEATURES_COLS = [
    "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM",
    "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
]
```

Airline identity (OPERA) and month (MES) dominate. International flight type (TIPOVUELO_I) contributes. Reducing from 37 to 10 features does not degrade performance and improves inference speed.

## Part II — API

FastAPI app exposing two endpoints:

- `GET /health` — liveness check, returns `{"status": "OK"}`
- `POST /predict` — accepts a list of flights, returns delay predictions

Request/response shape:

```json
// request
{"flights": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3}]}
// response 200
{"predict": [0]}
// response 400 — invalid MES (not 1–12) or invalid TIPOVUELO (not I/N)
// response 403 — missing or wrong X-Api-Key (production only)
```

Input validation is handled at the route level before reaching the model. API key auth is env-gated via `API_KEY`: if unset (local dev, CI), the check is skipped entirely so tests pass without credentials.

## Part III — Cloud Deployment

### Architecture

Deployed on **GCP Cloud Run** — fully managed serverless containers. No VM provisioning, scales to zero (no idle costs). Image stored in Artifact Registry; Cloud Run pulls at deploy time.

Multi-stage Dockerfile: a `trainer` stage runs `train_model.py` at build time (bakes `model.pkl` into the image); the `runtime` stage copies the artifact and serves via `uvicorn`. Training data never enters the runtime image.

Model serialized with `joblib` (sklearn-recommended convention; efficient binary format for numpy arrays).

### Cost Protection

Three layers prevent runaway spend:

1. **Max instances cap** — `--max-instances 1` on Cloud Run. Single container, no horizontal scale-out cost.
2. **API key auth** — `X-Api-Key` header required on `/predict` in production. Random traffic gets 403.
3. **GCP billing budget** — $10 cap with alerts at 50%/90%/100%. Optionally link a budget action to disable billing automatically.

### Deployment Config

All values live in `.env` (gitignored). See `.env.example` for required variables.

### One-Time Setup

```bash
gcloud auth login
gcloud config set project ${GCP_PROJECT_ID}

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

gcloud artifacts repositories create ${AR_REPO} \
  --repository-format=docker \
  --location=${GCP_REGION}
```

### Build and Deploy

```bash
# Build image on GCP (avoids ARM/amd64 cross-compilation issues on Apple Silicon)
gcloud builds submit \
  --tag ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest \
  --project ${GCP_PROJECT_ID}

# Deploy to Cloud Run
gcloud run deploy ${CLOUD_RUN_SERVICE} \
  --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest \
  --platform managed \
  --region ${GCP_REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 1 \
  --concurrency 80 \
  --set-env-vars API_KEY=${API_KEY}
```

`gcloud builds submit` is used instead of local `docker build` to build on GCP's native linux/amd64 — avoids exec format errors when deploying from Apple Silicon (ARM).

### Verify

```bash
curl https://latam-challenge-api-719722839892.us-central1.run.app/health
# {"status":"OK"}

curl -X POST https://latam-challenge-api-719722839892.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: <api_key>" \
  -d '{"flights":[{"OPERA":"Grupo LATAM","TIPOVUELO":"N","MES":3}]}'
# {"predict":[0]}
```

## Part IV — CI/CD

GitHub Actions workflows in `.github/workflows/`:

### CI (`ci.yml`)

Triggers on push to any non-`main` branch and on PRs to `main`. Runs:

1. `make model-test` — unit tests for `DelayModel`
2. `make api-test` — integration tests via FastAPI `TestClient`

### CD (`cd.yml`)

Triggers on push to `main`. Steps:

1. Authenticate to GCP via service account key (`GCP_SA_KEY` secret)
2. Submit build to Cloud Build (`--async`) — builds and pushes the image to Artifact Registry tagged with `github.sha`
3. Poll `gcloud builds describe` until `SUCCESS` or terminal failure state
4. Deploy new image to Cloud Run with `gcloud run deploy`

`--async` + polling is used instead of streaming logs because the service account does not need GCS log bucket read access — fewer required IAM roles, cleaner permission scope.

### GitHub Config

| Type | Name | Purpose |
|---|---|---|
| Secret | `GCP_SA_KEY` | GCP service account JSON key |
| Secret | `API_KEY` | Endpoint auth key injected into Cloud Run |
| Variable | `GCP_PROJECT_ID` | GCP project ID |
| Variable | `GCP_REGION` | Region for Artifact Registry + Cloud Run |
| Variable | `AR_REPO` | Artifact Registry repository name |
| Variable | `IMAGE_NAME` | Docker image name |
| Variable | `CLOUD_RUN_SERVICE` | Cloud Run service name |
