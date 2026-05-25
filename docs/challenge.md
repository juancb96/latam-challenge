# Challenge Design Decisions

## Model Selection

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

**Test thresholds:** class 0 recall < 0.60, class 0 f1 < 0.70 · class 1 recall > 0.60, class 1 f1 > 0.30

Only two models pass all thresholds: XGBoost (top 10 + balance) and LogisticRegression (top 10 + balance).

### Decision: Logistic Regression with top 10 features and `class_weight={1: n_y0/n, 0: n_y1/n}`

Both balanced models achieve identical class 1 recall (0.69) — the primary business metric. The F1 delta (0.37 vs 0.36) is within run-to-run noise and not meaningful. Logistic Regression is preferred: simpler inference, smaller memory footprint, and directly interpretable coefficients. Occam's razor applies — no complexity should be added without meaningful performance gain.

### Top 10 Features

Derived from XGBoost feature importance on the full feature set:

```python
FEATURES_COLS = [
    "OPERA_Latin American Wings", "MES_7", "MES_10", "OPERA_Grupo LATAM",
    "MES_12", "TIPOVUELO_I", "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
]
```

Airline identity (OPERA) and month (MES) dominate. International flight type (TIPOVUELO_I) contributes. Reducing from 37 to 10 features does not degrade performance and improves inference speed.

## Model Serialization

The trained model is serialized with `joblib` rather than `pickle`. Both work, but `joblib` is the sklearn-recommended convention because it uses compressed numpy binary format under the hood — more efficient for the numpy arrays (coefficients, intercepts) stored inside fitted sklearn estimators. For a `LogisticRegression` on 10 features the size difference is negligible, but `joblib` is used as the idiomatic choice with zero extra dependency cost (it ships with `scikit-learn`).

The serialization lifecycle:

1. `train_model.py` — loads `data/data.csv`, runs `preprocess` + `fit`, saves `challenge/model.pkl`
2. `Dockerfile` (trainer stage) — runs `train_model.py` at build time; `data/data.csv` never enters the runtime image
3. `challenge/api.py` — loads `model.pkl` at server startup via `@app.on_event("startup")`

`challenge/model.pkl` is excluded from version control (`.gitignore`). It is a build artifact, not source.

## GCP Deployment

### Architecture

Cloud Run is a fully managed serverless container platform — no VM provisioning, no Kubernetes. It scales to zero (no idle costs) and scales out under load automatically. The container image is stored in Artifact Registry and pulled by Cloud Run at deploy time.

### Prerequisites

1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
2. Install [Docker](https://docs.docker.com/get-docker/) (or Colima on macOS: `brew install colima docker`)
3. Create a GCP project at https://console.cloud.google.com

### One-Time Setup

```bash
# Authenticate
gcloud auth login
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Set project
gcloud config set project ${GCP_PROJECT_ID}

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create ${AR_REPO} \
  --repository-format=docker \
  --location=${GCP_REGION}
```

### Build and Push

```bash
# Load env vars
source .env

# Build image (multi-stage: trainer → runtime)
docker build \
  -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest \
  .

# Push to Artifact Registry
docker push \
  ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest
```

### Deploy to Cloud Run

```bash
gcloud run deploy ${CLOUD_RUN_SERVICE} \
  --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:latest \
  --platform managed \
  --region ${GCP_REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1
```

The command outputs the service URL. Update `STRESS_URL` in `Makefile` line 26 with that URL, then run `make stress-test`.

### Verify Deployment

```bash
curl https://<service-url>/health
# {"status":"OK"}

curl -X POST https://<service-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"flights":[{"OPERA":"Grupo LATAM","TIPOVUELO":"N","MES":3}]}'
# {"predict":[0]}
```

### Environment Variables

All GCP-specific values live in `.env` (gitignored). `.env.example` documents required variables:

| Variable | Description |
|---|---|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_REGION` | Region for Artifact Registry + Cloud Run (e.g. `us-central1`) |
| `AR_REPO` | Artifact Registry repository name |
| `IMAGE_NAME` | Docker image name |
| `CLOUD_RUN_SERVICE` | Cloud Run service name |
