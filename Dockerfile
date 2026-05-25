# syntax=docker/dockerfile:1.2

# Stage 1 — base: shared Python + dependencies (reused by both stages below)
FROM python:3.9-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 — trainer: has data + code, trains model, serializes to model.pkl
#            data/data.csv never makes it past this stage
FROM base AS trainer
COPY challenge/ ./challenge/
COPY data/ ./data/
COPY train_model.py .
RUN python train_model.py

# Stage 3 — runtime: lean image, no data, no train script
#            only gets model.pkl artifact from trainer via COPY --from
FROM base AS runtime
COPY challenge/ ./challenge/
COPY --from=trainer /app/challenge/model.pkl ./challenge/model.pkl
EXPOSE 8080
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
