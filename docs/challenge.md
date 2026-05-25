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
