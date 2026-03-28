# Numerai Improvement Program

## Goal
Maximize **CORR Sharpe** on the held-out validation eras.
Secondary goal: keep **MMC Sharpe > 0** (we want to be additive to the Numerai Meta Model).

## Current best (rolling back to single-model baseline for speed)
- CORR Sharpe: **1.4786** (commit 085176f)
- MMC Sharpe: 0.896
- Config: medium features (780), LightGBM, n_estimators=5000, lr=0.005, max_depth=7,
  num_leaves=127, colsample_bytree=0.1, subsample=0.8, subsample_freq=1,
  DOWNSAMPLE_EVERY_N_ERAS=2, single target ("target")

BATCH_BASELINE: 1.4786
(The 1.550357 in results.tsv was from a 3-target ensemble that took 60+ min/iter — too slow, rolled back.
Compare all experiments against 1.4786, not 1.550357.)

---

## ✅ Already explored (do NOT retry these)
- [x] small features → medium features (big win)
- [x] rank predictions within era (hurt badly — do not retry)
- [x] colsample_bytree 0.1 → 0.3 (worse)
- [x] n_estimators 2000 → 5000, lr 0.01 → 0.005 (small win)
- [x] subsample=0.8, subsample_freq=1 (small win)
- [x] downsample every 2nd era (small win)
- [x] use all eras / no downsampling (worse — old eras have distribution drift)
- [x] max_depth 5→6, num_leaves 31→63 (small win)
- [x] target_cyrus — CRASHED due to wrong name. Correct name: `target_cyrusd_20`
- [x] target ensemble (target + target_cyrusd_20 + target_victor_20) — corr_sharpe=1.550 but 60+ min/iter (too slow, rolled back)
- [x] per-era mean-centering — untested but abandoned due to runtime; worth trying with DOWNSAMPLE_EVERY_N_ERAS=4

---

## 🎯 Priority ideas for next batch (focus on ensembling & alternative targets)

### 1. Target ensembling (HIGHEST PRIORITY — try this first)
Train the current best LightGBM config on MULTIPLE targets independently,
then average their predictions. This is fast (same model, same features) and
tends to improve both CORR and MMC because different targets capture different
return signals.

Implementation pattern:
```python
TARGETS = ["target", "target_cyrusd_20", "target_victor_20"]
preds = []
for t in TARGETS:
    m = lgb.LGBMRegressor(**MODEL_PARAMS)
    m.fit(train[features], train[t])
    preds.append(m.predict(validation[features]))
validation["prediction"] = np.mean(preds, axis=0)
```
Note: all 3 targets exist in both train.parquet and validation.parquet columns.
Load them: `columns=["era", "target"] + TARGETS + features`

### 2. XGBoost + LightGBM ensemble
Train one XGBoost model and one LightGBM model, average predictions.
```python
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.005,
                               max_depth=6, subsample=0.8, colsample_bytree=0.1)
# average: (lgbm_preds + xgb_preds) / 2
```

### 3. Per-era prediction mean-centering (NOT ranking)
Subtract the mean prediction within each era (not rank normalization — that hurt).
This removes any era-level bias.
```python
validation["prediction"] = validation.groupby("era")["prediction"].transform(
    lambda x: x - x.mean()
)
```

### 4. Alternative single targets (try if ensembling crashes)
- `target_cyrusd_20` — known strong signal
- `target_victor_20` — known strong signal
- `target_jerome_20` — alternative signal

### 5. "Deep" LightGBM config (if time budget allows)
```python
MODEL_PARAMS = dict(
    n_estimators=30_000,
    learning_rate=0.001,
    max_depth=10,
    num_leaves=1024,
    colsample_bytree=0.1,
    min_data_in_leaf=10000,
    subsample=0.8,
    subsample_freq=1,
)
```

### 6. all feature set (2748 features) with current best model
May require more RAM but worth trying.

---

## Constraints
- **Target runtime: < 30 minutes total** (train + evaluate). The 3-model ensemble on every-2nd-era takes 60+ min — too slow.
- To keep iterations fast, follow these rules:
  - **Ensembles (multiple targets or models):** set `DOWNSAMPLE_EVERY_N_ERAS = 4` to reduce training data
  - **Single model experiments:** `DOWNSAMPLE_EVERY_N_ERAS = 2` is fine
  - **Max 2 targets** in any ensemble — 3 targets × every-2nd-era exceeds 60 min
- No external data sources
- No new pip installs — xgboost and numpy are already installed
- Output file interface must stay the same: `validation_predictions.parquet` with columns `[era, prediction, target]`
- `target` column in output must always be the default `target`, regardless of what was trained on

## Notes on Numerai domain
- Features are binned integers 0-4; missing data is coded as 2
- Targets are binned floats: 0, 0.25, 0.5, 0.75, 1.0
- Each era is one week; targets measure 20-business-day forward returns
- The 4-era embargo is mandatory (already in train.py — do not remove it)
- Do NOT rank predictions within era — it was tested and hurt badly
- Per-era MEAN-centering (subtracting mean, not ranking) is untested and promising
