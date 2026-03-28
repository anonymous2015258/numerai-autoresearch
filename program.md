# Numerai Improvement Program

## Goal
Maximize **CORR Sharpe** on the held-out validation eras.
Secondary goal: keep **MMC Sharpe > 0** (we want to be additive to the Numerai Meta Model).

## Current best (update this after each session)
- CORR Sharpe: 0.0 (not yet established — first run will set baseline)
- MMC Sharpe: 0.0

## Search space — ideas to explore (roughly in priority order)

### Feature sets
- [ ] Switch from `small` (42) → `medium` (780 features)
- [ ] Switch from `small` → `all` (2748 features)
- [ ] Use all eras instead of downsampling every 4th era

### Target selection
- [ ] Train on `target_cyrus` instead of `target` (often stronger signal)
- [ ] Train on `target_jerome`
- [ ] Train on `target_victor`
- [ ] Ensemble predictions from 2-3 targets (average their predictions)

### Model hyperparameters
- [ ] Increase `n_estimators` to 5000 with `learning_rate=0.005`
- [ ] Try "deep" config: `n_estimators=30_000`, `learning_rate=0.001`, `max_depth=10`, `num_leaves=1024`, `min_data_in_leaf=10000`
- [ ] Increase `colsample_bytree` to 0.3 or 0.5
- [ ] Add `subsample=0.8` and `subsample_freq=1`

### Feature engineering
- [ ] Normalize predictions per era (rank within each era before outputting)
- [ ] Add era-level feature means as additional features
- [ ] Drop features with near-zero variance across eras

### Model architecture
- [ ] Try XGBoost instead of LightGBM
- [ ] Try CatBoost
- [ ] Try a simple neural network (MLPRegressor or torch)

## Constraints
- Must run in < 45 minutes total (train + evaluate)
- No external data sources
- No new pip installs beyond what is already installed
- Output file interface must stay the same: `validation_predictions.parquet` with columns `[era, prediction, target]`

## Notes on Numerai domain
- Features are binned integers 0-4; missing data is coded as 2
- Targets are binned floats: 0, 0.25, 0.5, 0.75, 1.0
- Each era is one week; targets measure 20-business-day forward returns
- Era-based scoring is the ground truth — single-era overfitting is penalized
- The 4-era embargo is mandatory (already in train.py — do not remove it)
- Prediction neutralization (rank within era) often improves Sharpe by reducing drawdown
