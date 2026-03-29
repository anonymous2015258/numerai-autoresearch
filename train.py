# EXPERIMENT: add reg_lambda=1.0 (L2 regularization) to target+cyrusd_20 ensemble for additional regularization

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI

# --- Config (agent edits this section) ---
DATA_VERSION = "v5.2"
FEATURE_SET = "medium"         # "small" | "medium" | "all"
DOWNSAMPLE_EVERY_N_ERAS = 4    # 4 = every 4th era (faster for ensembles)
TARGETS = ["target", "target_cyrusd_20"]

MODEL_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=7,
    num_leaves=2**7 - 1,
    colsample_bytree=0.05,
    subsample=0.8,
    subsample_freq=1,
    reg_lambda=1.0,
)
# -------------------------------------------

napi = NumerAPI()

# Download metadata
napi.download_dataset(f"{DATA_VERSION}/features.json")
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
features = feature_metadata["feature_sets"][FEATURE_SET]

# --- Training data ---
napi.download_dataset(f"{DATA_VERSION}/train.parquet")
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era"] + TARGETS + features,
)
train = train[train["era"].isin(train["era"].unique()[::DOWNSAMPLE_EVERY_N_ERAS])]
print(f"Training on {train['era'].nunique()} eras, {len(train):,} rows, {len(features)} features")

# --- Model: ensemble over two targets ---
preds_val = []
preds_live = []

# --- Validation predictions ---
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type"] + TARGETS + features,
)
validation = validation[validation["data_type"] == "validation"].drop(columns=["data_type"])
# Embargo the 4 eras following the last train era to prevent data leakage
# (targets look 20 business days / ~4 eras forward)
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(last_train_era + i).zfill(4) for i in range(4)]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# --- Live data ---
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=features)

for t in TARGETS:
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(train[features], train[t])
    preds_val.append(model.predict(validation[features]))
    preds_live.append(model.predict(live_features[features]))
    print(f"Trained on target: {t}")

validation["prediction"] = np.mean(preds_val, axis=0)
validation[["era", "prediction", "target"]].to_parquet("validation_predictions.parquet")
print(f"Validation predictions saved: {len(validation):,} rows across {validation['era'].nunique()} eras")

# --- Live predictions ---
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=features)
live_predictions = model.predict(live_features[features])
live_predictions = np.mean(preds_live, axis=0)
pd.Series(live_predictions, index=live_features.index).to_frame("prediction").to_parquet("live_predictions.parquet")
print(f"Live predictions saved: {len(live_features):,} stocks")
