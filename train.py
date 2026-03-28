# EXPERIMENT: increase max_depth to 6 and num_leaves to 63 for moderately deeper trees

import json
import pandas as pd
import lightgbm as lgb
from numerapi import NumerAPI

# --- Config (agent edits this section) ---
DATA_VERSION = "v5.2"
FEATURE_SET = "medium"         # "small" | "medium" | "all"
DOWNSAMPLE_EVERY_N_ERAS = 2    # 1 = use all eras, 4 = every 4th era (faster)
TARGET_COL = "target"

MODEL_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=6,
    num_leaves=2**6 - 1,
    colsample_bytree=0.1,
    subsample=0.8,
    subsample_freq=1,
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
    columns=["era", TARGET_COL] + features,
)
train = train[train["era"].isin(train["era"].unique()[::DOWNSAMPLE_EVERY_N_ERAS])]
print(f"Training on {train['era'].nunique()} eras, {len(train):,} rows, {len(features)} features")

# --- Model ---
model = lgb.LGBMRegressor(**MODEL_PARAMS)
model.fit(train[features], train[TARGET_COL])

# --- Validation predictions ---
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type", TARGET_COL] + features,
)
validation = validation[validation["data_type"] == "validation"].drop(columns=["data_type"])
validation = validation[validation["era"].isin(validation["era"].unique()[::DOWNSAMPLE_EVERY_N_ERAS])]

# Embargo the 4 eras following the last train era to prevent data leakage
# (targets look 20 business days / ~4 eras forward)
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(last_train_era + i).zfill(4) for i in range(4)]
validation = validation[~validation["era"].isin(eras_to_embargo)]

validation["prediction"] = model.predict(validation[features])
validation[["era", "prediction", "target"]].to_parquet("validation_predictions.parquet")
print(f"Validation predictions saved: {len(validation):,} rows across {validation['era'].nunique()} eras")

# --- Live predictions ---
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=features)
live_predictions = model.predict(live_features[features])
pd.Series(live_predictions, index=live_features.index).to_frame("prediction").to_parquet("live_predictions.parquet")
print(f"Live predictions saved: {len(live_features):,} stocks")
