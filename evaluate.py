"""
Fixed evaluation harness — never edited by the agent.
Reads validation_predictions.parquet, computes canonical metrics, writes metrics.json.
"""

import json
import sys
import pandas as pd
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution

META_MODEL_PATH = "v4.3/meta_model.parquet"
META_MODEL_ROUND = 842


def compute_metrics(per_era_df: pd.DataFrame) -> dict:
    """Compute mean, std, sharpe, max_drawdown from a per-era scores DataFrame."""
    scores = per_era_df["prediction"]
    mean = float(scores.mean())
    std = float(scores.std(ddof=0))
    sharpe = mean / std if std > 0 else 0.0
    cumulative = scores.cumsum()
    max_drawdown = float((cumulative.expanding(min_periods=1).max() - cumulative).max())
    return {"mean": mean, "std": std, "sharpe": sharpe, "max_drawdown": max_drawdown}


def main():
    # Load validation predictions produced by train.py
    try:
        validation = pd.read_parquet("validation_predictions.parquet")
    except FileNotFoundError:
        print("ERROR: validation_predictions.parquet not found. Run train.py first.")
        sys.exit(1)

    # Download meta model (cached after first run)
    napi = NumerAPI()
    napi.download_dataset(META_MODEL_PATH, round_num=META_MODEL_ROUND)
    meta_model = pd.read_parquet(META_MODEL_PATH)["numerai_meta_model"]
    validation["meta_model"] = meta_model

    # Per-era CORR
    per_era_corr = validation.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
    )

    # Per-era MMC (only eras where meta_model is available)
    per_era_mmc = validation.dropna(subset=["meta_model"]).groupby("era").apply(
        lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
    )

    corr_metrics = compute_metrics(per_era_corr)
    mmc_metrics = compute_metrics(per_era_mmc)

    results = {"corr": corr_metrics, "mmc": mmc_metrics}

    # Print human-readable summary
    print(f"CORR  — mean: {corr_metrics['mean']:.6f}  std: {corr_metrics['std']:.6f}  "
          f"sharpe: {corr_metrics['sharpe']:.4f}  max_dd: {corr_metrics['max_drawdown']:.4f}")
    print(f"MMC   — mean: {mmc_metrics['mean']:.6f}  std: {mmc_metrics['std']:.6f}  "
          f"sharpe: {mmc_metrics['sharpe']:.4f}  max_dd: {mmc_metrics['max_drawdown']:.4f}")

    # Write machine-readable output for pipeline.py
    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Metrics saved to metrics.json")

    return results


if __name__ == "__main__":
    main()
