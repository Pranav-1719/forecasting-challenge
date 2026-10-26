"""Main prediction pipeline for the Cloud Workload Forecasting Challenge.

Usage:
    python -m cwfc.predict --data challenge_data --output predictions.csv

This pipeline:
1. Loads the VM CPU readings and metadata from the specified data directory.
2. Constructs the 5-minute aggregate target time series (active_vm_count and aggregate_cpu_utilization).
3. Evaluates simple baselines (Persistence, Moving Average, Seasonal Naive) on the training window.
4. Executes time-aware expanding-window cross-validation (4 folds) using LightGBM.
5. Fits the final forecasting models strictly on the 110-step training window.
6. Generates recursive multi-step predictions across the 24-step (2-hour) holdout horizon.
7. Saves the output to predictions.csv satisfying all challenge contract requirements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to sys.path to allow execution both as script and as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cwfc.data_loader import load_cpu_readings, load_metadata, build_aggregate_targets
from cwfc.baselines import persistence_forecast, seasonal_naive_forecast
from cwfc.forecasting import moving_average_forecast, run_pipeline
from cwfc.metrics import mae, rmse, smape, normalized_mae


def run_forecast(data_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Execute the end-to-end forecasting pipeline and generate predictions.csv.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing vm_cpu_readings.csv.gz and vm_metadata.csv.gz.
    output_path : str or Path
        Target path for the output CSV file (e.g. 'predictions.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame of generated predictions matching SUBMISSION.md schema.
    """
    data_path = Path(data_dir)
    print("=" * 80)
    print("         CLOUD WORKLOAD FORECASTING - PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Dataset directory: {data_path.resolve()}")

    # 1. Load data
    print("\n[1/5] Loading datasets...")
    cpu_df = load_cpu_readings(data_path)
    metadata_df = load_metadata(data_path)
    print(f"  -> CPU readings: {len(cpu_df):,} rows")
    print(f"  -> VM metadata:  {len(metadata_df):,} distinct VMs")

    # 2. Build aggregate targets
    print("\n[2/5] Constructing derived aggregate target series (5-minute cadence)...")
    targets = build_aggregate_targets(cpu_df)
    n_total = len(targets)
    n_train = 110
    n_holdout = 24

    if n_total != n_train + n_holdout:
        raise ValueError(
            f"Expected {n_train + n_holdout} timestamps, found {n_total} in aggregate targets."
        )

    train_targets = targets.iloc[:n_train]
    holdout_targets = targets.iloc[n_train:]
    holdout_timestamps = holdout_targets.index.to_numpy()

    print(f"  -> Total series length:  {n_total} timestamps (~{n_total * 5 / 60:.1f} hours)")
    print(f"  -> Training window:      {n_train} timestamps (steps 0..{n_train - 1}, ts {train_targets.index.min()}..{train_targets.index.max()})")
    print(f"  -> Holdout forecast:     {n_holdout} timestamps (steps {n_train}..{n_total - 1}, ts {holdout_targets.index.min()}..{holdout_targets.index.max()})")

    # 3. Baseline evaluation on holdout
    print("\n[3/5] Computing baseline forecasts for comparison...")
    baselines: dict[str, dict[str, np.ndarray]] = {}

    for target_col in ["active_vm_count", "aggregate_cpu_utilization"]:
        y_train = train_targets[target_col].to_numpy(dtype=float)
        y_holdout = holdout_targets[target_col].to_numpy(dtype=float)

        # Baseline A: Persistence (last known training observation)
        p_pred = np.full(n_holdout, y_train[-1])

        # Baseline B: Moving Average (window = 12 steps / 1 hour)
        ma12_pred = moving_average_forecast(y_train, horizon=n_holdout, window=12)

        # Baseline C: Seasonal Naive (repeat last 24 steps / 2 hours)
        sn_pred = y_train[-24:]

        baselines[target_col] = {
            "persistence": p_pred,
            "ma12": ma12_pred,
            "seasonal24": sn_pred,
        }

        print(f"\n  Baseline Metrics for '{target_col}':")
        print(f"    Persistence:   MAE = {mae(y_holdout, p_pred):.4f} | RMSE = {rmse(y_holdout, p_pred):.4f} | sMAPE = {smape(y_holdout, p_pred):.2f}%")
        print(f"    MA(12):        MAE = {mae(y_holdout, ma12_pred):.4f} | RMSE = {rmse(y_holdout, ma12_pred):.4f} | sMAPE = {smape(y_holdout, ma12_pred):.2f}%")
        print(f"    Seasonal(24):  MAE = {mae(y_holdout, sn_pred):.4f} | RMSE = {rmse(y_holdout, sn_pred):.4f} | sMAPE = {smape(y_holdout, sn_pred):.2f}%")

    # 4. Run time-aware ML forecasting pipeline
    print("\n[4/5] Executing LightGBM time-aware cross-validation & forecasting pipeline...")

    final_predictions: dict[str, np.ndarray] = {}

    # Target 1: active_vm_count
    print("\n  Target: active_vm_count")
    vms_result = run_pipeline(
        targets_df=targets,
        target_col="active_vm_count",
        n_train=n_train,
        n_holdout=n_holdout,
    )
    vms_preds = vms_result.predictions.to_numpy()
    # Clamping and domain validation: non-negative integers
    vms_preds_clamped = np.maximum(np.round(vms_preds), 0).astype(int)
    final_predictions["active_vm_count"] = vms_preds_clamped

    vms_true = holdout_targets["active_vm_count"].to_numpy(dtype=float)
    print(f"    Holdout Evaluation -> MAE: {mae(vms_true, vms_preds_clamped):.4f} | RMSE: {rmse(vms_true, vms_preds_clamped):.4f} | sMAPE: {smape(vms_true, vms_preds_clamped):.2f}%")

    # Target 2: aggregate_cpu_utilization
    print("\n  Target: aggregate_cpu_utilization")
    cpu_result = run_pipeline(
        targets_df=targets,
        target_col="aggregate_cpu_utilization",
        n_train=n_train,
        n_holdout=n_holdout,
    )
    cpu_preds = cpu_result.predictions.to_numpy()
    # Clamping to [0, 100] as specified in SUBMISSION.md
    cpu_preds_clamped = np.clip(cpu_preds, 0.0, 100.0)
    final_predictions["aggregate_cpu_utilization"] = cpu_preds_clamped

    cpu_true = holdout_targets["aggregate_cpu_utilization"].to_numpy(dtype=float)
    cpu_mae = mae(cpu_true, cpu_preds_clamped)
    cpu_rmse = rmse(cpu_true, cpu_preds_clamped)
    cpu_smape = smape(cpu_true, cpu_preds_clamped)
    cpu_norm_mae = normalized_mae(cpu_true, cpu_preds_clamped, baselines["aggregate_cpu_utilization"]["persistence"])

    print("    Expanding-Window Cross-Validation (4 folds):")
    cv_info = cpu_result.cv_metrics
    for f in cv_info.get("folds", []):
        print(f"      Fold {f['fold']} (Train {f['train_size']} -> Val {f['val_size']}): LGB MAE = {f['lgb_mae']:.4f} | Baseline MAE = {f['baseline_mae']:.4f} | Norm MAE = {f['normalized_mae']:.4f}")
    print(f"      Mean CV LGB MAE:       {cv_info.get('lgb_mae_mean', 0.0):.4f} ± {cv_info.get('lgb_mae_std', 0.0):.4f}")
    print(f"      Mean CV Baseline MAE:  {cv_info.get('baseline_mae_mean', 0.0):.4f}")
    print(f"      Mean CV Normalized MAE:{cv_info.get('normalized_mae_mean', 1.0):.4f}")

    print(f"\n    Holdout Evaluation:")
    print(f"      Holdout MAE:            {cpu_mae:.4f}")
    print(f"      Holdout RMSE:           {cpu_rmse:.4f}")
    print(f"      Holdout sMAPE:          {cpu_smape:.2f}%")
    print(f"      Normalized MAE (vs Persist): {cpu_norm_mae:.4f} ({(1.0 - cpu_norm_mae) * 100:.1f}% relative error reduction)")

    # 5. Format and save predictions.csv
    print("\n[5/5] Formatting and exporting predictions.csv...")
    predictions_df = pd.DataFrame({
        "timestamp": holdout_timestamps,
        "active_vm_count": final_predictions["active_vm_count"],
        "aggregate_cpu_utilization": np.round(final_predictions["aggregate_cpu_utilization"], 6),
    })

    # Validate against SUBMISSION.md contract
    assert len(predictions_df) == 24, f"Expected 24 rows, got {len(predictions_df)}"
    assert list(predictions_df.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"], (
        f"Incorrect column names: {predictions_df.columns}"
    )
    assert not predictions_df.isna().any().any(), "Found NaN values in predictions"
    assert (predictions_df["active_vm_count"] >= 0).all(), "active_vm_count contains negative values"
    assert (predictions_df["aggregate_cpu_utilization"] >= 0).all(), "aggregate_cpu_utilization contains values < 0"
    assert (predictions_df["aggregate_cpu_utilization"] <= 100).all(), "aggregate_cpu_utilization contains values > 100"

    out_file = Path(output_path)
    predictions_df.to_csv(out_file, index=False)
    print(f"\nSuccessfully wrote predictions to: {out_file.resolve()}")
    print("\n--- Predictions Preview (All 24 Horizon Timestamps) ---")
    print(predictions_df.to_string(index=False))
    print("=" * 80)

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Cloud Workload Forecasting Challenge - Generate Predictions"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to the challenge data directory containing vm_cpu_readings.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to output predictions CSV (default: predictions.csv)",
    )
    args = parser.parse_args()
    run_forecast(args.data, args.output)


if __name__ == "__main__":
    main()
