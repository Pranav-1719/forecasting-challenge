"""Data loading and target series construction for the Cloud Workload Forecasting Challenge."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _resolve_file_path(data_dir: str | Path, base_name: str) -> Path:
    """Resolve file path supporting directories, compressed (.csv.gz), and uncompressed (.csv) formats."""
    path = Path(data_dir)
    if path.is_file():
        return path

    gz_path = path / f"{base_name}.csv.gz"
    if gz_path.is_file():
        return gz_path

    csv_path = path / f"{base_name}.csv"
    if csv_path.is_file():
        return csv_path

    raise FileNotFoundError(
        f"Could not find '{base_name}.csv.gz' or '{base_name}.csv' in '{data_dir}'"
    )


def load_cpu_readings(data_dir: str | Path) -> pd.DataFrame:
    """Load VM CPU readings dataset.

    Args:
        data_dir: Directory containing 'vm_cpu_readings.csv.gz' (or path to the file itself).

    Returns:
        pd.DataFrame: CPU readings sorted by timestamp and vm_id with columns:
            ['timestamp', 'vm_id', 'min_cpu', 'max_cpu', 'avg_cpu'].
    """
    file_path = _resolve_file_path(data_dir, "vm_cpu_readings")
    df = pd.read_csv(
        file_path,
        dtype={
            "timestamp": "int64",
            "vm_id": "str",
            "min_cpu": "float64",
            "max_cpu": "float64",
            "avg_cpu": "float64",
        },
    )
    df = df.sort_values(by=["timestamp", "vm_id"]).reset_index(drop=True)
    return df


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load VM metadata dataset.

    Args:
        data_dir: Directory containing 'vm_metadata.csv.gz' (or path to the file itself).

    Returns:
        pd.DataFrame: VM metadata sorted by vm_id with columns including:
            ['vm_id', 'subscription_id', 'deployment_id', 'timestamp_vm_created',
             'timestamp_vm_deleted', 'max_cpu', 'avg_cpu', 'p95_max_cpu',
             'vm_category', 'virtual_core_bucket', 'memory_gb_bucket'].
    """
    file_path = _resolve_file_path(data_dir, "vm_metadata")
    df = pd.read_csv(file_path)
    if "vm_id" in df.columns:
        df = df.sort_values(by="vm_id").reset_index(drop=True)
    return df


def build_aggregate_targets(cpu_df: pd.DataFrame) -> pd.DataFrame:
    """Construct aggregate target time series from VM-level CPU readings.

    Constructs the two required target time series:
    1. active_vm_count: number of distinct VMs active at each timestamp
    2. aggregate_cpu_utilization: mean of VM-level avg_cpu across active VMs at each timestamp

    Args:
        cpu_df: DataFrame containing at least 'timestamp', 'vm_id', and 'avg_cpu' columns,
            or indexed by 'timestamp' with 'vm_id' and 'avg_cpu' columns.

    Returns:
        pd.DataFrame: Indexed by 'timestamp' (sorted ascending in seconds) with columns:
            - 'active_vm_count' (int64)
            - 'aggregate_cpu_utilization' (float64)
    """
    if cpu_df.empty:
        empty_df = pd.DataFrame(
            columns=["active_vm_count", "aggregate_cpu_utilization"]
        )
        empty_df.index.name = "timestamp"
        empty_df["active_vm_count"] = empty_df["active_vm_count"].astype("int64")
        empty_df["aggregate_cpu_utilization"] = empty_df[
            "aggregate_cpu_utilization"
        ].astype("float64")
        return empty_df

    if "timestamp" in cpu_df.columns:
        required_cols = {"timestamp", "vm_id", "avg_cpu"}
        missing = required_cols - set(cpu_df.columns)
        if missing:
            raise ValueError(f"cpu_df missing required columns: {missing}")
        grouped = cpu_df.groupby("timestamp", sort=True)
    elif cpu_df.index.name == "timestamp":
        required_cols = {"vm_id", "avg_cpu"}
        missing = required_cols - set(cpu_df.columns)
        if missing:
            raise ValueError(f"cpu_df missing required columns: {missing}")
        grouped = cpu_df.groupby(level="timestamp", sort=True)
    else:
        raise ValueError(
            "cpu_df must either contain a 'timestamp' column or have index named 'timestamp'"
        )

    targets = grouped.agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    ).sort_index()

    targets.index.name = "timestamp"
    targets["active_vm_count"] = targets["active_vm_count"].astype("int64")
    targets["aggregate_cpu_utilization"] = targets["aggregate_cpu_utilization"].astype(
        "float64"
    )
    return targets


def train_test_split(
    targets_df: pd.DataFrame,
    n_train: int = 110,
    n_holdout: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aggregate target time series chronologically into train and holdout sets.

    Args:
        targets_df: Aggregate targets DataFrame indexed chronologically by timestamp.
        n_train: Number of chronological observations for the training window (default 110).
        n_holdout: Number of chronological observations for the holdout horizon (default 24).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, holdout_df)

    Raises:
        ValueError: If n_train or n_holdout are non-positive, or if targets_df has
            fewer rows than n_train + n_holdout.
    """
    if n_train <= 0 or n_holdout <= 0:
        raise ValueError(
            f"n_train and n_holdout must be positive integers, got n_train={n_train}, n_holdout={n_holdout}"
        )

    total_required = n_train + n_holdout
    if len(targets_df) < total_required:
        raise ValueError(
            f"targets_df has {len(targets_df)} rows, expected at least {total_required} "
            f"(n_train={n_train}, n_holdout={n_holdout})"
        )

    if not targets_df.index.is_monotonic_increasing:
        targets_df = targets_df.sort_index()

    train_df = targets_df.iloc[:n_train].copy()
    holdout_df = targets_df.iloc[n_train : n_train + n_holdout].copy()

    return train_df, holdout_df


__all__ = [
    "load_cpu_readings",
    "load_metadata",
    "build_aggregate_targets",
    "train_test_split",
]
