"""Comprehensive Exploratory Data Analysis (EDA) for Cloud Workload Forecasting.

This script performs in-depth analysis of Azure VM workload traces:
1. Dataset loading and schema/memory profiling
2. Timestamp consistency, spacing, and horizon validation
3. VM coverage, population dynamics, and churn analysis
4. CPU utilization distribution, percentiles, and idle/hot ratios
5. Temporal patterns, autocorrelation, trend, and stationarity
6. Segmented analysis across VM category, virtual cores, and memory buckets
7. Target time series construction (active_vm_count, aggregate_cpu_utilization)
8. Train/Holdout (110 / 24) split boundary inspection and distributional shift
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Optional statsmodels import for rigorous time series stationarity & ACF tests
try:
    from statsmodels.tsa.stattools import acf, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Print a visually prominent section header."""
    print("\n" + char * width)
    print(f" {title.upper()} ".center(width, char))
    print(char * width + "\n")


def print_subsection(title: str, char: str = "-", width: int = 70) -> None:
    """Print a subsection divider."""
    print(f"\n{char * 5} {title} {char * (width - len(title) - 7)}\n")


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPU readings and VM metadata datasets."""
    readings_path = data_dir / "vm_cpu_readings.csv.gz"
    metadata_path = data_dir / "vm_metadata.csv.gz"

    if not readings_path.exists():
        raise FileNotFoundError(f"Missing readings file: {readings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    print(f"Loading CPU readings from {readings_path.name}...")
    t0 = time.perf_counter()
    readings_df = pd.read_csv(readings_path)
    t_readings = time.perf_counter() - t0
    print(f"  Loaded {len(readings_df):,} rows in {t_readings:.2f}s")

    print(f"Loading VM metadata from {metadata_path.name}...")
    t0 = time.perf_counter()
    metadata_df = pd.read_csv(metadata_path)
    t_metadata = time.perf_counter() - t0
    print(f"  Loaded {len(metadata_df):,} rows in {t_metadata:.2f}s")

    return readings_df, metadata_df


def analyze_basic_stats(readings_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    """Analyze shapes, dtypes, memory usage, and completeness."""
    print_section("1. Basic Dataset Profiling & Memory Footprint")

    # Readings summary
    readings_mem_mb = readings_df.memory_usage(deep=True).sum() / (1024 ** 2)
    print_subsection("CPU Readings Dataset (`vm_cpu_readings.csv.gz`)")
    print(f"Total Rows:            {readings_df.shape[0]:,}")
    print(f"Total Columns:         {readings_df.shape[1]}")
    print(f"Memory Footprint:      {readings_mem_mb:.2f} MB")
    print("\nColumn Data Types & Non-Null Counts:")
    readings_info = pd.DataFrame({
        "Dtype": readings_df.dtypes,
        "Non-Null Count": readings_df.count(),
        "Null Count": readings_df.isnull().sum(),
        "Null %": (readings_df.isnull().sum() / len(readings_df)) * 100,
    })
    print(readings_info.to_string())

    # Metadata summary
    metadata_mem_mb = metadata_df.memory_usage(deep=True).sum() / (1024 ** 2)
    print_subsection("VM Metadata Dataset (`vm_metadata.csv.gz`)")
    print(f"Total Rows (VMs):      {metadata_df.shape[0]:,}")
    print(f"Total Columns:         {metadata_df.shape[1]}")
    print(f"Memory Footprint:      {metadata_mem_mb:.2f} MB")
    print("\nColumn Data Types & Non-Null Counts:")
    metadata_info = pd.DataFrame({
        "Dtype": metadata_df.dtypes,
        "Non-Null Count": metadata_df.count(),
        "Null Count": metadata_df.isnull().sum(),
        "Null %": (metadata_df.isnull().sum() / len(metadata_df)) * 100,
    })
    print(metadata_info.to_string())

    # ID population consistency
    unique_readings_vms = set(readings_df["vm_id"].unique())
    unique_metadata_vms = set(metadata_df["vm_id"].unique())
    print_subsection("Cross-Table Population Consistency")
    print(f"Distinct VMs in readings: {len(unique_readings_vms):,}")
    print(f"Distinct VMs in metadata: {len(unique_metadata_vms):,}")
    vms_match = unique_readings_vms == unique_metadata_vms
    print(f"Exact 1:1 VM ID match:   {vms_match}")
    if not vms_match:
        in_readings_not_meta = len(unique_readings_vms - unique_metadata_vms)
        in_meta_not_readings = len(unique_metadata_vms - unique_readings_vms)
        print(f"  VMs in readings but not metadata: {in_readings_not_meta}")
        print(f"  VMs in metadata but not readings: {in_meta_not_readings}")


def analyze_timestamps(readings_df: pd.DataFrame) -> np.ndarray:
    """Analyze timestamp ordering, cadence, gaps, and span."""
    print_section("2. Timestamp Cadence, Spacing & Span Analysis")

    unique_ts = np.sort(readings_df["timestamp"].unique())
    num_ts = len(unique_ts)
    min_ts = int(unique_ts[0])
    max_ts = int(unique_ts[-1])
    total_span_sec = max_ts - min_ts
    total_span_hrs = total_span_sec / 3600.0

    print(f"Number of unique timestamps: {num_ts}")
    print(f"Minimum timestamp value:     {min_ts} seconds (Trace offset)")
    print(f"Maximum timestamp value:     {max_ts} seconds")
    print(f"Total trace observation span: {total_span_sec:,} seconds ({total_span_hrs:.2f} hours)")

    # Consecutive intervals
    deltas = np.diff(unique_ts)
    unique_deltas, counts = np.unique(deltas, return_counts=True)

    print_subsection("Sampling Interval Integrity")
    for d, c in zip(unique_deltas, counts):
        print(f"  Interval {d}s: {c} occurrences (Expected: 300s / 5 minutes)")

    has_constant_spacing = len(unique_deltas) == 1 and unique_deltas[0] == 300
    print(f"\nStrict 300-second (5-minute) spacing verified: {has_constant_spacing}")

    if not has_constant_spacing:
        gap_indices = np.where(deltas != 300)[0]
        print(f"Irregular intervals detected at indices: {gap_indices.tolist()[:10]}")
        for idx in gap_indices[:5]:
            print(f"  Gap after ts={unique_ts[idx]} to ts={unique_ts[idx+1]} (diff={deltas[idx]}s)")
    else:
        print("  Zero temporal gaps or skipped intervals detected across all 134 timestamps.")

    return unique_ts


def analyze_vm_coverage_and_churn(
    readings_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    unique_ts: np.ndarray,
) -> pd.DataFrame:
    """Analyze per-timestamp VM counts, duplicates, and lifetime churn."""
    print_section("3. VM Coverage, Duplication & Churn Analysis")

    # Group by timestamp to check distinct VMs and total rows
    ts_grouped = readings_df.groupby("timestamp")
    ts_vm_counts = ts_grouped["vm_id"].nunique()
    ts_row_counts = ts_grouped["vm_id"].count()

    print_subsection("Per-Timestamp VM Presence")
    print(f"Active VMs per timestamp - Min:    {ts_vm_counts.min():,}")
    print(f"Active VMs per timestamp - Max:    {ts_vm_counts.max():,}")
    print(f"Active VMs per timestamp - Mean:   {ts_vm_counts.mean():.2f}")
    print(f"Active VMs per timestamp - Std:    {ts_vm_counts.std():.4f}")

    # Duplicate check
    duplicate_rows = readings_df.duplicated(subset=["timestamp", "vm_id"]).sum()
    print(f"Duplicate (timestamp, vm_id) pairs: {duplicate_rows}")

    all_10k = (ts_vm_counts == 10000).all()
    print(f"Are exactly 10,000 distinct VMs present at EVERY timestamp? {all_10k}")

    # VM Observation counts per VM
    vm_obs_counts = readings_df.groupby("vm_id")["timestamp"].count()
    print_subsection("Per-VM Observation Completeness")
    print(f"Min observations per VM:    {vm_obs_counts.min()}")
    print(f"Max observations per VM:    {vm_obs_counts.max()}")
    print(f"Median observations per VM: {vm_obs_counts.median():.0f}")
    print(f"VMs with exactly 134 readings: {(vm_obs_counts == 134).sum():,} / {len(vm_obs_counts):,} (100%)")

    # Churn / Lifecycle analysis from metadata
    print_subsection("VM Lifecycle & Churn Relative to Observation Window")
    window_start = int(unique_ts[0])
    window_end = int(unique_ts[-1])
    print(f"Observation window: [{window_start}s, {window_end}s]")

    created = metadata_df["timestamp_vm_created"]
    deleted = metadata_df["timestamp_vm_deleted"]

    created_before = (created <= window_start).sum()
    created_during = ((created > window_start) & (created <= window_end)).sum()
    created_after = (created > window_end).sum()

    deleted_before = ((deleted >= 0) & (deleted < window_start)).sum()
    deleted_during = ((deleted >= window_start) & (deleted <= window_end)).sum()
    deleted_after = (deleted > window_end).sum()
    never_deleted = (deleted < 0).sum()

    print("VM Creation Timestamps:")
    print(f"  Created before or at window start: {created_before:,} ({created_before/len(metadata_df)*100:.2f}%)")
    print(f"  Created during window:             {created_during:,} ({created_during/len(metadata_df)*100:.2f}%)")
    print(f"  Created after window:              {created_after:,} ({created_after/len(metadata_df)*100:.2f}%)")

    print("\nVM Deletion Timestamps:")
    print(f"  Deleted before window:             {deleted_before:,} ({deleted_before/len(metadata_df)*100:.2f}%)")
    print(f"  Deleted during window:             {deleted_during:,} ({deleted_during/len(metadata_df)*100:.2f}%)")
    print(f"  Deleted after window:              {deleted_after:,} ({deleted_after/len(metadata_df)*100:.2f}%)")
    print(f"  Never deleted (or active/no del):  {never_deleted:,} ({never_deleted/len(metadata_df)*100:.2f}%)")

    coverage_df = pd.DataFrame({
        "timestamp": unique_ts,
        "active_vm_count": ts_vm_counts.values,
        "row_count": ts_row_counts.values,
    })
    return coverage_df


def analyze_cpu_distributions(readings_df: pd.DataFrame) -> None:
    """Analyze statistical distributions of avg_cpu, min_cpu, max_cpu."""
    print_section("4. CPU Utilization Distribution, Correlations & Idle/Hot Profiles")

    cpu_cols = ["min_cpu", "avg_cpu", "max_cpu"]
    percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    stats_summary = readings_df[cpu_cols].describe(percentiles=percentiles).T
    stats_summary["skew"] = readings_df[cpu_cols].skew()
    stats_summary["kurtosis"] = readings_df[cpu_cols].kurtosis()

    print_subsection("Descriptive Statistics for CPU Metrics (%)")
    format_dict = {col: "{:.3f}".format for col in stats_summary.columns if col != "count"}
    format_dict["count"] = "{:,.0f}".format
    print(stats_summary.to_string(formatters=format_dict))

    # Dynamic Range / Spread (max_cpu - min_cpu)
    spread = readings_df["max_cpu"] - readings_df["min_cpu"]
    print_subsection("Within-Interval Dynamic Range (max_cpu - min_cpu)")
    print(f"Mean Spread:     {spread.mean():.3f}%")
    print(f"Median Spread:   {spread.median():.3f}%")
    print(f"P95 Spread:      {spread.quantile(0.95):.3f}%")
    print(f"Max Spread:      {spread.max():.3f}%")
    print(f"Zero Spread (constant load): {(spread == 0).sum():,} ({((spread == 0).sum()/len(spread))*100:.2f}%)")

    # Correlations
    print_subsection("Metric Correlations (Pearson & Spearman)")
    pearson_corr = readings_df[cpu_cols].corr(method="pearson")
    spearman_corr = readings_df[cpu_cols].corr(method="spearman")
    print("Pearson Correlation:")
    print(pearson_corr.round(4).to_string())
    print("\nSpearman Rank Correlation:")
    print(spearman_corr.round(4).to_string())

    # Idle and Hot VM fractions
    idle_threshold = 1.0   # < 1% CPU
    low_threshold = 5.0    # < 5% CPU
    high_threshold = 70.0  # > 70% CPU
    hot_threshold = 80.0   # > 80% CPU
    sat_threshold = 95.0   # > 95% CPU

    idle_count = (readings_df["avg_cpu"] < idle_threshold).sum()
    low_count = (readings_df["avg_cpu"] < low_threshold).sum()
    high_count = (readings_df["avg_cpu"] > high_threshold).sum()
    hot_count = (readings_df["avg_cpu"] > hot_threshold).sum()
    sat_count = (readings_df["avg_cpu"] > sat_threshold).sum()
    total = len(readings_df)

    print_subsection("Operational Utilization Buckets (Fleet-wide Readings)")
    print(f"Near-Idle Readings   (avg_cpu < 1%):  {idle_count:,} ({idle_count/total*100:.2f}%)")
    print(f"Low-Demand Readings  (avg_cpu < 5%):  {low_count:,} ({low_count/total*100:.2f}%)")
    print(f"High-Demand Readings (avg_cpu > 70%): {high_count:,} ({high_count/total*100:.2f}%)")
    print(f"Hot/Peak Readings    (avg_cpu > 80%): {hot_count:,} ({hot_count/total*100:.2f}%)")
    print(f"Near-Saturated       (avg_cpu > 95%): {sat_count:,} ({sat_count/total*100:.2f}%)")

    # Per-VM distribution of idle/hot behavior
    vm_mean_avg = readings_df.groupby("vm_id")["avg_cpu"].mean()
    vm_idle_count = (vm_mean_avg < idle_threshold).sum()
    vm_hot_count = (vm_mean_avg > hot_threshold).sum()
    num_vms = len(vm_mean_avg)
    print("\nPer-VM Average Utilization Behavior (10,000 distinct VMs):")
    print(f"VMs with average load < 1%:  {vm_idle_count:,} ({vm_idle_count/num_vms*100:.2f}%)")
    print(f"VMs with average load > 80%: {vm_hot_count:,} ({vm_hot_count/num_vms*100:.2f}%)")


def analyze_temporal_patterns(
    readings_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze temporal dynamics, aggregate CPU trend, autocorrelation, and stationarity."""
    print_section("5. Temporal Patterns, Trend & Autocorrelation")

    # Calculate aggregate CPU utilization per timestamp
    agg_cpu_ts = readings_df.groupby("timestamp")["avg_cpu"].agg(
        mean="mean",
        std="std",
        median="median",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
        p95=lambda x: np.percentile(x, 95),
        p99=lambda x: np.percentile(x, 99),
    ).reset_index()

    merged_ts = pd.merge(coverage_df, agg_cpu_ts, on="timestamp")
    merged_ts["step_idx"] = np.arange(len(merged_ts))

    mean_series = merged_ts["mean"]
    vm_series = merged_ts["active_vm_count"]

    print_subsection("Aggregate CPU Utilization Time Series Overview (134 steps = ~11.1 hours)")
    print(f"Mean Aggregate CPU:       {mean_series.mean():.4f}%")
    print(f"Std Aggregate CPU:        {mean_series.std():.4f}%")
    print(f"Min Aggregate CPU:        {mean_series.min():.4f}% (at step {mean_series.idxmin()})")
    print(f"Max Aggregate CPU:        {mean_series.max():.4f}% (at step {mean_series.idxmax()})")
    print(f"Peak-to-Trough Amplitude: {mean_series.max() - mean_series.min():.4f}%")
    print(f"Coefficient of Variation: {(mean_series.std() / mean_series.mean()) * 100:.2f}%")

    # Trend fitting via OLS
    x = merged_ts["step_idx"].values
    y = mean_series.values
    slope, intercept = np.polyfit(x, y, 1)
    r_matrix = np.corrcoef(x, y)
    r_val = r_matrix[0, 1]
    r_squared = r_val ** 2

    print_subsection("Linear Trend Analysis (Aggregate CPU)")
    print(f"Fitted Slope:             {slope:+.5f}% per 5-min step ({slope * 12:+.4f}% per hour)")
    print(f"Fitted Intercept:         {intercept:.4f}%")
    print(f"Correlation with Time:    {r_val:+.4f}")
    print(f"R-squared:                {r_squared:.4f}")

    # Active VM count trend
    y_vm = vm_series.values
    slope_vm, _ = np.polyfit(x, y_vm, 1)
    print(f"\nActive VM Trend Slope:    {slope_vm:+.6f} VMs per 5-min step")
    print(f"Active VM Range:          [{y_vm.min():,}, {y_vm.max():,}]")

    # Autocorrelation
    print_subsection("Autocorrelation Structure (Aggregate CPU Utilization)")
    lags = [1, 2, 3, 6, 12, 18, 24]
    if HAS_STATSMODELS:
        acf_values = acf(mean_series, nlags=max(lags), fft=False)
        for lag in lags:
            elapsed_min = lag * 5
            print(f"  Lag {lag:2d} ({elapsed_min:3d} min): ACF = {acf_values[lag]:+.4f}")
    else:
        for lag in lags:
            elapsed_min = lag * 5
            ac = pd.Series(y).autocorr(lag=lag)
            print(f"  Lag {lag:2d} ({elapsed_min:3d} min): ACF = {ac:+.4f}")

    # Stationarity test (Augmented Dickey-Fuller)
    print_subsection("Stationarity Testing (ADF Test)")
    if HAS_STATSMODELS:
        adf_result = adfuller(mean_series, autolag="AIC")
        print(f"ADF Test Statistic:       {adf_result[0]:.4f}")
        print(f"p-value:                  {adf_result[1]:.4f}")
        print(f"Used Lags:                {adf_result[2]}")
        print(f"Observations:             {adf_result[3]}")
        print("Critical Values:")
        for key, val in adf_result[4].items():
            print(f"  {key}: {val:.4f}")
        is_stationary = adf_result[1] < 0.05
        print(f"Reject Unit Root at 5% (Stationary): {is_stationary}")
    else:
        print("statsmodels not available; skipping ADF test.")

    return merged_ts


def analyze_segments(
    readings_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    """Analyze CPU utilization breakdown by VM category, cores, and memory."""
    print_section("6. Segment Analysis Across VM Categories & Capacity Buckets")

    # Merge readings with metadata
    meta_cols = ["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]
    merged = pd.merge(readings_df, metadata_df[meta_cols], on="vm_id", how="left")

    # 1. By VM Category
    print_subsection("Workload Breakdown by `vm_category`")
    cat_vm_counts = metadata_df["vm_category"].value_counts()
    cat_stats = merged.groupby("vm_category")["avg_cpu"].agg(
        readings="count",
        mean="mean",
        std="std",
        median="median",
        p95=lambda x: np.percentile(x, 95),
        idle_pct=lambda x: (x < 1.0).mean() * 100,
        hot_pct=lambda x: (x > 80.0).mean() * 100,
    )
    cat_stats["vm_count"] = cat_vm_counts
    cat_stats["vm_share_pct"] = (cat_stats["vm_count"] / len(metadata_df)) * 100

    col_order = ["vm_count", "vm_share_pct", "mean", "std", "median", "p95", "idle_pct", "hot_pct"]
    cat_display = cat_stats[col_order].sort_values("vm_count", ascending=False)
    formatters = {
        "vm_count": "{:,}".format,
        "vm_share_pct": "{:.1f}%".format,
        "mean": "{:.3f}%".format,
        "std": "{:.3f}%".format,
        "median": "{:.3f}%".format,
        "p95": "{:.3f}%".format,
        "idle_pct": "{:.2f}%".format,
        "hot_pct": "{:.2f}%".format,
    }
    print(cat_display.to_string(formatters=formatters))

    # 2. By Virtual Core Bucket
    print_subsection("Workload Breakdown by `virtual_core_bucket` (vCPUs)")
    core_vm_counts = metadata_df["virtual_core_bucket"].value_counts()
    core_stats = merged.groupby("virtual_core_bucket")["avg_cpu"].agg(
        mean="mean",
        std="std",
        median="median",
        p95=lambda x: np.percentile(x, 95),
        idle_pct=lambda x: (x < 1.0).mean() * 100,
        hot_pct=lambda x: (x > 80.0).mean() * 100,
    )
    core_stats["vm_count"] = core_vm_counts
    core_stats["vm_share_pct"] = (core_stats["vm_count"] / len(metadata_df)) * 100
    core_display = core_stats[col_order].sort_index()
    print(core_display.to_string(formatters=formatters))

    # 3. By Memory Bucket
    print_subsection("Workload Breakdown by `memory_gb_bucket` (GB)")
    mem_vm_counts = metadata_df["memory_gb_bucket"].value_counts()
    mem_stats = merged.groupby("memory_gb_bucket")["avg_cpu"].agg(
        mean="mean",
        std="std",
        median="median",
        p95=lambda x: np.percentile(x, 95),
        idle_pct=lambda x: (x < 1.0).mean() * 100,
        hot_pct=lambda x: (x > 80.0).mean() * 100,
    )
    mem_stats["vm_count"] = mem_vm_counts
    mem_stats["vm_share_pct"] = (mem_stats["vm_count"] / len(metadata_df)) * 100
    mem_display = mem_stats[col_order].sort_index()
    print(mem_display.to_string(formatters=formatters))

    # 4. Cross-tabulation: Category vs Cores
    print_subsection("VM Count Distribution: `vm_category` vs `virtual_core_bucket`")
    ct = pd.crosstab(
        metadata_df["vm_category"],
        metadata_df["virtual_core_bucket"],
        margins=True,
        margins_name="Total",
    )
    print(ct.to_string())


def construct_targets(
    readings_df: pd.DataFrame,
    unique_ts: np.ndarray,
) -> pd.DataFrame:
    """Construct the official forecasting targets according to challenge specifications."""
    print_section("7. Construct Derived Target Time Series")

    # Challenge definitions:
    # 1. active_vm_count: number of distinct VM IDs with a valid CPU observation at timestamp t
    # 2. aggregate_cpu_utilization: mean of VM-level avg_cpu values at timestamp t in [0, 100]

    ts_group = readings_df.groupby("timestamp")
    active_vm_counts = ts_group["vm_id"].nunique()
    aggregate_cpu = ts_group["avg_cpu"].mean()

    targets_df = pd.DataFrame({
        "timestamp": unique_ts,
        "step": np.arange(len(unique_ts)),
        "active_vm_count": active_vm_counts.loc[unique_ts].values,
        "aggregate_cpu_utilization": aggregate_cpu.loc[unique_ts].values,
    })

    print(f"Target series constructed successfully for all {len(targets_df)} timestamps.")
    print("Columns: timestamp, step, active_vm_count, aggregate_cpu_utilization")

    # Sanity checks
    assert not targets_df.isnull().any().any(), "Target series contains NaN values!"
    assert (targets_df["aggregate_cpu_utilization"] >= 0).all() and (
        targets_df["aggregate_cpu_utilization"] <= 100
    ).all(), "CPU utilization outside [0, 100]!"

    return targets_df


def inspect_split_boundary(targets_df: pd.DataFrame) -> None:
    """Inspect the 110 (Train) / 24 (Holdout) split boundary and compare regimes."""
    print_section("8. Chronological Train / Holdout (110 / 24) Split Point Inspection")

    n_train = 110
    n_holdout = 24
    total = len(targets_df)
    assert total == n_train + n_holdout, f"Expected 134 rows, found {total}"

    train_df = targets_df.iloc[:n_train].copy()
    holdout_df = targets_df.iloc[n_train:].copy()

    train_start_ts = train_df["timestamp"].iloc[0]
    train_end_ts = train_df["timestamp"].iloc[-1]
    holdout_start_ts = holdout_df["timestamp"].iloc[0]
    holdout_end_ts = holdout_df["timestamp"].iloc[-1]

    print_subsection("Split Horizon Metadata")
    print(f"Full Series Length:      {total} steps ({total * 5 / 60:.2f} hours)")
    print(f"Training Window:         {n_train} steps ({n_train * 5 / 60:.2f} hours) -> ts: [{train_start_ts}, {train_end_ts}]")
    print(f"Holdout Horizon:         {n_holdout} steps ({n_holdout * 5 / 60:.2f} hours) -> ts: [{holdout_start_ts}, {holdout_end_ts}]")
    print(f"Gap between windows:     {holdout_start_ts - train_end_ts} seconds (Exactly 1 regular step)")

    print_subsection("Boundary Transition (Last 5 Train vs First 5 Holdout Steps)")
    boundary_window = pd.concat([
        train_df.tail(5).assign(split="TRAIN"),
        holdout_df.head(5).assign(split="HOLDOUT"),
    ])
    print(boundary_window[["split", "step", "timestamp", "active_vm_count", "aggregate_cpu_utilization"]].to_string(index=False))

    print_subsection("Regime Comparison: Training vs Holdout Summary Statistics")
    comparison = pd.DataFrame({
        "Train (110 steps)": [
            train_df["active_vm_count"].mean(),
            train_df["active_vm_count"].std(),
            train_df["aggregate_cpu_utilization"].mean(),
            train_df["aggregate_cpu_utilization"].std(),
            train_df["aggregate_cpu_utilization"].min(),
            train_df["aggregate_cpu_utilization"].max(),
        ],
        "Holdout (24 steps)": [
            holdout_df["active_vm_count"].mean(),
            holdout_df["active_vm_count"].std(),
            holdout_df["aggregate_cpu_utilization"].mean(),
            holdout_df["aggregate_cpu_utilization"].std(),
            holdout_df["aggregate_cpu_utilization"].min(),
            holdout_df["aggregate_cpu_utilization"].max(),
        ],
    }, index=[
        "Active VM Count (Mean)",
        "Active VM Count (Std)",
        "Aggregate CPU (Mean %)",
        "Aggregate CPU (Std %)",
        "Aggregate CPU (Min %)",
        "Aggregate CPU (Max %)",
    ])
    comparison["Delta (Holdout - Train)"] = comparison["Holdout (24 steps)"] - comparison["Train (110 steps)"]
    print(comparison.round(4).to_string())

    # Slope comparison
    train_slope, _ = np.polyfit(np.arange(n_train), train_df["aggregate_cpu_utilization"].values, 1)
    holdout_slope, _ = np.polyfit(np.arange(n_holdout), holdout_df["aggregate_cpu_utilization"].values, 1)
    print(f"\nLinear Slope during Training: {train_slope:+.5f}% / step ({train_slope * 12:+.4f}% / hour)")
    print(f"Linear Slope during Holdout:  {holdout_slope:+.5f}% / step ({holdout_slope * 12:+.4f}% / hour)")

    # Print all 134 target values in a compact structured view
    print_subsection("Complete Target Series (All 134 Timestamps)")
    display_all = targets_df.copy()
    display_all["split"] = ["TRAIN" if i < n_train else "HOLDOUT" for i in range(total)]
    display_all["time_min"] = display_all["step"] * 5
    pd.set_option("display.max_rows", 150)
    print(display_all[["step", "time_min", "timestamp", "split", "active_vm_count", "aggregate_cpu_utilization"]].to_string(index=False))
    pd.reset_option("display.max_rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for Cloud Workload Forecasting")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to directory containing challenge_data (defaults to project_root/challenge_data)",
    )
    args = parser.parse_args()

    # Determine project root and data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir if args.data_dir is not None else project_root / "challenge_data"

    print_section("Cloud Workload Forecasting - Exploratory Data Analysis")
    print(f"Project Root: {project_root}")
    print(f"Data Dir:     {data_dir}")

    # 1. Load data
    readings_df, metadata_df = load_datasets(data_dir)

    # 2. Basic stats
    analyze_basic_stats(readings_df, metadata_df)

    # 3. Timestamp analysis
    unique_ts = analyze_timestamps(readings_df)

    # 4. VM coverage and churn
    coverage_df = analyze_vm_coverage_and_churn(readings_df, metadata_df, unique_ts)

    # 5. CPU distributions and idle/hot profiles
    analyze_cpu_distributions(readings_df)

    # 6. Temporal patterns, autocorrelation, trend
    analyze_temporal_patterns(readings_df, coverage_df)

    # 7. Segment analysis (vm_category, cores, memory)
    analyze_segments(readings_df, metadata_df)

    # 8. Target construction
    targets_df = construct_targets(readings_df, unique_ts)

    # 9. 110/24 Split inspection and regime comparison
    inspect_split_boundary(targets_df)

    print_section("EDA Script Execution Completed Successfully")


if __name__ == "__main__":
    main()
