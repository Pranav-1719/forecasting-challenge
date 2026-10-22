from __future__ import annotations

import io
import gzip
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from cwfc.data_loader import (
    load_cpu_readings,
    load_metadata,
    build_aggregate_targets,
    train_test_split,
)


def test_build_aggregate_targets_calculation():
    # Construct a controlled synthetic dataset
    # Timestamp 0: 3 readings across 2 distinct VMs
    # Timestamp 300: 2 readings across 2 distinct VMs
    data = {
        "timestamp": [0, 0, 0, 300, 300],
        "vm_id": ["vm1", "vm2", "vm1", "vm1", "vm2"],
        "min_cpu": [1.0, 2.0, 1.0, 3.0, 4.0],
        "max_cpu": [10.0, 20.0, 10.0, 30.0, 40.0],
        "avg_cpu": [5.0, 15.0, 5.0, 10.0, 20.0],
    }
    df = pd.DataFrame(data)

    targets = build_aggregate_targets(df)

    assert list(targets.columns) == ["active_vm_count", "aggregate_cpu_utilization"]
    assert targets.index.name == "timestamp"
    assert list(targets.index) == [0, 300]

    # Timestamp 0: distinct VMs = {'vm1', 'vm2'} -> 2
    # avg_cpu mean = (5.0 + 15.0 + 5.0) / 3 = 25.0 / 3
    assert targets.loc[0, "active_vm_count"] == 2
    assert np.isclose(targets.loc[0, "aggregate_cpu_utilization"], 25.0 / 3)

    # Timestamp 300: distinct VMs = {'vm1', 'vm2'} -> 2
    # avg_cpu mean = (10.0 + 20.0) / 2 = 15.0
    assert targets.loc[300, "active_vm_count"] == 2
    assert np.isclose(targets.loc[300, "aggregate_cpu_utilization"], 15.0)


def test_build_aggregate_targets_timestamp_index():
    data = {
        "vm_id": ["vm1", "vm2"],
        "avg_cpu": [10.0, 20.0],
    }
    df = pd.DataFrame(data, index=pd.Index([0, 0], name="timestamp"))
    targets = build_aggregate_targets(df)
    assert targets.loc[0, "active_vm_count"] == 2
    assert targets.loc[0, "aggregate_cpu_utilization"] == 15.0


def test_build_aggregate_targets_empty():
    empty_df = pd.DataFrame(columns=["timestamp", "vm_id", "avg_cpu"])
    targets = build_aggregate_targets(empty_df)
    assert targets.empty
    assert list(targets.columns) == ["active_vm_count", "aggregate_cpu_utilization"]
    assert targets.index.name == "timestamp"


def test_build_aggregate_targets_missing_columns():
    df = pd.DataFrame({"timestamp": [0], "vm_id": ["vm1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_aggregate_targets(df)


def test_train_test_split_standard():
    n_total = 134
    n_train = 110
    n_holdout = 24
    timestamps = [i * 300 for i in range(n_total)]
    df = pd.DataFrame(
        {
            "active_vm_count": np.full(n_total, 10000),
            "aggregate_cpu_utilization": np.random.uniform(5.0, 15.0, n_total),
        },
        index=pd.Index(timestamps, name="timestamp"),
    )

    train, holdout = train_test_split(df, n_train=n_train, n_holdout=n_holdout)

    assert len(train) == 110
    assert len(holdout) == 24
    assert train.index[0] == 0
    assert train.index[-1] == 109 * 300
    assert holdout.index[0] == 110 * 300
    assert holdout.index[-1] == 133 * 300


def test_train_test_split_unsorted_index():
    timestamps = [300, 0, 600, 900]
    df = pd.DataFrame(
        {
            "active_vm_count": [1, 2, 3, 4],
            "aggregate_cpu_utilization": [10.0, 20.0, 30.0, 40.0],
        },
        index=pd.Index(timestamps, name="timestamp"),
    )

    train, holdout = train_test_split(df, n_train=2, n_holdout=2)
    assert list(train.index) == [0, 300]
    assert list(holdout.index) == [600, 900]


def test_train_test_split_validation_errors():
    df = pd.DataFrame({"active_vm_count": [1, 2]}, index=[0, 300])

    with pytest.raises(ValueError, match="positive integers"):
        train_test_split(df, n_train=0, n_holdout=2)

    with pytest.raises(ValueError, match="positive integers"):
        train_test_split(df, n_train=2, n_holdout=-1)

    with pytest.raises(ValueError, match="expected at least"):
        train_test_split(df, n_train=2, n_holdout=1)


def test_load_cpu_readings_and_metadata_mock(tmp_path: Path):
    cpu_csv = tmp_path / "vm_cpu_readings.csv.gz"
    meta_csv = tmp_path / "vm_metadata.csv.gz"

    cpu_data = "timestamp,vm_id,min_cpu,max_cpu,avg_cpu\n300,vm2,1.0,10.0,5.0\n0,vm1,2.0,20.0,10.0\n"
    meta_data = "vm_id,subscription_id,deployment_id,vm_category\nvm2,s1,d1,Interactive\nvm1,s1,d1,Delay-insensitive\n"

    with gzip.open(cpu_csv, "wt", encoding="utf-8") as f:
        f.write(cpu_data)

    with gzip.open(meta_csv, "wt", encoding="utf-8") as f:
        f.write(meta_data)

    # Test loading from directory
    cpu_df = load_cpu_readings(tmp_path)
    meta_df = load_metadata(tmp_path)

    # Verify sorting
    assert list(cpu_df["timestamp"]) == [0, 300]
    assert list(meta_df["vm_id"]) == ["vm1", "vm2"]

    # Test loading direct file path
    cpu_df_direct = load_cpu_readings(cpu_csv)
    assert len(cpu_df_direct) == 2

    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_cpu_readings(tmp_path / "nonexistent")
