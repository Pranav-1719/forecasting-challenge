from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.forecasting import (
    MovingAverageModel,
    PipelineResult,
    create_features,
    expanding_window_cv,
    forecast_recursive,
    get_feature_columns,
    moving_average_forecast,
    run_pipeline,
    train_lightgbm_model,
)


@pytest.fixture
def sample_targets_df() -> pd.DataFrame:
    np.random.seed(42)
    t = np.arange(134)
    y_cpu = 9.0 + np.sin(t * 0.1) * 0.5 + np.random.normal(0, 0.05, size=134)
    y_vms = np.full(134, 10000.0)
    return pd.DataFrame({
        "timestamp": t * 300,
        "active_vm_count": y_vms,
        "aggregate_cpu_utilization": y_cpu,
    })


def test_create_features_structure_and_presence(sample_targets_df: pd.DataFrame):
    df = create_features(sample_targets_df, target_col="aggregate_cpu_utilization", max_lag=24)
    
    # 1. Minimum 12 lags up to max_lag
    for lag in range(1, 25):
        assert f"lag_{lag}" in df.columns

    # 2. Rolling statistics for windows 3, 6, 12, 24
    for w in [3, 6, 12, 24]:
        assert f"rolling_mean_{w}" in df.columns
        assert f"rolling_std_{w}" in df.columns

    # 3. Rate of change and difference features
    for step in [1, 3, 6, 12]:
        assert f"rate_of_change_{step}" in df.columns
        assert f"diff_{step}" in df.columns

    # 4. Position feature
    assert "position" in df.columns
    assert df["position"].iloc[0] == 0.0
    assert df["position"].iloc[-1] == len(df) - 1


def test_create_features_strictly_no_data_leakage(sample_targets_df: pd.DataFrame):
    """Verify that features at row t do not change if row t's target value changes."""
    df1 = sample_targets_df.copy()
    df2 = sample_targets_df.copy()
    
    # Change target value at index 50 in df2
    df2.loc[50, "aggregate_cpu_utilization"] += 100.0
    
    feat1 = create_features(df1, target_col="aggregate_cpu_utilization", max_lag=24)
    feat2 = create_features(df2, target_col="aggregate_cpu_utilization", max_lag=24)
    
    feature_cols = get_feature_columns(feat1.columns)
    
    # Features at row 50 MUST be identical between feat1 and feat2
    for col in feature_cols:
        val1 = feat1.loc[50, col]
        val2 = feat2.loc[50, col]
        assert np.isclose(val1, val2, equal_nan=True), f"Leakage detected in {col} at index 50"

    # Only row 51 and beyond should reflect the change at row 50
    assert feat2.loc[51, "lag_1"] != feat1.loc[51, "lag_1"]


def test_create_features_from_series():
    s = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], name="cpu")
    feat_df = create_features(s, target_col="cpu", max_lag=12)
    assert isinstance(feat_df, pd.DataFrame)
    assert "cpu" in feat_df.columns
    assert "lag_1" in feat_df.columns
    assert "rolling_mean_3" in feat_df.columns
    assert "position" in feat_df.columns


def test_expanding_window_cv_splits(sample_targets_df: pd.DataFrame):
    train_df = sample_targets_df.iloc[:110]
    splits = expanding_window_cv(
        train_df,
        target_col="aggregate_cpu_utilization",
        min_train=50,
        step=12,
        horizon=24,
    )
    
    assert len(splits) == 4
    
    # Fold 0: 0..49 train (50), 50..73 val (24)
    assert len(splits[0][0]) == 50
    assert len(splits[0][1]) == 24
    assert splits[0][0][-1] == 49
    assert splits[0][1][0] == 50
    assert splits[0][1][-1] == 73

    # Fold 1: expands by 12 points
    assert len(splits[1][0]) == 62
    assert len(splits[1][1]) == 24
    assert splits[1][0][-1] == 61
    assert splits[1][1][0] == 62
    assert splits[1][1][-1] == 85

    # Fold 3: ends at 109
    assert splits[3][1][-1] == 109

    # Verify no overlap between train and val within fold
    for tr, val in splits:
        assert set(tr).isdisjoint(set(val))
        assert tr[-1] < val[0]


def test_train_lightgbm_model_and_early_stopping(sample_targets_df: pd.DataFrame):
    feat_df = create_features(sample_targets_df.iloc[:80], target_col="aggregate_cpu_utilization")
    cols = get_feature_columns(feat_df.columns)
    
    X_train = feat_df[cols].iloc[:60]
    y_train = feat_df["aggregate_cpu_utilization"].iloc[:60]
    X_val = feat_df[cols].iloc[60:]
    y_val = feat_df["aggregate_cpu_utilization"].iloc[60:]
    
    # Model without val set
    model_simple = train_lightgbm_model(X_train, y_train, n_estimators=20)
    assert hasattr(model_simple, "predict")
    
    # Model with val set (early stopping)
    model_early = train_lightgbm_model(X_train, y_train, X_val=X_val, y_val=y_val, n_estimators=50)
    assert hasattr(model_early, "predict")
    preds = model_early.predict(X_val)
    assert len(preds) == len(X_val)


def test_moving_average_model_and_forecast(sample_targets_df: pd.DataFrame):
    y = sample_targets_df["aggregate_cpu_utilization"].iloc[:50]
    
    model = MovingAverageModel(window=12)
    model.fit(y=y)
    preds = model.predict(np.zeros(24))
    assert len(preds) == 24
    expected_mean = float(y.iloc[-12:].mean())
    assert np.isclose(preds[0], expected_mean)
    
    direct_preds = moving_average_forecast(y, horizon=24, window=12)
    assert len(direct_preds) == 24
    assert np.allclose(preds, direct_preds)


def test_forecast_recursive_modes(sample_targets_df: pd.DataFrame):
    train_df = sample_targets_df.iloc[:110]
    feat_df = create_features(train_df, target_col="aggregate_cpu_utilization")
    cols = get_feature_columns(feat_df.columns)
    
    model = train_lightgbm_model(feat_df[cols], feat_df["aggregate_cpu_utilization"], n_estimators=30)
    
    # Mode 1: Passing history dataframe
    preds1 = forecast_recursive(
        model=model,
        last_known_features=train_df,
        horizon=24,
        feature_creator_fn=create_features,
        target_col="aggregate_cpu_utilization",
    )
    assert isinstance(preds1, np.ndarray)
    assert len(preds1) == 24
    assert np.all(np.isfinite(preds1))
    assert np.all(preds1 >= 0.0)
    assert np.all(preds1 <= 100.0)

    # Mode 2: Passing feature row
    preds2 = forecast_recursive(
        model=model,
        last_known_features=feat_df[cols].iloc[-1:],
        horizon=24,
        target_col="aggregate_cpu_utilization",
    )
    assert isinstance(preds2, np.ndarray)
    assert len(preds2) == 24
    assert np.all(np.isfinite(preds2))


def test_run_pipeline_end_to_end(sample_targets_df: pd.DataFrame):
    result = run_pipeline(
        targets_df=sample_targets_df,
        target_col="aggregate_cpu_utilization",
        n_train=110,
        n_holdout=24,
    )
    
    # Check PipelineResult unpacking and attribute access
    assert isinstance(result, tuple)
    assert isinstance(result, PipelineResult)
    preds, cv_metrics = result
    
    assert len(preds) == 24
    assert preds.name == "aggregate_cpu_utilization"
    assert len(preds.index) == 24
    assert preds.index[0] == sample_targets_df["timestamp"].iloc[110]
    assert preds.index[-1] == sample_targets_df["timestamp"].iloc[133]
    
    assert "folds" in cv_metrics
    assert len(cv_metrics["folds"]) == 4
    assert "lgb_mae_mean" in cv_metrics
    assert "baseline_mae_mean" in cv_metrics
    assert "normalized_mae_mean" in cv_metrics
    assert cv_metrics["lgb_mae_mean"] > 0.0
    assert cv_metrics["baseline_mae_mean"] > 0.0


def test_run_pipeline_active_vm_count(sample_targets_df: pd.DataFrame):
    result = run_pipeline(
        targets_df=sample_targets_df,
        target_col="active_vm_count",
        n_train=110,
        n_holdout=24,
    )
    preds = result.predictions
    assert len(preds) == 24
    assert np.all(preds >= 0.0)
    assert np.allclose(preds.values, 10000.0)


def test_run_pipeline_with_timestamp_index(sample_targets_df: pd.DataFrame):
    df_indexed = sample_targets_df.set_index("timestamp")
    result = run_pipeline(
        targets_df=df_indexed,
        target_col="aggregate_cpu_utilization",
        n_train=110,
        n_holdout=24,
    )
    assert len(result.predictions) == 24
    assert list(result.predictions.index) == list(sample_targets_df["timestamp"].iloc[110:134])

