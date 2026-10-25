from __future__ import annotations

import inspect
from typing import Any, Callable, NamedTuple
import warnings

import numpy as np
import pandas as pd

from cwfc.metrics import mae, normalized_mae, rmse, smape

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False


FEATURE_PREFIXES = ("lag_", "rolling_", "rate_of_change_", "diff_")


class PipelineResult(NamedTuple):
    """Container for pipeline execution results.
    
    Supports both attribute access (result.predictions, result.cv_metrics)
    and tuple unpacking (predictions, cv_metrics = result).
    """
    predictions: pd.Series
    cv_metrics: dict[str, Any]


class MovingAverageModel:
    """Simple moving average forecasting model for time series comparison.
    
    Follows scikit-learn estimator interface (.fit and .predict).
    """

    def __init__(self, window: int = 12):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.mean_value_: float = 0.0

    def fit(self, X: Any = None, y: Any = None) -> MovingAverageModel:
        if y is None:
            raise ValueError("y cannot be None for MovingAverageModel.fit")
        vals = np.asarray(y, dtype=float)
        if len(vals) == 0:
            raise ValueError("y cannot be empty")
        effective_window = min(len(vals), self.window)
        self.mean_value_ = float(np.mean(vals[-effective_window:]))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.mean_value_, dtype=float)


def moving_average_forecast(
    history: pd.DataFrame | pd.Series | np.ndarray,
    horizon: int = 24,
    window: int = 12,
    target_col: str | None = None,
) -> np.ndarray:
    """Generate a constant moving average forecast over `horizon` steps.
    
    Parameters
    ----------
    history : pd.DataFrame, pd.Series, or np.ndarray
        Historical observations.
    horizon : int
        Number of steps to forecast ahead.
    window : int
        Number of most recent observations to average over.
    target_col : str, optional
        Target column name if history is a DataFrame.
        
    Returns
    -------
    np.ndarray
        Array of length `horizon` with the constant forecast value.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if window <= 0:
        raise ValueError("window must be positive")

    if isinstance(history, pd.DataFrame):
        if target_col is None:
            raise ValueError("target_col must be specified when history is a DataFrame")
        vals = history[target_col].astype(float).to_numpy()
    elif isinstance(history, pd.Series):
        vals = history.astype(float).to_numpy()
    else:
        vals = np.asarray(history, dtype=float)

    if len(vals) == 0:
        raise ValueError("history cannot be empty")

    effective_window = min(len(vals), window)
    mean_val = float(np.mean(vals[-effective_window:]))
    return np.full(horizon, mean_val, dtype=float)


def get_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return all column names that correspond to engineered features.
    
    Includes columns beginning with 'lag_', 'rolling_', 'rate_of_change_',
    'diff_', or exactly matching 'position'.
    """
    return [
        col for col in columns
        if col.startswith(FEATURE_PREFIXES) or col == "position"
    ]


def create_features(
    series: pd.DataFrame | pd.Series,
    target_col: str,
    max_lag: int = 24,
) -> pd.DataFrame:
    """Create tabular features from a univariate time series.
    
    Features created:
    - Lag features: `lag_1` through `lag_{max_lag}` (at minimum lag_1 to lag_12).
    - Rolling statistics: rolling mean and std for windows (3, 6, 12, 24).
    - Rate of change and difference features: 1, 3, 6, 12 steps.
    - Position/index feature: observation number along the timeline.
    
    Data Leakage Prevention:
    All features for row t are constructed strictly from observations prior
    to time t (using shift(1) or higher). No information at or after time t
    is used in the feature columns.
    
    Parameters
    ----------
    series : pd.DataFrame or pd.Series
        Input series or DataFrame containing `target_col`.
    target_col : str
        Name of the target column.
    max_lag : int, default=24
        Maximum lag to create (minimum 12).
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing original columns plus all engineered features.
    """
    if isinstance(series, pd.Series):
        df = pd.DataFrame({target_col: series.values}, index=series.index)
    elif isinstance(series, pd.DataFrame):
        df = series.copy()
    else:
        raise TypeError(f"series must be a pd.DataFrame or pd.Series, got {type(series)}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input series/DataFrame")

    s = df[target_col].astype(float)
    # Most recent past observation relative to time t is s.shift(1)
    s_lag1 = s.shift(1)

    # 1. Lag features (at minimum 1 through 12, up to max_lag)
    n_lags = max(int(max_lag), 12)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = s.shift(lag)

    # 2. Rolling statistics (windows 3, 6, 12, 24) computed on past data (s_lag1)
    for w in [3, 6, 12, 24]:
        df[f"rolling_mean_{w}"] = s_lag1.rolling(window=w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = (
            s_lag1.rolling(window=w, min_periods=2).std().fillna(0.0)
        )

    # 3. Rate of change and difference features
    for step in [1, 3, 6, 12]:
        prior_val = s.shift(1 + step)
        df[f"rate_of_change_{step}"] = (s_lag1 - prior_val) / (prior_val.abs() + 1e-8)
        df[f"diff_{step}"] = s_lag1 - prior_val

    # 4. Position feature (observation index along sequence)
    df["position"] = np.arange(len(df), dtype=float)

    return df


def expanding_window_cv(
    train_df: pd.DataFrame,
    target_col: str | None = None,
    min_train: int = 50,
    step: int = 12,
    horizon: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate time-aware expanding window cross-validation splits.
    
    Each fold expands the training window from `min_train` in increments of `step`,
    and sets the next `horizon` observations as the validation set.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data frame.
    target_col : str, optional
        Target column name (validated if present in train_df).
    min_train : int, default=50
        Minimum number of initial training points for fold 0.
    step : int, default=12
        Step size by which the training window expands each fold.
    horizon : int, default=24
        Validation forecast horizon (number of observations).
        
    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min_train <= 0:
        raise ValueError("min_train must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    n = len(train_df)
    if n < min_train + horizon:
        raise ValueError(
            f"train_df length ({n}) is too short for min_train ({min_train}) + horizon ({horizon})"
        )

    if target_col is not None and isinstance(train_df, pd.DataFrame):
        if target_col not in train_df.columns:
            warnings.warn(
                f"target_col '{target_col}' not found in train_df columns; continuing with index split.",
                UserWarning,
            )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    while train_end + horizon <= n:
        train_idx = np.arange(0, train_end, dtype=int)
        val_idx = np.arange(train_end, train_end + horizon, dtype=int)
        splits.append((train_idx, val_idx))
        train_end += step

    return splits


def train_lightgbm_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    n_estimators: int = 50,
    max_depth: int = 3,
    num_leaves: int = 7,
    learning_rate: float = 0.05,
    min_child_samples: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 10,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """Train a LightGBM regressor with defaults tailored for short time series.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : pd.Series or np.ndarray
        Training target values.
    X_val : pd.DataFrame or np.ndarray, optional
        Validation feature matrix for early stopping.
    y_val : pd.Series or np.ndarray, optional
        Validation target values for early stopping.
    n_estimators : int, default=50
        Number of boosting trees.
    max_depth : int, default=3
        Maximum tree depth (shallow depth to prevent overfitting on short series).
    num_leaves : int, default=7
        Maximum number of leaves per tree.
    learning_rate : float, default=0.05
        Boosting learning rate.
    min_child_samples : int, default=5
        Minimum number of samples required per leaf.
    subsample : float, default=0.8
        Row subsampling ratio (bagging fraction).
    colsample_bytree : float, default=0.8
        Feature subsampling ratio.
    early_stopping_rounds : int, default=10
        Early stopping rounds if validation data is provided.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : Any
        Additional keyword arguments passed to LGBMRegressor.
        
    Returns
    -------
    lgb.LGBMRegressor
        Trained LightGBM model.
    """
    if not HAS_LIGHTGBM:
        raise ImportError(
            "LightGBM is required for train_lightgbm_model. Please install lightgbm."
        )

    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": 1 if subsample < 1.0 else 0,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "verbose": -1,
    }
    model_params.update(kwargs)

    model = lgb.LGBMRegressor(**model_params)

    has_val = (
        X_val is not None
        and y_val is not None
        and len(X_val) > 0
        and len(y_val) > 0
    )

    if has_val:
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        fit_params = inspect.signature(model.fit).parameters
        if "eval_X" in fit_params:
            model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks=callbacks)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    else:
        model.fit(X_train, y_train)

    return model


def _enforce_domain_bounds(value: float, target_col: str | None) -> float:
    """Enforce domain constraints based on the challenge target contract."""
    if target_col == "aggregate_cpu_utilization":
        return float(min(100.0, max(0.0, value)))
    if target_col == "active_vm_count":
        return float(max(0.0, value))
    return float(value)


def forecast_recursive(
    model: Any,
    last_known_features: pd.DataFrame | pd.Series | np.ndarray,
    horizon: int,
    feature_creator_fn: Callable | None = None,
    target_col: str | None = None,
) -> np.ndarray:
    """Generate multi-step recursive forecasts.
    
    At each forecast step, the model predicts one step ahead. The predicted value
    is then fed back to update lag features, rolling statistics, rate of change,
    and position for the next step.
    
    Parameters
    ----------
    model : Any
        Fitted forecasting model with a .predict(X) method.
    last_known_features : pd.DataFrame, pd.Series, or np.ndarray
        Historical observations or feature table/row up to forecast origin.
    horizon : int
        Number of steps to forecast into the future.
    feature_creator_fn : Callable, optional
        Function to create features from series (e.g., `create_features`).
        If provided, features for each step are dynamically generated.
    target_col : str, optional
        Name of the target column being forecast.
        
    Returns
    -------
    np.ndarray
        Array of length `horizon` containing recursive predictions.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    forecasts: list[float] = []

    # Check if last_known_features contains full target history
    is_history_df = (
        isinstance(last_known_features, pd.DataFrame)
        and target_col is not None
        and target_col in last_known_features.columns
    )
    is_history_series = (
        isinstance(last_known_features, pd.Series)
        and not any(str(idx).startswith("lag_") for idx in last_known_features.index)
    )

    if is_history_df or is_history_series:
        if is_history_df:
            assert target_col is not None
            history_vals = list(last_known_features[target_col].astype(float).values)
        else:
            history_vals = list(last_known_features.astype(float).values)
            if target_col is None:
                target_col = str(last_known_features.name or "target")

        creator = feature_creator_fn if feature_creator_fn is not None else create_features
        current_series = pd.DataFrame({target_col: history_vals})
        feature_names = getattr(model, "feature_name_", None)

        for _ in range(horizon):
            # Append NaN placeholder for the next step to derive its features
            step_df = pd.concat(
                [current_series, pd.DataFrame({target_col: [np.nan]})],
                ignore_index=True,
            )
            all_feats = creator(step_df, target_col=target_col)

            if feature_names is not None:
                step_features = [c for c in feature_names if c in all_feats.columns]
            else:
                step_features = get_feature_columns(all_feats.columns)

            X_step = all_feats[step_features].iloc[[-1]]
            pred = float(model.predict(X_step)[0])
            pred = _enforce_domain_bounds(pred, target_col)

            forecasts.append(pred)
            current_series = pd.concat(
                [current_series, pd.DataFrame({target_col: [pred]})],
                ignore_index=True,
            )

    else:
        # last_known_features is a feature row or feature DataFrame
        if isinstance(last_known_features, pd.Series):
            feat_row = last_known_features.to_frame().T
        elif isinstance(last_known_features, pd.DataFrame):
            feat_row = last_known_features.iloc[[-1]].copy()
        else:
            raise TypeError(
                f"Unsupported type for last_known_features: {type(last_known_features)}"
            )

        feature_names = getattr(model, "feature_name_", None)
        if feature_names is not None:
            step_features = [c for c in feature_names if c in feat_row.columns]
        else:
            step_features = get_feature_columns(feat_row.columns)

        lag_cols = [c for c in feat_row.columns if c.startswith("lag_")]
        lag_nums = sorted([int(c.split("_")[1]) for c in lag_cols])

        if lag_nums:
            history_buffer = [
                float(feat_row[f"lag_{k}"].iloc[0]) for k in reversed(lag_nums)
            ]
        else:
            history_buffer = []

        pos = (
            float(feat_row["position"].iloc[0])
            if "position" in feat_row.columns
            else 0.0
        )

        curr_row = feat_row.copy()

        for _ in range(horizon):
            X_step = curr_row[step_features]
            pred = float(model.predict(X_step)[0])
            pred = _enforce_domain_bounds(pred, target_col)
            forecasts.append(pred)

            history_buffer.append(pred)
            pos += 1.0

            # Update lag features
            for k in reversed(lag_nums):
                if k == 1:
                    curr_row["lag_1"] = pred
                else:
                    curr_row[f"lag_{k}"] = curr_row[f"lag_{k-1}"]

            if "position" in curr_row.columns:
                curr_row["position"] = pos

            # Update rolling statistics
            buf = np.array(history_buffer, dtype=float)
            for w in [3, 6, 12, 24]:
                if f"rolling_mean_{w}" in curr_row.columns:
                    curr_row[f"rolling_mean_{w}"] = float(np.mean(buf[-w:]))
                if f"rolling_std_{w}" in curr_row.columns:
                    curr_row[f"rolling_std_{w}"] = (
                        float(np.std(buf[-w:], ddof=1)) if len(buf[-w:]) > 1 else 0.0
                    )

            # Update rate of change & diff features
            for s_step in [1, 3, 6, 12]:
                if len(buf) > s_step:
                    v_prev = buf[-(1 + s_step)]
                    roc = (pred - v_prev) / (abs(v_prev) + 1e-8)
                    diff = pred - v_prev
                else:
                    roc = 0.0
                    diff = 0.0
                if f"rate_of_change_{s_step}" in curr_row.columns:
                    curr_row[f"rate_of_change_{s_step}"] = roc
                if f"diff_{s_step}" in curr_row.columns:
                    curr_row[f"diff_{s_step}"] = diff

    return np.array(forecasts, dtype=float)


def run_pipeline(
    targets_df: pd.DataFrame,
    target_col: str,
    n_train: int = 110,
    n_holdout: int = 24,
    min_train: int = 50,
    cv_step: int = 12,
    cv_horizon: int = 24,
    model_params: dict[str, Any] | None = None,
) -> PipelineResult:
    """End-to-end forecasting pipeline: Feature Engineering -> CV -> Train -> Forecast.
    
    1. Slices the training window (first `n_train` observations).
    2. Runs expanding window cross-validation comparing LightGBM against a Moving Average baseline.
    3. Fits the final LightGBM model on the entire training set.
    4. Recursively forecasts the next `n_holdout` observations.
    5. Returns predictions and cross-validation performance metrics.
    
    Strictly prevents data leakage: holdout observations are never seen during feature
    generation, cross-validation, or model training.
    
    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing target series and timestamps.
    target_col : str
        Target column to forecast ('active_vm_count' or 'aggregate_cpu_utilization').
    n_train : int, default=110
        Number of training observations (first n_train rows).
    n_holdout : int, default=24
        Forecast horizon (number of holdout observations).
    min_train : int, default=50
        Minimum training window size for expanding window CV.
    cv_step : int, default=12
        Step size between expanding CV folds.
    cv_horizon : int, default=24
        Validation horizon size for each CV fold.
    model_params : dict, optional
        Custom hyperparameters for LightGBM.
        
    Returns
    -------
    PipelineResult (NamedTuple with .predictions and .cv_metrics)
        predictions : pd.Series
            Holdout forecast predictions (length n_holdout).
        cv_metrics : dict[str, Any]
            Cross-validation metrics for LightGBM and Moving Average baseline.
    """
    if not isinstance(targets_df, pd.DataFrame):
        raise TypeError("targets_df must be a pd.DataFrame")
    if target_col not in targets_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in targets_df columns")
    if len(targets_df) < n_train:
        raise ValueError(
            f"targets_df has {len(targets_df)} rows, but n_train={n_train} requested"
        )

    # 1. Separate training data (strict leakage prevention)
    train_df = targets_df.iloc[:n_train].copy().reset_index(drop=True)

    # 2. Engineer features on training data only
    train_features = create_features(train_df, target_col=target_col, max_lag=cv_horizon)
    feature_cols = get_feature_columns(train_features.columns)

    # 3. Expanding window cross-validation
    splits = expanding_window_cv(
        train_df,
        target_col=target_col,
        min_train=min_train,
        step=cv_step,
        horizon=cv_horizon,
    )

    params = model_params or {}
    fold_records: list[dict[str, Any]] = []
    lgb_maes: list[float] = []
    lgb_rmses: list[float] = []
    lgb_smapes: list[float] = []
    baseline_maes: list[float] = []
    baseline_rmses: list[float] = []
    baseline_smapes: list[float] = []
    normalized_maes: list[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        fold_train_feats = train_features.iloc[tr_idx]
        fold_val_feats = train_features.iloc[val_idx]

        X_tr = fold_train_feats[feature_cols]
        y_tr = fold_train_feats[target_col]
        X_va = fold_val_feats[feature_cols]
        y_va = fold_val_feats[target_col]

        # Train LightGBM model for fold
        fold_model = train_lightgbm_model(
            X_tr,
            y_tr,
            X_val=X_va,
            y_val=y_va,
            **params,
        )

        # Recursive forecast on validation horizon
        fold_train_history = train_df.iloc[tr_idx]
        fold_preds = forecast_recursive(
            model=fold_model,
            last_known_features=fold_train_history,
            horizon=len(val_idx),
            feature_creator_fn=create_features,
            target_col=target_col,
        )

        # Moving average baseline forecast
        baseline_preds = moving_average_forecast(
            history=y_tr,
            horizon=len(val_idx),
            window=min(12, len(y_tr)),
        )

        y_true = y_va.to_numpy(dtype=float)

        fold_lgb_mae = mae(y_true, fold_preds)
        fold_lgb_rmse = rmse(y_true, fold_preds)
        fold_lgb_smape = smape(y_true, fold_preds)

        fold_base_mae = mae(y_true, baseline_preds)
        fold_base_rmse = rmse(y_true, baseline_preds)
        fold_base_smape = smape(y_true, baseline_preds)

        fold_norm_mae = normalized_mae(y_true, fold_preds, baseline_preds)

        lgb_maes.append(fold_lgb_mae)
        lgb_rmses.append(fold_lgb_rmse)
        lgb_smapes.append(fold_lgb_smape)
        baseline_maes.append(fold_base_mae)
        baseline_rmses.append(fold_base_rmse)
        baseline_smapes.append(fold_base_smape)
        normalized_maes.append(fold_norm_mae)

        fold_records.append({
            "fold": fold_idx,
            "train_size": len(tr_idx),
            "val_size": len(val_idx),
            "lgb_mae": fold_lgb_mae,
            "lgb_rmse": fold_lgb_rmse,
            "lgb_smape": fold_lgb_smape,
            "baseline_mae": fold_base_mae,
            "baseline_rmse": fold_base_rmse,
            "baseline_smape": fold_base_smape,
            "normalized_mae": fold_norm_mae,
        })

    cv_summary: dict[str, Any] = {
        "target_col": target_col,
        "n_folds": len(splits),
        "lgb_mae_mean": float(np.mean(lgb_maes)) if lgb_maes else 0.0,
        "lgb_mae_std": float(np.std(lgb_maes)) if lgb_maes else 0.0,
        "lgb_rmse_mean": float(np.mean(lgb_rmses)) if lgb_rmses else 0.0,
        "lgb_smape_mean": float(np.mean(lgb_smapes)) if lgb_smapes else 0.0,
        "baseline_mae_mean": float(np.mean(baseline_maes)) if baseline_maes else 0.0,
        "baseline_rmse_mean": float(np.mean(baseline_rmses)) if baseline_rmses else 0.0,
        "baseline_smape_mean": float(np.mean(baseline_smapes)) if baseline_smapes else 0.0,
        "normalized_mae_mean": float(np.mean(normalized_maes)) if normalized_maes else 1.0,
        "folds": fold_records,
    }

    # 4. Train final model on full training set
    X_full_train = train_features[feature_cols]
    y_full_train = train_features[target_col]
    final_model = train_lightgbm_model(
        X_full_train,
        y_full_train,
        X_val=None,
        y_val=None,
        **params,
    )

    # 5. Holdout recursive forecast
    final_preds = forecast_recursive(
        model=final_model,
        last_known_features=train_df,
        horizon=n_holdout,
        feature_creator_fn=create_features,
        target_col=target_col,
    )

    # Format predictions Series
    if len(targets_df) >= n_train + n_holdout:
        if "timestamp" in targets_df.columns:
            holdout_index = targets_df.iloc[n_train : n_train + n_holdout]["timestamp"].values
        else:
            holdout_index = targets_df.index[n_train : n_train + n_holdout]
        preds_series = pd.Series(final_preds, index=holdout_index, name=target_col)
    else:
        preds_series = pd.Series(
            final_preds,
            index=range(n_train, n_train + n_holdout),
            name=target_col,
        )

    return PipelineResult(predictions=preds_series, cv_metrics=cv_summary)
