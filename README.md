# Cloud Workload Forecasting Challenge

AI/ML take-home challenge built from the Microsoft Azure Public Dataset V2 VM trace.

## Quick Start

### 1. Environment Setup

Python 3.11+ is required.

```bash
# Create virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install lightgbm scikit-learn matplotlib statsmodels
```

### 2. Dataset

Download the `data-v1` release and extract into `challenge_data/`:

```bash
# Download from: https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
# Extract:
tar -xzf data-v1.tar.gz -C challenge_data
```

Expected layout:
```
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

### 3. Verify Installation

```bash
python -m pytest -q
```

### 4. Generate Predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This single command:
1. Loads the VM CPU readings and metadata
2. Constructs aggregate target time series
3. Splits into train (110 timestamps) and holdout (24 timestamps)
4. Trains baseline + LightGBM models with time-aware validation
5. Generates `predictions.csv` with 24 holdout predictions

### 5. Run EDA Analysis

```bash
python notebooks/01_eda.py
```

## Project Structure

```
├── challenge_data/          # Dataset (from data-v1 release)
├── notebooks/
│   └── 01_eda.py           # Exploratory data analysis
├── reports/
│   └── final_report.md     # 4-6 page analysis report
├── src/cwfc/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # Module entry point
│   ├── baselines.py         # Persistence & seasonal naive baselines
│   ├── data_loader.py       # Data loading & target construction
│   ├── forecasting.py       # Feature engineering & LightGBM pipeline
│   ├── metrics.py           # MAE, RMSE, sMAPE, normalized MAE
│   ├── predict.py           # Main prediction pipeline
│   └── validation.py        # Rolling-origin temporal validation
├── tests/
│   ├── test_metrics.py      # Metric unit tests
│   ├── test_validation.py   # Validation unit tests
│   ├── test_data_loader.py  # Data loader and split unit tests
│   └── test_forecasting.py  # Feature engineering & ML pipeline unit tests
├── predictions.csv          # Generated predictions (24 rows)
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Modeling Decisions

### Targets
- **`active_vm_count`**: Count of distinct VMs per timestamp. Nearly constant (~10,000) since all VMs have complete traces.
- **`aggregate_cpu_utilization`**: Mean of VM-level avg_cpu per timestamp. Shows temporal variation suitable for forecasting.

### Approach
1. **Baselines**: Persistence (last value), Moving Average (12-step window), Seasonal Naive (24-step pattern)
2. **Main model**: LightGBM with lag features, rolling statistics, and rate-of-change features
3. **Validation**: Expanding-window cross-validation respecting temporal ordering
4. **Forecasting**: Recursive multi-step prediction (each step feeds back as input)

### Key Design Choices
- **Shallow trees (max_depth=3)**: Only 110 training observations — deep models would overfit
- **Lag-based features only**: Timestamps are anonymized, so no calendar features
- **Automatic model selection**: Uses the better of LightGBM vs. persistence baseline per target
- **No data leakage**: All features use only data available at forecast time

## Source

Microsoft Azure Public Dataset V2: https://github.com/Azure/AzurePublicDataset
