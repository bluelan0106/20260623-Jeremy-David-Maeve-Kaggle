# S&P 500 Forward Excess Return Prediction

A walk-forward machine-learning pipeline for forecasting S&P 500 forward excess returns and converting the predictions into position-sized trading strategies. The project sweeps **3 models × 5 feature datasets × 2 feature-selection modes × 2 loss functions = 60 prediction configurations**, then turns each into Vanilla and Kelly strategies for backtesting.

> Kaggle competition: *Hull Tactical – Market Prediction* (forward excess return forecasting).

---

## Pipeline Overview

```
base_processed.csv
        │
        ▼
01_pipeline_feature_engineering.py   → 5 feature datasets
        ▼
02_pipeline_model_predictions.py     → 60-column prediction matrix (walk-forward)
        ▼
03_pipeline_signal_and_backtest.py   → 120 strategies + metrics
        ▼
04_pipeline_evaluation_plots.py      → 28 evaluation plots
```

| Stage | Script | Output |
|-------|--------|--------|
| 1. Feature engineering | `01_pipeline_feature_engineering.py` | `pipeline_data/datasets/dataset_1~5.csv` |
| 2. Rolling prediction | `02_pipeline_model_predictions.py` | `pipeline_data/predictions/all_models_predictions_matrix.csv` |
| 3. Signal + backtest | `03_pipeline_signal_and_backtest.py` | `pipeline_data/backtest/*.csv` |
| 4. Evaluation plots | `04_pipeline_evaluation_plots.py` | `pipeline_data/evaluation/**/*.png` |

The four scripts are **sequential** — each consumes the previous stage's output. Pipeline 02 supports checkpoint resume (completed columns are skipped, so an interrupted run can be restarted directly).

Full design rationale, hyperparameters, and lookahead-bias handling are documented in **[`Pipeline_Architecture.md`](Pipeline_Architecture.md)**.

---

## Methodology Highlights

- **Walk-forward, no folds.** 750-day fixed training window rolling 1 day at a time; out-of-sample prediction on the most recent 180 days.
- **Lookahead-bias isolation.** Since target `Y` (forward excess return) for day `t` settles at `t+1` close, training is cut off at `t-2`. See `Pipeline_Architecture.md §四` for the time-isolation proof.
- **Dynamic feature selection** via Permutation Importance (model-agnostic), recomputed every 20 days, keeping Top-30 features.
- **Two position-sizing strategies:**
  - *Vanilla* — static min-max scaling of predictions into a `[0, 2]` leverage range.
  - *Kelly* — raw Kelly sized by `ŷ / (ATR² + ε)`, with a breadth-conviction penalty and 3/4-Kelly cap.
- **Benchmark:** Buy-and-Hold.

---

## Models

| Model | Library | Loss variants |
|-------|---------|---------------|
| LSTM | TensorFlow / Keras | MSE, Huber |
| XGBoost | xgboost | MSE, pseudo-Huber (dynamic delta) |
| REG | scikit-learn / scipy | LinearRegression, PseudoHuberRegressor |

---

## Project Structure

```
.
├── 01_pipeline_feature_engineering.py
├── 02_pipeline_model_predictions.py
├── 03_pipeline_signal_and_backtest.py
├── 04_pipeline_evaluation_plots.py
├── robustness_checks.ipynb            # DSR / SPA / PBO robustness tests
├── Pipeline_Architecture.md           # full design document (zh-TW)
├── feature_engineering/
│   └── datasets/base_processed.csv    # raw input (8042 × 120)
└── pipeline_data/
    ├── datasets/                      # 5 engineered feature datasets
    ├── predictions/                   # prediction matrix + PI feature logs
    ├── backtest/                      # daily returns, positions, metrics
    └── evaluation/                    # equity_curves/ + prediction_signals/
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
scipy
scikit-learn
xgboost
tensorflow
matplotlib
```

> TensorFlow is required for the LSTM model. On Apple Silicon, install `tensorflow-macos` (+ optionally `tensorflow-metal`) instead of `tensorflow`.

---

## Usage

Run the stages in order from the project root:

```bash
python 01_pipeline_feature_engineering.py
python 02_pipeline_model_predictions.py     # longest stage; resumable
python 03_pipeline_signal_and_backtest.py
python 04_pipeline_evaluation_plots.py
```

Outputs land under `pipeline_data/`. Metrics for all 120 strategies are written to `pipeline_data/backtest/strategy_metrics_comparison.csv` (columns: `Model, Strategy, Sharpe_Ratio, MDD, Total_Return`).

---

## Column Naming Convention

```
{MODEL}_ds{N}_fs{Y/N}_{loss}

LSTM_ds1_fsY_huber   → LSTM, dataset 1, with FS, Huber loss
XGB_ds3_fsN_mse      → XGBoost, dataset 3, no FS, MSE loss
REG_ds5_fsY_huber    → Regression, dataset 5, with FS, Huber loss
```

Strategy columns append the sizing method: `..._vanilla` / `..._kelly`.

---

## Authors

Jeremy · David · Maeve — HWTeng FIMA Lab, NYCU.
