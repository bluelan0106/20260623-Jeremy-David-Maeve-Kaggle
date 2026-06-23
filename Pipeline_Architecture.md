# 股價預測模型 Pipeline 架構文件 (V6)

## 一、執行順序總覽

```
01_pipeline_feature_engineering.py
        ↓ 產出：pipeline_data/datasets/*.csv（5個資料集）
02_pipeline_model_predictions.py
        ↓ 產出：pipeline_data/predictions/all_models_predictions_matrix.csv
03_pipeline_signal_and_backtest.py
        ↓ 產出：pipeline_data/backtest/all_strategies_daily_returns.csv
                pipeline_data/backtest/strategy_metrics_comparison.csv
04_pipeline_evaluation_plots.py
        ↓ 產出：pipeline_data/evaluation/equity_curves/*.png
                pipeline_data/evaluation/prediction_signals/*.png
```

---

## 二、各 Pipeline 說明

---

### Pipeline 01｜特徵工程
**檔案：** `01_pipeline_feature_engineering.py`
**輸入：** `../feature_engineering/datasets/base_processed.csv`
**輸出：** `pipeline_data/datasets/dataset_1~5.csv`

#### 特徵群組分類
| 群組 | 前綴 | 轉換方式 |
|------|------|---------|
| Group A | M, P, V, E | Pct Change `(t - t-1) / t-1` |
| Group B | S, I | Diff `t - (t-1)` |
| Group C | D, `_is_missing` | 原始值（不轉換） |

#### 5 個資料集組成
| 資料集 | 內容 |
|--------|------|
| dataset_1_Base | 原始特徵 X |
| dataset_2_Delta | 變化率特徵 ΔX |
| dataset_3_Historical | 歷史報酬特徵 R（lag 1~60 + cumsum 1~60） |
| dataset_4_Original_plus_Hist | X + R |
| dataset_5_Delta_plus_Hist | ΔX + R |

#### Meta 欄位（不進入訓練，僅供 Pipeline 03 使用）
- `meta_atr`：`max(std_5d, std_20d)` of realized returns + ε
- `meta_spread`：`Z_252(slow_momentum) - Z_252(fast_momentum)`

---

### Pipeline 02｜滾動預測（60 種組合）
**檔案：** `02_pipeline_model_predictions.py`
**輸入：** `pipeline_data/datasets/*.csv`
**輸出：** `pipeline_data/predictions/all_models_predictions_matrix.csv`
　　　　`pipeline_data/predictions/metrics_comparison_summary.csv`

#### 實驗設計
```
3 models × 5 datasets × 2 (FS/No FS) × 2 (huber/mse) = 60 種組合
```

#### 模型與超參數
| 模型 | 預設超參數 | Huber Loss |
|------|-----------|-----------|
| LSTM | units=32, dropout=0.1, lr=1e-3, epochs=20, batch=256 | `tf.keras.losses.Huber()` |
| XGBoost | n_estimators=300, max_depth=3, lr=0.05 | `reg:pseudohubererror`，delta=`2.5 × std(y_train)`（動態） |
| REG | LinearRegression / PseudoHuberRegressor | L-BFGS-B 優化，delta=`2.5 × std(y_train)`（動態） |

#### Walk-Forward 設計
```
訓練窗口：750 天（固定長度，每天滾動 1 天）
OOS 預測：最後 180 天
測試：單次預測，無 fold，無 EarlyStopping
```

#### 時間隔離設計（Lookahead Bias 防護）

`market_forward_excess_returns`（目標變數 Y）的定義：
- `date_id = t` 這列的 Y = 第 t 天買入、第 t+1 天賣出的超額報酬
- 因此 Y 的值必須等到 **t+1 收盤**後才能確認

這導致一個隱性的 Lookahead Bias 風險：
若訓練窗口截止到 `date_id < current_date`，則最後一筆訓練資料
的 Y（`current_date - 1` 的報酬）需等到 `current_date` 收盤才知道，
但模型在 `current_date` 開盤前就使用了這筆資料，構成資料洩漏。

**正確的時間隔離方式：**

```
date_id:   t-751    ...    t-2    t-1  |  t
                                       |
訓練窗口： [t-751, t-2]                |
  每列 X = 當天的 features             |
  每列 Y = 當天的 forward_return       |
           （隔天收盤才確認，          |
             最後一筆 t-2 的 Y         |
             在 t-1 收盤時已知）     |
                                       |
OOS 預測目標：                          t 這列
  輸入：t 當天的 X（開盤前已知）     |
  預測：t 的 Y = t→t+1 的超額報酬     |
        （今天決策，明天結算）        |
```

**程式碼對應：**
```python
# 訓練窗口：共 750 筆，Y 值全部已知
train_window = df[
    (df[ID_COL] >= current_date - ROLLING_WINDOW - 1) &
    (df[ID_COL] <  current_date - 1)   # 截止到 t-2，避免用到未確認的 Y
]

# OOS 預測：用 t 當天的 X 預測 t 的 Y
test_row = df[df[ID_COL] == current_date]
```

**LSTM 測試序列的額外注意：**

LSTM 需要連續 TIMESTEPS（60）天的 X 形成序列，取的是：
```python
df[(df[ID_COL] >= current_date - TIMESTEPS + 1) &
   (df[ID_COL] <= current_date)]
```
這 60 筆全部只使用 X（features），不涉及 current_date 的 Y，無洩漏風險 

#### 動態 Feature Selection（PI）
```
方法：Permutation Importance（跨樣本 shuffle）
觸發：第 1 天 + 每 20 天重跑一次
PI 模型：與預測主模型完全相同架構（非輕量版），確保重要性評估對齊主模型
保留數量：Top 30 features（固定 K）
無 FS 組：使用全部 features
```

#### 欄位命名規則
```
{MODEL}_ds{N}_fs{Y/N}_{loss}

範例：
  LSTM_ds1_fsY_huber   ← LSTM，Dataset 1，有 FS，Huber loss
  XGB_ds3_fsN_mse      ← XGBoost，Dataset 3，無 FS，MSE loss
  REG_ds5_fsY_huber    ← Regression，Dataset 5，有 FS，Huber loss
```

#### 斷點續傳
已完成的欄位自動跳過，中斷後可直接重新執行。

---

### Pipeline 03｜訊號轉換與回測
**檔案：** `03_pipeline_signal_and_backtest.py`
**輸入：** `pipeline_data/predictions/all_models_predictions_matrix.csv`
　　　　`pipeline_data/datasets/*.csv`（讀取 meta_atr、meta_spread）
**輸出：** `pipeline_data/backtest/all_strategies_daily_returns.csv`
　　　　`pipeline_data/backtest/strategy_metrics_comparison.csv`

#### 策略組合
```
60 預測欄 × 2 策略（Vanilla / Kelly）= 120 個策略欄位
+ Buy and Hold 基準線
```

#### Strategy 1：Vanilla（靜態 Min-Max Scaling）
```
f = 2.0 × (pred - min) / (max - min)
f = clamp(0, 2)
daily_return = f × actual_return
```
- min/max 來自測試期前 **750 天**的訓練集真實報酬，截止到 `test_start_id - 2`（與 Pipeline 02 採用相同時間隔離原則，`test_start_id - 1` 的 Y 需到測試期第一天收盤才確認，不得使用）

#### Strategy 2：Kelly Engine
```
# Step 1: Raw Kelly
f_raw = ŷ / (ATR² + ε)

# Step 2: Breadth Conviction Penalty
kappa = 0.5 + 0.5 × (1 - sigmoid(3.0 × (BS_t - 0.5)))

# Step 3: Final Position
f_final = clamp(0, 2, f_raw × 0.75 × kappa)

daily_return = f_final × actual_return
```

#### Kelly 參數
| 參數 | 值 | 說明 |
|------|-----|------|
| λ (LAMBDA_PARAM) | 0.75 | 3/4-Kelly，網格測試最佳 Sharpe |
| k (K_TAIL) | 3.0 | Sigmoid 斜率，避免懸崖效應 |
| θ (THETA) | 0.5 | Breadth Spread 啟動門檻 |

#### 回測指標
| 指標 | 計算方式 |
|------|---------|
| Sharpe Ratio | 年化（× √252） |
| Total Return | 複利 cumprod |
| Max Drawdown | 基於累積權益曲線 |

---

### Pipeline 04｜視覺化評估
**檔案：** `04_pipeline_evaluation_plots.py`
**輸入：** `pipeline_data/predictions/all_models_predictions_matrix.csv`
　　　　`pipeline_data/backtest/all_strategies_daily_returns.csv`
**輸出：** `pipeline_data/evaluation/equity_curves/*.png`
　　　　`pipeline_data/evaluation/prediction_signals/*.png`

#### A. Equity Curves（策略回測圖）
| 子組 | 比較維度 | 固定變數 | 張數 |
|------|---------|---------|------|
| A1 | Feature Selection（Y vs N） | loss × strat | 4 |
| A2 | Loss（huber vs mse） | fs × strat | 4 |
| A3 | Strategy（kelly vs vanilla） | fs × loss | 4 |
| A4 | Model（LSTM / XGB / REG） | fs × loss × strat | 8 |
| **合計** | | | **20 張** |

#### B. Prediction Signals（原始預測值圖）
| 子組 | 比較維度 | 固定變數 | 張數 |
|------|---------|---------|------|
| B1 | Feature Selection（Y vs N） | loss | 2 |
| B2 | Loss（huber vs mse） | fs | 2 |
| B3 | Model（LSTM / XGB / REG） | fs × loss | 4 |
| **合計** | | | **8 張** |

#### 圖表規格
- 每張圖：1×5 橫向排列（Dataset 1~5）
- 實線（Solid）= 實驗組，虛線（Dashed）= 控制組
- 灰色虛線 = Buy and Hold 基準
- 顏色：LSTM 藍、XGB 橘、REG 綠

---

## 三、資料流總覽

```
base_processed.csv
        │
        ▼
[Pipeline 01] 特徵工程
        │
        ├── dataset_1_Base.csv
        ├── dataset_2_Delta.csv
        ├── dataset_3_Historical.csv
        ├── dataset_4_Original_plus_Hist.csv
        └── dataset_5_Delta_plus_Hist.csv
                │
                ▼
        [Pipeline 02] 滾動預測（60 種組合）
                │  Walk-Forward 750天
                │  PI Feature Selection 每20天
                │
                └── all_models_predictions_matrix.csv（60欄）
                        │
                        ▼
                [Pipeline 03] 訊號轉換 + 回測
                        │  Vanilla + Kelly × 60 = 120 策略
                        │
                        ├── all_strategies_daily_returns.csv
                        └── strategy_metrics_comparison.csv
                                │
                                ▼
                        [Pipeline 04] 視覺化
                                │
                                ├── equity_curves/（20張）
                                └── prediction_signals/（8張）
```

---

## 四、關鍵設計決策

| 決策 | 選擇 | 理由 |
|------|------|------|
| 訓練窗口 | 750 天 | 避免 XGBoost 學到過時市場結構；覆蓋完整利率循環約3年 |
| 時間隔離 | 訓練截止 `t-2`，預測 `t` | Y 定義為 t→t+1 報酬，t-1 的 Y 需到 t 收盤才確認，訓練不得使用 |
| Feature Selection | Permutation Importance | 模型無關，跨樣本 shuffle，LSTM/XGB/REG 三者可比 |
| PI 重選頻率 | 每 20 天 | 平衡穩定性與適應性 |
| Top K | 30 | 樣本比 750/30=25，學術標準安全範圍 |
| Validation | 無 | Pipeline 03 已刪除，Walk-Forward OOS 即為評估 |
| EarlyStopping | 無 | 預設超參數下固定 20 epochs，結果可重現 |
| Huber delta | 動態 `2.5 × std(y_train)` | 自適應當期市場波動率，避免死板常數 |
| Kelly 參數 | 固定先驗值 | 避免資金曲線的高維度 Curve Fitting |
| Vanilla min/max | 測試期前 750 天 | 無 Lookahead Bias |

---

## 五、輸出檔案索引

| 檔案 | 產出 Pipeline | 說明 |
|------|-------------|------|
| `pipeline_data/datasets/dataset_*.csv` | 01 | 5 個特徵資料集 |
| `pipeline_data/predictions/all_models_predictions_matrix.csv` | 02 | 60 欄預測值 + date_id + actual_return |
| `pipeline_data/predictions/metrics_comparison_summary.csv` | 02 | 60 種組合的 OOS 預測指標 |
| `pipeline_data/backtest/all_strategies_daily_returns.csv` | 03 | 120 個策略每日報酬 + Buy and Hold |
| `pipeline_data/backtest/strategy_metrics_comparison.csv` | 03 | 120 個策略的 Sharpe、MDD、Total Return |
| `pipeline_data/evaluation/equity_curves/*.png` | 04 | 20 張策略回測比較圖 |
| `pipeline_data/evaluation/prediction_signals/*.png` | 04 | 8 張預測訊號比較圖 |