"""
03_pipeline_signal_and_backtest.py
====================================
直接接在 02_pipeline_model_predictions.py 之後執行。

流程說明：
- 讀取 02 產出的 60 欄預測矩陣
- 對每欄預測值產生兩種策略的每日報酬：
    1. Vanilla：靜態 Min-Max Scaling → 倉位 [0, 2]
    2. Kelly：Half-Kelly + Breadth Conviction Penalty → 倉位 [0, 2]
- 計算所有策略的 OOS 回測指標（Sharpe、MDD、Total Return）
- 輸出每日報酬矩陣與指標總表

欄位命名規則（承接 Pipeline 02）：
  {LSTM/XGB/REG}_ds{N}_fs{Y/N}_{huber/mse}

策略欄位命名：
  {原欄位名稱}_vanilla
  {原欄位名稱}_kelly
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. 路徑設定
# ==============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'pipeline_data', 'datasets')
PRED_DIR  = os.path.join(BASE_DIR, 'pipeline_data', 'predictions')
OUT_DIR   = os.path.join(BASE_DIR, 'pipeline_data', 'backtest')
os.makedirs(OUT_DIR, exist_ok=True)

PRED_FILE      = os.path.join(PRED_DIR, 'all_models_predictions_matrix.csv')
RETURNS_FILE   = os.path.join(OUT_DIR,  'all_strategies_daily_returns.csv')
POSITIONS_FILE = os.path.join(OUT_DIR,  'all_strategies_daily_positions.csv')
METRICS_FILE   = os.path.join(OUT_DIR,  'strategy_metrics_comparison.csv')


# ==============================================================================
# 2. Kelly Strategy 參數（對應 MD Section 5）
# ==============================================================================
LAMBDA_PARAM  = 0.75   # Risk Aversion Scalar（3/4-Kelly）
K_TAIL        = 3.0    # Sigmoid 斜率平滑度
THETA         = 0.5    # Breadth Spread 啟動門檻
ROLLING_WINDOW = 750   # 與 Pipeline 02 對齊（Vanilla min/max 計算窗口）


# ==============================================================================
# 3. 工具函數
# ==============================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calc_metrics(returns_series: pd.Series) -> tuple:
    """
    計算回測指標（學術標準複利版本）。
    回傳 (Sharpe_Ratio, MDD, Total_Return)。
    """
    r = returns_series.dropna()
    if len(r) == 0:
        return 0.0, 0.0, 0.0

    # Sharpe Ratio（年化，乘以 sqrt(252)）
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sr      = ann_ret / ann_vol if ann_vol != 0 else 0.0

    # Total Return（複利）
    cum_ret = (1 + r).cumprod()
    total_ret = cum_ret.iloc[-1] - 1

    # Maximum Drawdown
    running_max = cum_ret.cummax()
    mdd = ((cum_ret - running_max) / running_max).min()

    return sr, mdd, total_ret


def parse_ds_num(col: str) -> str:
    """
    從欄位名稱中擷取 dataset 號碼。
    新命名格式：LSTM_ds1_fsY_huber → '1'
    """
    match = re.search(r'_ds(\d+)_', col)
    return match.group(1) if match else '1'


# ==============================================================================
# 4. 主程序
# ==============================================================================
def run_backtest():
    # ------------------------------------------------------------------
    # 讀取預測矩陣
    # ------------------------------------------------------------------
    print("📖 讀取預測矩陣...")
    if not os.path.exists(PRED_FILE):
        print(f"找不到預測矩陣: {PRED_FILE}")
        return

    df_pred   = pd.read_csv(PRED_FILE)
    model_cols = [c for c in df_pred.columns if c not in ['date_id', 'actual_return']]
    print(f"偵測到 {len(model_cols)} 個預測欄位。")

    # 初始化報酬大表
    df_returns = pd.DataFrame({'date_id': df_pred['date_id']})
    df_returns['Buy_and_Hold'] = df_pred['actual_return']

    # 初始化倉位大表（供 net-cost 重算與 robustness 分析使用）
    # Buy_and_Hold 的隱含 f_t 恆為 1
    df_positions = pd.DataFrame({'date_id': df_pred['date_id']})
    df_positions['Buy_and_Hold'] = 1.0

    test_start_id = df_pred['date_id'].min()

    # ------------------------------------------------------------------
    # 預處理每個 Dataset 的 Meta 資料
    # ------------------------------------------------------------------
    print("載入各 Dataset 的 Meta 資料（ATR、Spread、min/max）...")
    ds_meta  = {}
    ds_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

    for ds_file in ds_files:
        match  = re.search(r'\d+', ds_file)
        ds_num = match.group() if match else '1'

        df_raw = pd.read_csv(os.path.join(DATA_DIR, ds_file))
        df_raw = df_raw.sort_values('date_id').reset_index(drop=True)

        # Vanilla：用測試期前750天的真實報酬找 min/max（無 Lookahead Bias）
        # Y 定義為 t→t+1 報酬，test_start_id - 1 的 Y 需到 test_start_id
        # 收盤才確認，與 Pipeline 02 訓練窗口採用相同的時間隔離原則。
        train_mask = (
            (df_raw['date_id'] >= test_start_id - ROLLING_WINDOW - 1) &
            (df_raw['date_id'] <  test_start_id - 1)
        )
        train_df  = df_raw[train_mask]
        scale_min = train_df['market_forward_excess_returns'].min()
        scale_max = train_df['market_forward_excess_returns'].max()

        # Kelly：測試期的 ATR 與 Breadth Spread（由 Pipeline 01 計算並存入 meta 欄位）
        test_df = df_raw[df_raw['date_id'] >= test_start_id].set_index('date_id')

        ds_meta[ds_num] = {
            'min':    scale_min,
            'max':    scale_max,
            'atr':    test_df['meta_atr'],
            'spread': test_df['meta_spread'],
        }
        print(f"   ds{ds_num}: min={scale_min:.6f}, max={scale_max:.6f}, "
              f"test_days={len(test_df)}")

    # ------------------------------------------------------------------
    # 對每個預測欄位計算 Vanilla 與 Kelly 策略的每日報酬
    # ------------------------------------------------------------------
    print("\n計算 Vanilla 與 Kelly 策略每日報酬...")

    actual = df_pred['actual_return'].values

    for col in model_cols:
        ds_num = parse_ds_num(col)
        meta   = ds_meta.get(ds_num)

        if meta is None:
            print(f"找不到 ds{ds_num} 的 Meta 資料，跳過 {col}")
            continue

        pred = df_pred[col].values

        # --------------------------------------------------------------
        # Strategy 1：Vanilla（Static Min-Max Scaling）
        # 最小值對應倉位 0，最大值對應倉位 2，線性插值
        # --------------------------------------------------------------
        scale_range = meta['max'] - meta['min'] + 1e-9
        f_vanilla   = 2.0 * (pred - meta['min']) / scale_range
        f_vanilla   = np.clip(f_vanilla, 0.0, 2.0)

        df_positions[f"{col}_vanilla"] = f_vanilla
        df_returns[f"{col}_vanilla"]   = f_vanilla * actual

        # --------------------------------------------------------------
        # Strategy 2：Kelly Engine（MD Section 5 公式）
        #
        # Step 1: Raw Kelly = pred / ATR²
        # Step 2: Conviction Penalty kappa = 0.5 + 0.5×(1 - sigmoid(k×(BS - θ)))
        # Step 3: Final = clamp(0, 2, Raw_Kelly × λ × kappa)
        # --------------------------------------------------------------
        atr    = meta['atr'].reindex(df_pred['date_id']).values
        bs_t   = meta['spread'].reindex(df_pred['date_id']).values

        f_kelly_raw = pred / (atr ** 2 + 1e-9)
        kappa_t     = 0.5 + 0.5 * (1.0 - sigmoid(K_TAIL * (bs_t - THETA)))
        f_final     = np.clip(f_kelly_raw * LAMBDA_PARAM * kappa_t, 0.0, 2.0)

        df_positions[f"{col}_kelly"] = f_final
        df_returns[f"{col}_kelly"]   = f_final * actual

    # 儲存每日報酬矩陣
    df_returns.to_csv(RETURNS_FILE, index=False)
    n_strat = len(df_returns.columns) - 2  # 排除 date_id, Buy_and_Hold
    print(f"每日報酬矩陣已儲存（{n_strat} 個策略欄位）: {RETURNS_FILE}")

    # 儲存每日倉位矩陣
    df_positions.to_csv(POSITIONS_FILE, index=False)
    print(f"每日倉位矩陣已儲存（{n_strat} 個策略欄位）: {POSITIONS_FILE}")

    # ------------------------------------------------------------------
    # 計算所有策略的 OOS 回測指標
    # ------------------------------------------------------------------
    print("\n計算回測指標（Sharpe、MDD、Total Return）...")
    records = []

    # Baseline：Buy and Hold
    sr, mdd, tr = calc_metrics(df_returns['Buy_and_Hold'])
    records.append({
        'Model': 'Buy_and_Hold', 'Strategy': 'Buy_and_Hold',
        'Sharpe_Ratio': round(sr, 4),
        'MDD':          round(mdd, 4),
        'Total_Return': round(tr, 4),
    })

    # 所有策略欄位
    strat_cols = [c for c in df_returns.columns if c not in ['date_id', 'Buy_and_Hold']]
    for col in strat_cols:
        sr, mdd, tr = calc_metrics(df_returns[col])
        strat_type  = col.split('_')[-1].capitalize()  # 'Vanilla' or 'Kelly'
        records.append({
            'Model':        col,
            'Strategy':     strat_type,
            'Sharpe_Ratio': round(sr, 4),
            'MDD':          round(mdd, 4),
            'Total_Return': round(tr, 4),
        })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(METRICS_FILE, index=False)

    # 印出簡易排行（依 Sharpe 降序）
    top5 = (df_metrics[df_metrics['Model'] != 'Buy_and_Hold']
            .sort_values('Sharpe_Ratio', ascending=False)
            .head(5))
    print("\nTop 5 策略（依 Sharpe Ratio）：")
    print(top5[['Model', 'Sharpe_Ratio', 'MDD', 'Total_Return']].to_string(index=False))

    print(f"\nPipeline 03 完成！")
    print(f"   每日報酬矩陣: {RETURNS_FILE}")
    print(f"   每日倉位矩陣: {POSITIONS_FILE}")
    print(f"   指標總表    : {METRICS_FILE}")


# ==============================================================================
if __name__ == "__main__":
    run_backtest()