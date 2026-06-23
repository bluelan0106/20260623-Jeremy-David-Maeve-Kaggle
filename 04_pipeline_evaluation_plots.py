"""
04_pipeline_evaluation_plots.py
=================================
直接接在 03_pipeline_signal_and_backtest.py 之後執行。

產出說明：
  A. Equity Curves（策略回測圖）
     - 比較維度：Feature Selection、Loss、Strategy
     - 每張圖：1×5 橫向排列（Dataset 1~5）
     - 實驗組（Solid）vs 控制組（Dashed）

  B. Prediction Signals（預測訊號圖）
     - 比較維度：Feature Selection、Loss
     - 每張圖：1×5 橫向排列（Dataset 1~5）

欄位命名規則（承接 Pipeline 02/03）：
  預測欄：{LSTM/XGB/REG}_ds{N}_fs{Y/N}_{huber/mse}
  策略欄：{預測欄}_vanilla / {預測欄}_kelly
"""

import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. 路徑設定
# ==============================================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(BASE_DIR, 'pipeline_data', 'predictions',
                         'all_models_predictions_matrix.csv')
RET_FILE  = os.path.join(BASE_DIR, 'pipeline_data', 'backtest',
                         'all_strategies_daily_returns.csv')

EVAL_DIR  = os.path.join(BASE_DIR, 'pipeline_data', 'evaluation')
EQ_DIR    = os.path.join(EVAL_DIR, 'equity_curves')
SIG_DIR   = os.path.join(EVAL_DIR, 'prediction_signals')
os.makedirs(EQ_DIR,  exist_ok=True)
os.makedirs(SIG_DIR, exist_ok=True)

# 三個模型的顏色（REG 取代原 OLS）
COLOR_MAP = {
    'LSTM': '#1f77b4',
    'XGB':  '#ff7f0e',
    'REG':  '#2ca02c',
}


# ==============================================================================
# 2. 核心繪圖函數
# ==============================================================================
def plot_5_datasets(df: pd.DataFrame,
                    x_col: str,
                    base_title: str,
                    file_name: str,
                    save_dir: str,
                    get_cols_func,
                    is_cumulative: bool = False):
    """
    繪製 5 個 Dataset 橫向排列的比較圖（1×5）。

    get_cols_func(ds_num: str) → dict：
        {
          'LSTM': {'exp': col_name, 'ctrl': col_name},
          'XGB':  {'exp': col_name, 'ctrl': col_name},
          'REG':  {'exp': col_name, 'ctrl': col_name},
        }
    實線（Solid）= 實驗組（exp），虛線（Dashed）= 控制組（ctrl）。
    """
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(base_title, fontsize=16, y=1.05, fontweight='bold')

    for i, ds in enumerate(['1', '2', '3', '4', '5']):
        ax       = axes[i]
        cols_dict = get_cols_func(ds)

        # Baseline（灰色虛線，底層）
        if 'Buy_and_Hold' in df.columns:
            y_bh = np.cumsum(df['Buy_and_Hold']) if is_cumulative else df['Buy_and_Hold']
            ax.plot(df[x_col], y_bh, color='gray', linestyle='--',
                    linewidth=1.2, alpha=0.6, label='Buy & Hold', zorder=1)
        elif 'actual_return' in df.columns:
            y_bh = df['actual_return']
            ax.plot(df[x_col], y_bh, color='gray', linestyle='--',
                    linewidth=1.2, alpha=0.6, label='Actual Return', zorder=1)

        # 三個模型（實線 = 實驗組，虛線 = 控制組）
        for model in ['LSTM', 'XGB', 'REG']:
            if model not in cols_dict:
                continue
            exp_col  = cols_dict[model].get('exp')
            ctrl_col = cols_dict[model].get('ctrl')
            color    = COLOR_MAP[model]

            if exp_col and exp_col in df.columns:
                y = np.cumsum(df[exp_col]) if is_cumulative else df[exp_col]
                ax.plot(df[x_col], y, color=color, linestyle='-',
                        linewidth=1.5, label=f'{model} (Exp)', zorder=2)

            if ctrl_col and ctrl_col in df.columns:
                y = np.cumsum(df[ctrl_col]) if is_cumulative else df[ctrl_col]
                ax.plot(df[x_col], y, color=color, linestyle='--',
                        linewidth=1.5, alpha=0.8, label=f'{model} (Ctrl)', zorder=2)

        ax.set_title(f'Dataset {ds}', fontsize=13)
        ax.set_xlabel('Date ID')
        if i == 0:
            ax.set_ylabel('Cumulative Return' if is_cumulative else 'Prediction Value')
        ax.grid(True, linestyle=':', alpha=0.6)

        # Legend 統整放在最後一欄右側
        if i == 4:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    plt.tight_layout()
    out_path = os.path.join(save_dir, file_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{file_name}")


# ==============================================================================
# 3. 欄位名稱建構輔助函數
# ==============================================================================
def pred_col(model: str, ds: str, fs: str, loss: str) -> str:
    """
    建構預測矩陣的欄位名稱。
    model: 'LSTM' | 'XGB' | 'REG'
    fs:    'Y' | 'N'
    loss:  'huber' | 'mse'
    """
    return f"{model}_ds{ds}_fs{fs}_{loss}"


def ret_col(model: str, ds: str, fs: str, loss: str, strat: str) -> str:
    """
    建構策略報酬矩陣的欄位名稱。
    strat: 'vanilla' | 'kelly'
    """
    return f"{pred_col(model, ds, fs, loss)}_{strat}"


# ==============================================================================
# 4. 主程序
# ==============================================================================
def generate_plots():
    print("載入資料...")
    df_ret  = pd.read_csv(RET_FILE)  if os.path.exists(RET_FILE)  else None
    df_pred = pd.read_csv(PRED_FILE) if os.path.exists(PRED_FILE) else None

    MODELS = ['LSTM', 'XGB', 'REG']

    # ==========================================================================
    # A. Equity Curves
    # ==========================================================================
    if df_ret is not None:
        print("\n繪製 Equity Curves...")

        # ----------------------------------------------------------------------
        # A1. 比較 Feature Selection（Exp=Y, Ctrl=N）
        #     固定：loss, strategy
        #     圖數：2 loss × 2 strat = 4 張
        # ----------------------------------------------------------------------
        print("  [A1] Feature Selection 比較...")
        for loss, strat in itertools.product(['huber', 'mse'], ['vanilla', 'kelly']):
            def get_cols(ds, _loss=loss, _strat=strat):
                return {
                    m: {
                        'exp':  ret_col(m, ds, 'Y', _loss, _strat),
                        'ctrl': ret_col(m, ds, 'N', _loss, _strat),
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_ret, 'date_id',
                f'Equity | Feature Selection (Solid=Y, Dash=N)\nLoss={loss}, Strategy={strat}',
                f'Equity_FS_{loss}_{strat}.png',
                EQ_DIR, get_cols, is_cumulative=True
            )

        # ----------------------------------------------------------------------
        # A2. 比較 Loss（Exp=huber, Ctrl=mse）
        #     固定：fs, strategy
        #     圖數：2 fs × 2 strat = 4 張
        # ----------------------------------------------------------------------
        print("  [A2] Loss 比較...")
        for fs, strat in itertools.product(['Y', 'N'], ['vanilla', 'kelly']):
            def get_cols(ds, _fs=fs, _strat=strat):
                return {
                    m: {
                        'exp':  ret_col(m, ds, _fs, 'huber', _strat),
                        'ctrl': ret_col(m, ds, _fs, 'mse',   _strat),
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_ret, 'date_id',
                f'Equity | Loss (Solid=Huber, Dash=MSE)\nFS={fs}, Strategy={strat}',
                f'Equity_Loss_fs{fs}_{strat}.png',
                EQ_DIR, get_cols, is_cumulative=True
            )

        # ----------------------------------------------------------------------
        # A3. 比較 Strategy（Exp=kelly, Ctrl=vanilla）
        #     固定：fs, loss
        #     圖數：2 fs × 2 loss = 4 張
        # ----------------------------------------------------------------------
        print("  [A3] Strategy 比較...")
        for fs, loss in itertools.product(['Y', 'N'], ['huber', 'mse']):
            def get_cols(ds, _fs=fs, _loss=loss):
                return {
                    m: {
                        'exp':  ret_col(m, ds, _fs, _loss, 'kelly'),
                        'ctrl': ret_col(m, ds, _fs, _loss, 'vanilla'),
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_ret, 'date_id',
                f'Equity | Strategy (Solid=Kelly, Dash=Vanilla)\nFS={fs}, Loss={loss}',
                f'Equity_Strat_fs{fs}_{loss}.png',
                EQ_DIR, get_cols, is_cumulative=True
            )

        # ----------------------------------------------------------------------
        # A4. 比較 Model（LSTM vs XGB vs REG，全畫在同一張）
        #     固定：fs, loss, strategy
        #     圖數：2 fs × 2 loss × 2 strat = 8 張
        #     這組不分 exp/ctrl，三條線同時畫
        # ----------------------------------------------------------------------
        print("  [A4] Model 比較...")
        for fs, loss, strat in itertools.product(['Y', 'N'], ['huber', 'mse'],
                                                  ['vanilla', 'kelly']):
            def get_cols(ds, _fs=fs, _loss=loss, _strat=strat):
                # 三個模型都是「實驗組」，各自對應自己的曲線
                # ctrl 設為 None 讓繪圖函數跳過虛線
                return {
                    m: {
                        'exp':  ret_col(m, ds, _fs, _loss, _strat),
                        'ctrl': None,
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_ret, 'date_id',
                f'Equity | Model Comparison (LSTM / XGB / REG)\nFS={fs}, Loss={loss}, Strategy={strat}',
                f'Equity_Model_fs{fs}_{loss}_{strat}.png',
                EQ_DIR, get_cols, is_cumulative=True
            )

    # ==========================================================================
    # B. Prediction Signals
    # ==========================================================================
    if df_pred is not None:
        print("\n繪製 Prediction Signals...")

        # ----------------------------------------------------------------------
        # B1. 比較 Feature Selection（Exp=Y, Ctrl=N）
        #     固定：loss
        #     圖數：2 loss = 2 張
        # ----------------------------------------------------------------------
        print("  [B1] Feature Selection 比較...")
        for loss in ['huber', 'mse']:
            def get_cols(ds, _loss=loss):
                return {
                    m: {
                        'exp':  pred_col(m, ds, 'Y', _loss),
                        'ctrl': pred_col(m, ds, 'N', _loss),
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_pred, 'date_id',
                f'Signal | Feature Selection (Solid=Y, Dash=N)\nLoss={loss}',
                f'Signal_FS_{loss}.png',
                SIG_DIR, get_cols, is_cumulative=False
            )

        # ----------------------------------------------------------------------
        # B2. 比較 Loss（Exp=huber, Ctrl=mse）
        #     固定：fs
        #     圖數：2 fs = 2 張
        # ----------------------------------------------------------------------
        print("  [B2] Loss 比較...")
        for fs in ['Y', 'N']:
            def get_cols(ds, _fs=fs):
                return {
                    m: {
                        'exp':  pred_col(m, ds, _fs, 'huber'),
                        'ctrl': pred_col(m, ds, _fs, 'mse'),
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_pred, 'date_id',
                f'Signal | Loss (Solid=Huber, Dash=MSE)\nFS={fs}',
                f'Signal_Loss_fs{fs}.png',
                SIG_DIR, get_cols, is_cumulative=False
            )

        # ----------------------------------------------------------------------
        # B3. 比較 Model（LSTM vs XGB vs REG）
        #     固定：fs, loss
        #     圖數：2 fs × 2 loss = 4 張
        # ----------------------------------------------------------------------
        print("  [B3] Model 比較...")
        for fs, loss in itertools.product(['Y', 'N'], ['huber', 'mse']):
            def get_cols(ds, _fs=fs, _loss=loss):
                return {
                    m: {
                        'exp':  pred_col(m, ds, _fs, _loss),
                        'ctrl': None,
                    }
                    for m in MODELS
                }
            plot_5_datasets(
                df_pred, 'date_id',
                f'Signal | Model Comparison (LSTM / XGB / REG)\nFS={fs}, Loss={loss}',
                f'Signal_Model_fs{fs}_{loss}.png',
                SIG_DIR, get_cols, is_cumulative=False
            )

    print(f"\n所有繪圖完成！")
    print(f"   Equity Curves  → {EQ_DIR}")
    print(f"   Pred Signals   → {SIG_DIR}")


# ==============================================================================
if __name__ == "__main__":
    generate_plots()
