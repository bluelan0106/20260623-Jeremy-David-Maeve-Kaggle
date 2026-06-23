"""
02_pipeline_model_predictions.py
=================================
直接接在 01_pipeline_feature_engineering.py 之後執行。

架構說明：
- 3 models × 5 datasets × 2 (FS/No FS) × 2 (huber/mse) = 60 種組合
- Walk-Forward OOS：750天訓練窗口，每天滾動1天，預測最後180天
- Feature Selection：每20天用當期訓練窗口重跑 Permutation Importance，動態選 Top 30
  （PI 模型與預測主模型架構完全一致，確保 feature 重要性評估對齊主模型的學習能力）
- 無 Optuna、無 EarlyStopping、無 validation fold，統一使用預設超參數
- 斷點續傳：已完成的欄位自動跳過

欄位命名規則：
  LSTM_ds{N}_fs{Y/N}_{huber/mse}
  XGB_ds{N}_fs{Y/N}_{huber/mse}
  REG_ds{N}_fs{Y/N}_{huber/mse}
"""

import os
import gc
import re
import random
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


# ==============================================================================
# 1. 硬體加速與隨機種子
# ==============================================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 硬體加速已啟用（偵測到 {len(gpus)} 個核心）")
        except Exception:
            pass
    elif tf.config.list_physical_devices('Metal'):
        print("Apple Silicon GPU (Metal) 加速已啟用")


def set_random_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"隨機種子已統一固定為 {seed}")


setup_gpu()
set_random_seed(42)


# ==============================================================================
# 2. 路徑與全域參數
# ==============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'pipeline_data', 'datasets')
OUT_DIR      = os.path.join(BASE_DIR, 'pipeline_data', 'predictions')
os.makedirs(OUT_DIR, exist_ok=True)

MATRIX_FILE   = os.path.join(OUT_DIR, 'all_models_predictions_matrix.csv')
METRICS_FILE  = os.path.join(OUT_DIR, 'metrics_comparison_summary.csv')
FEAT_LOG_DIR  = os.path.join(OUT_DIR, 'feature_logs')
os.makedirs(FEAT_LOG_DIR, exist_ok=True)

TARGET_COL       = 'market_forward_excess_returns'
ID_COL           = 'date_id'
ROLLING_WINDOW   = 750    # 約3年，避免XGBoost學到過時市場結構
OOS_DAYS         = 180    # 最後180天為樣本外預測期
TIMESTEPS        = 60     # LSTM序列長度（同時作為PI計算的序列長度）
TOP_K_FEATURES   = 30     # 每次PI保留的feature數
PI_REFRESH_EVERY = 20     # 每20天重跑一次PI
EPOCHS           = 20     # LSTM固定訓練輪數
BATCH_SIZE       = 2048
TARGET_MULT      = 100.0  # 放大target以利LSTM收斂

# LSTM 預設超參數
LSTM_PARAMS = {'units': 32, 'dropout': 0.1, 'lr': 1e-3}

# XGBoost 預設超參數
XGB_PARAMS = {'n_estimators': 300, 'max_depth': 3,
               'learning_rate': 0.05, 'n_jobs': -1, 'random_state': 42}


# ==============================================================================
# 3. 共用工具函數
# ==============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return None
    ric, _ = spearmanr(yt, yp)
    pic, _ = pearsonr(yt, yp) if np.std(yp) > 0 else (0.0, 0.0)
    return {
        'Model':      label,
        'RMSE':       round(np.sqrt(mean_squared_error(yt, yp)), 6),
        'MAE':        round(mean_absolute_error(yt, yp), 6),
        'R2':         round(r2_score(yt, yp), 6),
        'ACC':        round(accuracy_score(yt > 0, yp > 0), 6),
        'RANK_IC':    round(float(ric) if not np.isnan(ric) else 0.0, 6),
        'PEARSON_IC': round(float(pic) if not np.isnan(pic) else 0.0, 6),
    }


# ==============================================================================
# 4. Permutation Importance（模型無關，跨樣本shuffle）
# ==============================================================================
def _pi_scores_generic(predict_fn, X_ref: np.ndarray, y: np.ndarray,
                        feature_names: list, seed: int = 42) -> dict:
    """
    通用 Permutation Importance 計算核心。
    跨樣本shuffle（across samples），XGBoost / REG / LSTM 使用同一策略確保可比性。

    predict_fn: callable，接受與 X_ref 相同形狀的輸入，回傳 1D array。
    X_ref:      2D array (samples, features)，已經是模型直接吃的格式。
                對 LSTM 而言這是 flatten 前的 2D，PI 內部會做 reshape。
    """
    rng          = np.random.default_rng(seed)
    baseline_mse = mean_squared_error(y, predict_fn(X_ref))
    scores       = {}

    for i, feat in enumerate(feature_names):
        X_perm = X_ref.copy()
        shuffle_idx        = rng.permutation(X_ref.shape[0])
        X_perm[:, i]       = X_ref[shuffle_idx, i]
        perm_mse           = mean_squared_error(y, predict_fn(X_perm))
        scores[feat]       = max(0.0, perm_mse - baseline_mse)

    return scores


def select_top_features_by_pi(train_window: pd.DataFrame,
                               candidate_features: list,
                               model_type: str,
                               loss_type: str,
                               seed: int = 42) -> list:
    """
    用當期訓練窗口訓練與主模型完全相同架構的模型，跑 PI 後回傳 Top K feature 清單。
    PI 模型與預測主模型架構一致，確保 feature 重要性評估與主模型的學習能力對齊。
    model_type: 'lstm' | 'xgb' | 'reg'
    """
    sc      = StandardScaler()
    X_sc    = sc.fit_transform(train_window[candidate_features].values)
    y_vals  = train_window[TARGET_COL].values
    n_feats = len(candidate_features)

    # ------------------------------------------------------------------
    # LSTM：與主模型完全相同（units=32, epochs=20, loss 對齊）
    # ------------------------------------------------------------------
    if model_type == 'lstm':
        y_mult     = y_vals * TARGET_MULT
        X_3d, y_3d = _create_sequences(X_sc, y_mult, TIMESTEPS)
        if len(X_3d) == 0:
            return candidate_features[:TOP_K_FEATURES]

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)

        # 使用主模型建構函數，與 _predict_one_day_lstm 完全一致
        pi_model = _build_lstm(n_feats, loss_type)
        pi_model.fit(X_3d, y_3d, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     verbose=0, shuffle=False)

        def lstm_predict(X_2d):
            seqs, _ = _create_sequences(X_2d, np.zeros(len(X_2d)), TIMESTEPS)
            if len(seqs) == 0:
                return np.zeros(len(y_3d))
            return pi_model.predict(seqs, batch_size=512, verbose=0).flatten()

        scores = _pi_scores_generic(lstm_predict, X_sc, y_3d, candidate_features, seed)

        del pi_model
        tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # XGBoost：與主模型完全相同（n_estimators=300，動態 huber delta）
    # ------------------------------------------------------------------
    elif model_type == 'xgb':
        # huber_delta 計算：先將 y 縮放至 TARGET_MULT 尺度（×100）計算 std，
        # 再除回 TARGET_MULT 對齊 y_vals 的原始尺度。
        # 若不除回，delta ≈ 0.75 遠大於殘差範圍 ±0.04，
        # 所有樣本都落入 MSE 區間，Huber 退化成純 MSE。
        std_y       = np.std(y_vals * TARGET_MULT)
        huber_delta = 2.5 * (std_y if std_y > 0 else 1e-6) / TARGET_MULT

        # 使用主模型建構函數，與 _predict_one_day_xgb 完全一致
        pi_model = _build_xgb(loss_type, huber_delta)
        pi_model.fit(X_sc, y_vals)

        def xgb_predict(X_2d):
            return pi_model.predict(X_2d)

        scores = _pi_scores_generic(xgb_predict, X_sc, y_vals, candidate_features, seed)

    # ------------------------------------------------------------------
    # REG：與主模型完全相同（動態 huber delta）
    # ------------------------------------------------------------------
    else:
        # huber_delta 計算：同 XGB，縮放後計算再除回原始尺度。
        # 確保 delta 落在殘差的合理切換點（約 2.5 倍標準差），
        # 而非遠大於所有殘差導致 Huber 退化成純 MSE。
        std_y       = np.std(y_vals * TARGET_MULT)
        huber_delta = 2.5 * (std_y if std_y > 0 else 1e-6) / TARGET_MULT

        # 使用主模型建構函數，與 _predict_one_day_reg 完全一致
        pi_model = _build_reg(loss_type, huber_delta)
        pi_model.fit(X_sc, y_vals)

        def reg_predict(X_2d):
            return pi_model.predict(X_2d)

        scores = _pi_scores_generic(reg_predict, X_sc, y_vals, candidate_features, seed)

    # Top K
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in sorted_feats[:TOP_K_FEATURES]]
    gc.collect()
    return top_features


# ==============================================================================
# 5. LSTM 工具
# ==============================================================================
def _create_sequences(X: np.ndarray, y: np.ndarray, timesteps: int):
    """向量化序列建構，取代 Python for loop，大幅降低 CPU 前處理時間。"""
    if len(X) <= timesteps:
        return np.array([]), np.array([])
    n   = len(X) - timesteps
    idx = np.arange(timesteps)[None, :] + np.arange(n)[:, None]
    Xs  = X[idx]
    ys  = y[timesteps:]
    return Xs.astype(np.float32), ys.astype(np.float32)


def _build_lstm(n_feats: int, loss_type: str) -> tf.keras.Model:
    loss_fn = tf.keras.losses.Huber() if loss_type == 'huber' else 'mse'
    model = Sequential([
        Input(shape=(TIMESTEPS, n_feats)),
        LSTM(LSTM_PARAMS['units'], kernel_regularizer=l2(0.001), activation='tanh'),
        Dropout(LSTM_PARAMS['dropout'], seed=42),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_PARAMS['lr']),
        loss=loss_fn
    )
    return model


# ==============================================================================
# 6. XGBoost 工具
# ==============================================================================
def _build_xgb(loss_type: str, huber_delta: float = 1.0) -> XGBRegressor:
    obj = 'reg:pseudohubererror' if loss_type == 'huber' else 'reg:squarederror'
    params = XGB_PARAMS.copy()
    params['objective'] = obj
    if loss_type == 'huber':
        params['huber_slope'] = huber_delta
    return XGBRegressor(**params)


# ==============================================================================
# 7. Regression 工具（PseudoHuberRegressor + LinearRegression）
# ==============================================================================
class PseudoHuberRegressor:
    """
    線性模型 + Pseudo-Huber loss，與 XGBoost 的 reg:pseudohubererror 對齊。
    L(r) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
    使用 L-BFGS-B 優化。
    """
    def __init__(self, delta: float = 1.0):
        self.delta      = delta
        self.coef_      = None
        self.intercept_ = None

    def _loss(self, params, X, y):
        w, b = params[:-1], params[-1]
        r    = y - (X @ w + b)
        d    = self.delta
        return np.mean(d ** 2 * (np.sqrt(1 + (r / d) ** 2) - 1))

    def _grad(self, params, X, y):
        w, b = params[:-1], params[-1]
        r    = y - (X @ w + b)
        d    = self.delta
        g    = r / np.sqrt(1 + (r / d) ** 2)
        n    = len(y)
        return np.concatenate([-X.T @ g / n, [-np.sum(g) / n]])

    def fit(self, X, y):
        x0     = np.zeros(X.shape[1] + 1)
        result = minimize(self._loss, x0, args=(X, y),
                          jac=self._grad, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10})
        self.coef_      = result.x[:-1]
        self.intercept_ = result.x[-1]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def _build_reg(loss_type: str, huber_delta: float = 1.0):
    if loss_type == 'huber':
        return PseudoHuberRegressor(delta=huber_delta)
    return LinearRegression()


# ==============================================================================
# 8. 單日預測函數（每個模型類型各一）
# ==============================================================================
def _predict_one_day_lstm(train_window: pd.DataFrame,
                          test_row: pd.DataFrame,
                          features: list,
                          loss_type: str) -> float:
    sc             = StandardScaler()
    X_train_sc     = sc.fit_transform(train_window[features].values)
    y_train        = train_window[TARGET_COL].values * TARGET_MULT

    # 測試序列：取最近 TIMESTEPS 天 + 當天，形成最後一筆序列
    test_window_sc = sc.transform(test_row[features].values)

    X_tr_3d, y_tr_3d = _create_sequences(X_train_sc, y_train, TIMESTEPS)
    if len(X_tr_3d) == 0:
        return np.nan

    if len(test_window_sc) < TIMESTEPS:
        return np.nan
    X_te_3d = test_window_sc[-TIMESTEPS:].reshape(1, TIMESTEPS, len(features)).astype(np.float32)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = _build_lstm(len(features), loss_type)
    model.fit(X_tr_3d, y_tr_3d, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, shuffle=False)

    pred = model.predict(X_te_3d, verbose=0)[0][0]
    del model
    tf.keras.backend.clear_session()
    return float(pred) / TARGET_MULT


def _predict_one_day_xgb(train_window: pd.DataFrame,
                          test_row: pd.DataFrame,
                          features: list,
                          loss_type: str) -> float:
    X_tr = train_window[features].values
    y_tr = train_window[TARGET_COL].values
    X_te = test_row[features].values[-1].reshape(1, -1)

    # huber_delta：縮放至 TARGET_MULT 尺度計算 std，再除回對齊 y_tr 原始尺度。
    # 確保切換點約在殘差分佈的 2.5 倍標準差，Huber 才能正常區分一般誤差與極端值。
    std_y       = np.std(y_tr * TARGET_MULT)
    huber_delta = 2.5 * (std_y if std_y > 0 else 1e-6) / TARGET_MULT

    model = _build_xgb(loss_type, huber_delta)
    model.fit(X_tr, y_tr)
    return float(model.predict(X_te)[0])


def _predict_one_day_reg(train_window: pd.DataFrame,
                          test_row: pd.DataFrame,
                          features: list,
                          loss_type: str) -> float:
    sc   = StandardScaler()
    X_tr = sc.fit_transform(train_window[features].values)
    y_tr = train_window[TARGET_COL].values
    X_te = sc.transform(test_row[features].values[-1].reshape(1, -1))

    # huber_delta：同 XGB，縮放至 TARGET_MULT 尺度計算 std，再除回對齊 y_tr 原始尺度。
    std_y       = np.std(y_tr * TARGET_MULT)
    huber_delta = 2.5 * (std_y if std_y > 0 else 1e-6) / TARGET_MULT

    model = _build_reg(loss_type, huber_delta)
    model.fit(X_tr, y_tr)
    return float(model.predict(X_te)[0])


# ==============================================================================
# 9. 通用滾動預測迴圈
# ==============================================================================
def _run_rolling(df: pd.DataFrame,
                 all_features: list,
                 test_dates: list,
                 model_type: str,
                 use_fs: bool,
                 loss_type: str,
                 col_name: str) -> tuple:
    """
    單一組合的完整 Walk-Forward 預測。
    回傳 (daily_preds, feature_log)：
      daily_preds:  {date_id: prediction}
      feature_log:  {date_id: [feature1, ...]}（僅 use_fs=True 時有內容）
    """
    predict_fns = {
        'lstm': _predict_one_day_lstm,
        'xgb':  _predict_one_day_xgb,
        'reg':  _predict_one_day_reg,
    }
    predict_fn = predict_fns[model_type]

    daily_preds      = {}
    feature_log      = {}   # 記錄每次 PI 重選後的 feature 清單
    current_features = None  # 每個 config 開始時清空，第一天強制跑 PI

    for day_idx, current_date in enumerate(test_dates):

        # 取出訓練窗口
        # 訓練窗口截止到 current_date - 2（即 < current_date - 1）：
        # date_id = t 的 Y（forward_return）是 t→t+1 的報酬，
        # 需要到 t+1 收盤才能確認。current_date - 1 的 Y 要到
        # current_date 收盤才知道，訓練時不應使用，避免 Lookahead Bias。
        train_start  = current_date - ROLLING_WINDOW - 1
        train_window = df[
            (df[ID_COL] >= train_start) &
            (df[ID_COL] <  current_date - 1)
        ].copy()

        if len(train_window) < TIMESTEPS + 10:
            print(f"Day {day_idx}: 訓練資料不足，跳過")
            daily_preds[current_date] = np.nan
            continue

        # ------------------------------------------------------------------
        # Feature Selection：每20天（或第一天）重跑 PI
        # ------------------------------------------------------------------
        if use_fs:
            if current_features is None or day_idx % PI_REFRESH_EVERY == 0:
                print(f"Day {day_idx}: 重跑 PI ({model_type.upper()})...", end=' ')
                current_features = select_top_features_by_pi(
                    train_window      = train_window,
                    candidate_features= all_features,
                    model_type        = model_type,
                    loss_type         = loss_type,
                    seed              = 42 + day_idx
                )
                print(f"選出 {len(current_features)} 個 features。")
                # 記錄本次 PI 選出的 feature 清單（key = 觸發重選的 date_id）
                feature_log[int(current_date)] = current_features.copy()
        else:
            current_features = all_features

        # ------------------------------------------------------------------
        # 測試點資料：current_date 這列的 X → 預測 current_date 的 Y
        # （即今天做決策，預測今天→明天的報酬，current_date 的 Y 在明天才結算）
        # LSTM 需要 TIMESTEPS 天歷史來形成序列，XGB/REG 只需當天 X
        # ------------------------------------------------------------------
        if model_type == 'lstm':
            # 取 current_date 前 TIMESTEPS 天到 current_date 當天的 X
            # 注意：這裡只用 X（features），不用到 current_date 的 Y
            test_row = df[
                (df[ID_COL] >= current_date - TIMESTEPS + 1) &
                (df[ID_COL] <= current_date)
            ].copy()
        else:
            test_row = df[df[ID_COL] == current_date].copy()

        if test_row.empty:
            daily_preds[current_date] = np.nan
            continue

        # ------------------------------------------------------------------
        # 單日預測
        # ------------------------------------------------------------------
        try:
            pred = predict_fn(train_window, test_row, current_features, loss_type)
        except Exception as e:
            print(f"Day {day_idx} 預測失敗: {e}")
            pred = np.nan

        daily_preds[current_date] = pred

        if day_idx % 10 == 0:
            print(f"[{col_name}] 進度: {day_idx + 1}/{len(test_dates)} 天")
            gc.collect()

    return daily_preds, feature_log


# ==============================================================================
# 10. 主程序
# ==============================================================================
def process_rolling_predictions():
    """
    執行所有 60 種組合的 Walk-Forward OOS 預測：
      3 models × 5 datasets × 2 (FS/No FS) × 2 (huber/mse) = 60 欄
    """
    datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

    # 60 種組合
    configs = [
        (model_type, use_fs, loss_type)
        for model_type in ['lstm', 'xgb', 'reg']
        for use_fs     in [True, False]
        for loss_type  in ['huber', 'mse']
    ]

    # 斷點續傳
    if os.path.exists(MATRIX_FILE):
        matrix_df = pd.read_csv(MATRIX_FILE)
        print(f"偵測到既有矩陣，已完成 {len(matrix_df.columns) - 2} 個模型欄位。")
    else:
        matrix_df = None

    # ------------------------------------------------------------------
    # 外層迴圈：依資料集
    # ------------------------------------------------------------------
    for ds_file in datasets:
        ds_name = ds_file.replace('.csv', '')
        match   = re.search(r'\d+', ds_name)
        ds_num  = match.group() if match else ds_name

        print(f"\n{'='*65}")
        print(f"資料集: {ds_name}")

        df = pd.read_csv(os.path.join(DATA_DIR, ds_file))
        df = df.sort_values(ID_COL).reset_index(drop=True)

        exclude_cols = {ID_COL, TARGET_COL, 'meta_atr', 'meta_spread'}
        all_features = [c for c in df.columns if c not in exclude_cols]

        max_date_id   = df[ID_COL].max()
        test_start_id = max_date_id - OOS_DAYS + 1
        test_dates    = sorted(df[df[ID_COL] >= test_start_id][ID_COL].unique())

        # 初始化大表（只在首次建立）
        if matrix_df is None:
            matrix_df = (df[df[ID_COL] >= test_start_id]
                         [[ID_COL, TARGET_COL]].copy()
                         .rename(columns={TARGET_COL: 'actual_return'})
                         .reset_index(drop=True))
            matrix_df.to_csv(MATRIX_FILE, index=False)
            print(f"預測矩陣初始化，OOS 共 {len(test_dates)} 天。")

        # ------------------------------------------------------------------
        # 內層迴圈：依組合
        # ------------------------------------------------------------------
        for model_type, use_fs, loss_type in configs:
            prefix   = model_type.upper()
            col_name = (
                f"{prefix}_ds{ds_num}"
                f"_fs{'Y' if use_fs else 'N'}"
                f"_{loss_type}"
            )

            if col_name in matrix_df.columns:
                print(f"跳過 {col_name}（已完成）")
                continue

            print(f"啟動: {col_name}")

            daily_preds, feature_log = _run_rolling(
                df           = df,
                all_features = all_features,
                test_dates   = test_dates,
                model_type   = model_type,
                use_fs       = use_fs,
                loss_type    = loss_type,
                col_name     = col_name,
            )

            matrix_df[col_name] = matrix_df[ID_COL].map(daily_preds)
            matrix_df.to_csv(MATRIX_FILE, index=False)

            # Feature log 存成 JSON（use_fs=False 時為空 dict，仍存檔以利完整性）
            log_path = os.path.join(FEAT_LOG_DIR, f'{col_name}.json')
            with open(log_path, 'w') as f:
                json.dump(feature_log, f, indent=2)
            print(f"{col_name} 完成並寫入矩陣，feature log 已存至 {log_path}")

        del df
        gc.collect()

    # ------------------------------------------------------------------
    # OOS 指標總表
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("計算所有 60 種組合的 OOS 指標...")

    pred_cols = [c for c in matrix_df.columns if c not in [ID_COL, 'actual_return']]
    summary   = []
    for col in pred_cols:
        m = calculate_metrics(
            matrix_df['actual_return'].values,
            matrix_df[col].values,
            label=col
        )
        if m:
            summary.append(m)

    pd.DataFrame(summary).to_csv(METRICS_FILE, index=False)
    print(f"Pipeline 02 全部完成！")
    print(f"   預測矩陣 ({len(pred_cols)} 欄): {MATRIX_FILE}")
    print(f"   指標總表              : {METRICS_FILE}")


# ==============================================================================
if __name__ == "__main__":
    process_rolling_predictions()