import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Since this script runs from model/ directory, base data is in ../feature_engineering/datasets/base_processed.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '../feature_engineering/datasets/base_processed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'pipeline_data', 'datasets')

TARGET_COL = 'market_forward_excess_returns'
RAW_RETURN_COL = 'forward_returns'
ID_COL = 'date_id'
EXCLUDE_COLS = [TARGET_COL, RAW_RETURN_COL, ID_COL, 'risk_free_rate']

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_feature_groups(df):
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    group_a = [] # PctChange: M, P, V, E
    group_b = [] # Diff: S, I
    group_c = [] # Raw: D, _is_missing
    
    for c in cols:
        if c.endswith('_is_missing'):
            group_c.append(c)
            continue
            
        prefix = c[0]
        if prefix == 'D':
            group_c.append(c)
        elif prefix in ['S', 'I']:
            group_b.append(c)
        elif prefix in ['M', 'P', 'V', 'E']:
            group_a.append(c)
        else:
            group_a.append(c)
            
    return group_a, group_b, group_c

def create_delta_features(df, group_a, group_b, group_c):
    df_delta = pd.DataFrame(index=df.index)
    
    if group_a:
        pct = df[group_a].pct_change(fill_method=None)
        pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0)
        pct.columns = [f'pct_{c}' for c in group_a]
        df_delta = pd.concat([df_delta, pct], axis=1)
        
    if group_b:
        diff = df[group_b].diff().fillna(0)
        diff.columns = [f'diff_{c}' for c in group_b]
        df_delta = pd.concat([df_delta, diff], axis=1)
        
    if group_c:
        raw = df[group_c].copy()
        df_delta = pd.concat([df_delta, raw], axis=1)
        
    return df_delta

def create_historical_features(df, lags=60):
    if RAW_RETURN_COL not in df.columns:
        return pd.DataFrame(index=df.index)
        
    realized_ret = df[RAW_RETURN_COL].shift(1).fillna(0)
    log_ret = np.log1p(realized_ret).replace([np.inf, -np.inf], 0).fillna(0)
    
    features = {}
    for i in range(1, lags + 1):
        features[f'log_ret_lag_{i}'] = log_ret.shift(i - 1).fillna(0)
        features[f'log_ret_cum_{i}d'] = log_ret.rolling(window=i).sum().fillna(0)
        
    return pd.DataFrame(features, index=df.index)

def create_advanced_features(df):
    """
    Creates ATR_proxy and Breadth_proxy.
    """
    adv = pd.DataFrame(index=df.index)
    
    # 1. ATR Proxy: Max of short (5d) and long (20d) volatility of realized returns
    print("Calculating ATR Proxy...")
    realized_ret = df[RAW_RETURN_COL].shift(1).fillna(0)
    atr_5 = realized_ret.rolling(window=5, min_periods=1).std().fillna(0)
    atr_20 = realized_ret.rolling(window=20, min_periods=1).std().fillna(0)
    
    # Avoid zero ATR by adding a tiny epsilon
    eps = 1e-6
    atr_proxy = np.maximum(atr_5, atr_20) + eps
    adv['atr_proxy'] = atr_proxy
    
    # 2. Market Breadth Divergence Proxy
    print("Analyzing Momentum Factors for Breadth Proxy...")
    m_cols = [c for c in df.columns if c.startswith('M') and not c.endswith('_is_missing')]
    
    autocorrs = {}
    for c in m_cols:
        autocorrs[c] = df[c].autocorr(lag=1)
    
    # Sort by autocorrelation
    sorted_m = sorted(autocorrs.items(), key=lambda x: x[1])
    fast_m = sorted_m[0][0]   # Lowest autocorrelation (most erratic/fast)
    slow_m = sorted_m[-1][0]  # Highest autocorrelation (most smooth/trend)
    
    print(f"  -> Identified Fast Momentum: {fast_m} (autocorr: {autocorrs[fast_m]:.3f})")
    print(f"  -> Identified Slow Momentum: {slow_m} (autocorr: {autocorrs[slow_m]:.3f})")
    
    # Standardize them globally or rolling? Time series strictly backward looking
    # We will use 252-day rolling Z-score.
    window = 252
    def rolling_zscore(s):
        r = s.rolling(window=window, min_periods=1)
        mean = r.mean()
        std = r.std().replace(0, np.nan)
        z = (s - mean) / std
        return z.fillna(0)
        
    z_fast = rolling_zscore(df[fast_m])
    z_slow = rolling_zscore(df[slow_m])
    
    breadth_spread = z_slow - z_fast
    adv['breadth_spread'] = breadth_spread
    
    return adv

def process_datasets():
    print(f"Reading base dataset: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Base file not found. Ensure EDA step is complete.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(ID_COL).reset_index(drop=True)
    
    # 1. Identify Groups
    group_a, group_b, group_c = get_feature_groups(df)
    
    # 2. Generate Components
    print("Generating standard subset features...")
    # X (Original)
    x_cols = group_a + group_b + group_c
    ds_x = df[x_cols].copy()
    
    # Delta X
    ds_delta = create_delta_features(df, group_a, group_b, group_c)
    
    # R (Historical)
    ds_r = create_historical_features(df)
    
    # Advanced (ATR and Breadth)
    ds_adv = create_advanced_features(df)
    
    # Meta
    meta_cols = [c for c in [ID_COL, TARGET_COL] if c in df.columns]
    meta_df = df[meta_cols].copy()
    meta_df['meta_atr'] = ds_adv['atr_proxy']
    meta_df['meta_spread'] = ds_adv['breadth_spread']
    
    # 3. Assemble and Save 5 Datasets
    print("Assembling DataFrames...")
    datasets = {
        'dataset_1_Base': ds_x,
        'dataset_2_Delta': ds_delta,
        'dataset_3_Historical': ds_r,
        'dataset_4_Original_plus_Hist': pd.concat([ds_x, ds_r], axis=1),
        'dataset_5_Delta_plus_Hist': pd.concat([ds_delta, ds_r], axis=1)
    }
    
    for name, data in datasets.items():
        final_df = pd.concat([meta_df, data], axis=1)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved {name}.csv (Shape: {final_df.shape})")
        
    print("\nStage 1 Pipeline Completed Successfully.")

if __name__ == "__main__":
    process_datasets()
