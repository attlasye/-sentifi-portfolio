# integrated_pipeline.py
# FINAL VERSION: The single source of truth for all feature engineering

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_crypto_features(crypto_path: Path) -> pd.DataFrame:
    logging.info(f"Loading crypto features from {crypto_path}")
    df = pd.read_csv(crypto_path)
    df['date'] = pd.to_datetime(df['date'])
    logging.info(f"Loaded {len(df)} rows of crypto data")
    return df

def load_sentiment_features(sentiment_path: Path) -> pd.DataFrame:
    logging.info(f"Loading sentiment features from {sentiment_path}")
    df = pd.read_csv(sentiment_path)
    df['date'] = pd.to_datetime(df['date'])
    if 'Overall' in df.columns:
        df = df.rename(columns={'Overall': 'sentiment_overall'})
    df = df.bfill().ffill().fillna(0)
    logging.info(f"Loaded sentiment data with shape {df.shape}")
    return df

def create_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Creating sentiment-based features...")
    df['sentiment_momentum'] = df['sentiment_overall'].diff()
    df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
    df['sentiment_volatility'] = df['sentiment_overall'].rolling(window=14, min_periods=7).std().bfill().ffill()
    df['extreme_sentiment'] = ((df['sentiment_overall'] > 70) | (df['sentiment_overall'] < 30)).astype(int)
    return df

def create_interaction_features(
    crypto_df: pd.DataFrame,
    sentiment_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    logging.info("Creating interaction features")
    merged_df = pd.merge(crypto_df, sentiment_df, on='date', how='inner')
    interactions = []
    
    if 'momentum_14_z' in merged_df.columns and 'sentiment_overall' in merged_df.columns:
        merged_df['sentiment_momentum_interaction'] = merged_df['momentum_14_z'] * merged_df['sentiment_overall'] / 50
        interactions.append('sentiment_momentum_interaction')
    
    if 'v_14d_z' in merged_df.columns and 'news_volume' in merged_df.columns:
        merged_df['volume_news_interaction'] = merged_df['v_14d_z'] * np.log1p(merged_df['news_volume'])
        interactions.append('volume_news_interaction')
    
    if 'volatility_14_z' in merged_df.columns and 'sentiment_volatility' in merged_df.columns:
        merged_df['vol_sentiment_divergence'] = merged_df['volatility_14_z'] - (merged_df['sentiment_volatility'] / merged_df['sentiment_volatility'].std())
        interactions.append('vol_sentiment_divergence')
    
    if 'var_14_z' in merged_df.columns and 'sentiment_overall' in merged_df.columns:
        merged_df['risk_adjusted_sentiment'] = merged_df['sentiment_overall'] / (1 + np.abs(merged_df['var_14_z']))
        interactions.append('risk_adjusted_sentiment')

    # --- ADD NEW ADVANCED SENTIMENT FEATURES ---
    if 'sentiment_momentum' in merged_df.columns and 'sentiment_acceleration' in merged_df.columns:
        merged_df['sentiment_signal_strength'] = merged_df['sentiment_momentum'] * merged_df['sentiment_acceleration']
        interactions.append('sentiment_signal_strength')
    
    if 'momentum_14_z' in merged_df.columns and 'sentiment_momentum' in merged_df.columns:
        merged_df['sentiment_price_divergence'] = (
            (np.sign(merged_df['momentum_14_z']) != np.sign(merged_df['sentiment_momentum'])).astype(int)
        )
        interactions.append('sentiment_price_divergence')
    # -------------------------------------------
    
    return merged_df, interactions

def scale_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    logging.info("Scaling features cross-sectionally")
    scaled_dfs = []
    for date in df['date'].unique():
        date_df = df[df['date'] == date].copy()
        for col in feature_cols:
            if date_df[col].nunique() > 1:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                date_df[f'{col}_z'] = scaler.fit_transform(date_df[[col]])
            else:
                date_df[f'{col}_z'] = 0
        scaled_dfs.append(date_df)
    result = pd.concat(scaled_dfs, ignore_index=True)
    return result

def create_lagged_features(df: pd.DataFrame, feature_cols: List[str], lags: List[int] = [1]) -> Tuple[pd.DataFrame, List[str]]:
    logging.info(f"Creating lagged features for {len(feature_cols)} columns")
    df = df.sort_values(['symbol', 'date'])
    for lag in lags:
        for col in feature_cols:
            df[f'{col}_lag{lag}'] = df.groupby('symbol')[col].shift(lag)
    lag_cols = [col for col in df.columns if '_lag' in col]
    df = df.dropna(subset=lag_cols)
    return df, lag_cols

def integrate_features(
    crypto_path: Path,
    sentiment_path: Path,
    output_dir: Path,
    save_intermediate: bool = True
) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    
    logging.info("="*60)
    logging.info("Starting feature integration pipeline")
    logging.info("="*60)
    
    crypto_df = load_crypto_features(crypto_path)
    sentiment_df = load_sentiment_features(sentiment_path)
    
    sentiment_df = create_sentiment_features(sentiment_df)
    integrated_df, interaction_cols = create_interaction_features(crypto_df, sentiment_df)
    
    crypto_features_orig = [col for col in crypto_df.columns if col.endswith('_z')]
    sentiment_features_orig = [col for col in sentiment_df.columns if col.startswith('sentiment_')]
    features_to_scale = sentiment_features_orig + interaction_cols
    
    integrated_df = scale_features(integrated_df, features_to_scale)
    
    all_features = crypto_features_orig + [f'{col}_z' for col in features_to_scale]
    
    integrated_df, lag_cols = create_lagged_features(integrated_df, all_features)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "integrated_features.csv"
    integrated_df.to_csv(output_path, index=False)
    
    feature_info = pd.DataFrame({'feature': lag_cols, 'type': ['crypto' if 'sentiment' not in f and 'interaction' not in f else 'sentiment' if 'sentiment' in f else 'interaction' for f in lag_cols]})
    feature_info.to_csv(output_dir / "feature_info.csv", index=False)
    
    logging.info("="*60)
    logging.info("Integration Summary:")
    logging.info(f"Total samples: {len(integrated_df)}")
    logging.info(f"Date range: {integrated_df['date'].min()} to {integrated_df['date'].max()}")
    logging.info(f"Number of symbols: {integrated_df['symbol'].nunique()}")
    logging.info(f"Total features used: {len(lag_cols)}")
    logging.info("="*60)
    
    return integrated_df
    
if __name__ == "__main__":
    crypto_path = Path("./results/stage_2_crypto_data.csv")
    sentiment_path = Path("./results/news_data/compound_timeseries_daily.csv")
    output_dir = Path("./results/integrated_results")
    
    integrated_df = integrate_features(crypto_path, sentiment_path, output_dir)