# run_ml_integrated.py
# FINAL VERSION with streamlined pipeline and bug fixes

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper functions ---
def build_predefined_split(dates: pd.Series, train_frac: float = 0.70) -> np.ndarray:
    uniq = np.sort(dates.unique())
    cut_date = uniq[int(len(uniq) * train_frac)]
    return np.where(dates <= cut_date, -1, 0)

def fit_ridge(x: np.ndarray, y: np.ndarray, ps: PredefinedSplit) -> RidgeCV:
    model = RidgeCV(alphas=[0.02, 0.05, 0.10], cv=ps, scoring="neg_mean_squared_error")
    return model.fit(x, y.ravel())

def fit_hgb(x: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=None,
        max_iter=300,
        l2_regularization=0.0,
        early_stopping=True,
        random_state=42,
    )
    return model.fit(x.astype(np.float32), y.ravel())

def fit_lgbm(x: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        num_leaves=8,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )
    return model.fit(x, y.ravel())

def fit_ols(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    return LinearRegression(n_jobs=-1).fit(x, y.ravel())

def select_top_features(X_train: pd.DataFrame, y_train: pd.Series, n_features: int = 30) -> List[str]:
    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_train.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X_train, y_train)
    scores = pd.DataFrame({'feature': X_train.columns, 'score': selector.scores_}).sort_values('score', ascending=False)
    logger.info("Top 10 features:\n%s", scores.head(10).to_string(index=False))
    return X_train.columns[selector.get_support()].tolist()

# FIX: Changed function to use the models dictionary keys
def dynamic_ensemble_weights(val_df: pd.DataFrame, models: Dict, target: str, predictors: List[str]) -> Tuple[float, float, float]:
    y_true = val_df[target]
    ridge_mse = mean_squared_error(y_true, models['ridge'].predict(val_df[predictors]))
    hgb_mse = mean_squared_error(y_true, models['hgb'].predict(val_df[predictors]))
    ols_mse = mean_squared_error(y_true, models['ols'].predict(val_df[predictors]))
    
    # Handle potential division by zero
    ridge_inv_mse = 1 / ridge_mse if ridge_mse > 1e-8 else 0
    hgb_inv_mse = 1 / hgb_mse if hgb_mse > 1e-8 else 0
    ols_inv_mse = 1 / ols_mse if ols_mse > 1e-8 else 0
    
    total_inverse_mse = ridge_inv_mse + hgb_inv_mse + ols_inv_mse
    
    if total_inverse_mse == 0:
        return 0.33, 0.33, 0.34 # Fallback to equal weights
    
    w_ridge = ridge_inv_mse / total_inverse_mse
    w_hgb = hgb_inv_mse / total_inverse_mse
    w_ols = ols_inv_mse / total_inverse_mse
    
    return w_ridge, w_hgb, w_ols

def create_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating multi-horizon targets (3, 5, 7 days)")
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])
    df['close'] = df.groupby('symbol')['close'].ffill()
    
    for days in [3, 5, 7]:
        df[f'return_{days}d'] = df.groupby('symbol')['close'].transform(
            lambda x: (x.shift(-days) / x - 1)
        )
    
    df['target_return'] = df['return_5d'].fillna(df['return'])
    df = df.dropna(subset=['target_return'])
    return df

# Main routine
def run_ml_training(
    integrated_features_df: pd.DataFrame,
    out_csv: Path,
    init_train_days: int = 500,
    retrain_every: int = 30
) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    df = integrated_features_df.copy()
    df = create_multi_horizon_targets(df)
    
    target = 'target_return'
    predictors = [col for col in df.columns if col.endswith('_z_lag1')]
    
    if not predictors or target not in df.columns:
        logger.error("Required columns ('target_return' or '_z_lag1' features) not found.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(["date", "symbol"]).sort_index(level="date")
    df = df.dropna(subset=[target] + predictors)
    
    crypto_features = [f for f in predictors if 'sentiment' not in f and 'interaction' not in f]
    sentiment_features = [f for f in predictors if 'sentiment' in f]
    interaction_features = [f for f in predictors if 'interaction' in f]
    
    logger.info("="*60)
    logger.info("Feature Summary:")
    logger.info(f"Total features: {len(predictors)}")
    logger.info(f"Crypto features: {len(crypto_features)}")
    logger.info(f"Sentiment features: {len(sentiment_features)}")
    logger.info(f"Interaction features: {len(interaction_features)}")
    logger.info("="*60)
    
    unique_dates = df.index.get_level_values("date").unique().sort_values()
    dates_np = unique_dates.to_numpy(dtype='datetime64[D]')
    
    first_cut_date = dates_np[0] + np.timedelta64(init_train_days - 1, "D")
    first_idx = np.searchsorted(dates_np, first_cut_date, side="right")
    
    retrain_idx = range(first_idx, len(dates_np), retrain_every)
    preds_collect: list[pd.DataFrame] = []
    
    selected_features = predictors
    
    for idx, cut in enumerate(retrain_idx):
        win_end = dates_np[cut]
        train_mask = df.index.get_level_values("date") <= win_end
        train_df = df.loc[train_mask]
        
        split_labels = build_predefined_split(
            train_df.index.get_level_values("date"), train_frac=0.70
        )
        ps = PredefinedSplit(split_labels)
        
        if idx == 0:
            selected_features = select_top_features(
                train_df[predictors],
                train_df[target],
                n_features=30
            )

        x_train = train_df[selected_features].values
        y_train = train_df[target].values
        
        ridge_model = fit_ridge(x_train, y_train, ps)
        hgb_model = fit_lgbm(x_train, y_train) # LightGBM
        ols_model = fit_ols(x_train, y_train)
        
        next_cut = cut + retrain_every if idx + 1 < len(retrain_idx) else len(dates_np)
        pred_dates = dates_np[cut + 1 : next_cut]
        if pred_dates.size == 0: continue
        
        pred_dates_ts = pd.to_datetime(pred_dates)
        pred_mask = df.index.get_level_values("date").isin(pred_dates_ts)
        pred_df = df.loc[pred_mask, selected_features].copy()
        if pred_df.empty: continue

        ridge_vals = ridge_model.predict(pred_df.values)
        hgb_vals = hgb_model.predict(pred_df.values.astype(np.float32))
        ols_vals = ols_model.predict(pred_df.values)
        
        models = {'ridge': ridge_model, 'hgb': hgb_model, 'ols': ols_model}
        weights = dynamic_ensemble_weights(train_df, models, target, selected_features)
        w_ridge, w_hgb, w_ols = weights
        
        ensemble_vals = (w_ridge * ridge_vals + w_hgb * hgb_vals + w_ols * ols_vals)
        
        out_df = pred_df.reset_index()[["date", "symbol"]]
        out_df["ridge_return_pred"] = ridge_vals
        out_df["hgb_return_pred"] = hgb_vals
        out_df["ols_return_pred"] = ols_vals
        out_df["ensemble_return_pred"] = ensemble_vals
        
        preds_collect.append(out_df)
        
        win_end_dt = pd.Timestamp(win_end)
        logger.info(
            f"Trained through {win_end_dt.date()} | "
            f"ridge_alpha={ridge_model.alpha_:g} | "
            f"ensemble_weights=({w_ridge:.2f}, {w_hgb:.2f}, {w_ols:.2f})"
        )
    
    if not preds_collect:
        logger.warning("No predictions were generated. Check date range and data quality.")
        return

    final_preds = pd.concat(preds_collect, ignore_index=True)
    final_preds.to_csv(out_csv, index=False)
    logger.info(f"Saved predictions to {out_csv.resolve()} ({len(final_preds):,} rows).")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_ml_integrated",
        description="Expanding-window ML for crypto + sentiment features.",
    )
    p.add_argument("--integrated-csv", required=True, help="Integrated features CSV")
    p.add_argument("--out", required=True, help="Output CSV for predictions")
    p.add_argument("--init-train-days", type=int, default=500, help="Initial training window")
    p.add_argument("--retrain-every", type=int, default=30, help="Retrain cadence")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    integrated_df = pd.read_csv(args.integrated_csv)
    run_ml_training(
        integrated_features_df=integrated_df,
        out_csv=Path(args.out),
        init_train_days=args.init_train_days,
        retrain_every=args.retrain_every,
    )