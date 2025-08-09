# main.py (Final Version - Based on your latest code with Granger Analysis added)

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Local script imports
from crypto_pipeline import stage1_etl as crypto_stage1, stage2_feature_engineering as crypto_stage2
from sentiment_pipeline import (
    stage1_collect_news as sentiment_stage1,
    stage2_add_columns as sentiment_stage2,
    stage3_sentiment_and_plots as sentiment_stage3
)
from integrated_pipeline import integrate_features
from run_ml_integrated import run_ml_training
from run_portfolios_integrated import run_portfolio_construction
from news_pipeline import run_granger_causality_analysis # <-- ADDED IMPORT

# --- Configuration (from your latest file) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = {
    "API_KEY": "YOUR_API_KEY_HERE",
    "NEWS_START_DATE": "2023-01-01",
    "NEWS_END_DATE": "2025-07-31",
    "CRYPTO_PAGES": [1, 2],
    "CRYPTO_TOP_LIMIT": 100,
    "CRYPTO_HISTORY_LIMIT": 2000,
    "ML_INITIAL_TRAIN_DAYS": 500,
    "ML_RETRAIN_EVERY_DAYS": 30,
    "PORTFOLIO_QUANTILES": 5,
    "BASE_DIR": Path("./results"),
    "CRYPTO_DATA_DIR": Path("./results/crypto_data"),
    "NEWS_DATA_DIR": Path("./results/news_data"),
    "INTEGRATED_DIR": Path("./results/integrated_results"),
    "FIGURES_DIR": Path("./results/figures"),
}
CONFIG.update({
    "CRYPTO_RAW_PATH": CONFIG["CRYPTO_DATA_DIR"] / "stage_1_crypto_data.csv",
    "CRYPTO_FEATURES_PATH": CONFIG["CRYPTO_DATA_DIR"] / "stage_2_crypto_data.csv",
    "NEWS_RAW_PATH": CONFIG["NEWS_DATA_DIR"] / "stage_1_news_raw.csv",
    "SENTIMENT_TIMESERIES_PATH": CONFIG["NEWS_DATA_DIR"] / "compound_timeseries_daily.csv",
    "INTEGRATED_FEATURES_PATH": CONFIG["INTEGRATED_DIR"] / "integrated_features.csv",
    "ML_PREDICTIONS_PATH": CONFIG["INTEGRATED_DIR"] / "ml_predictions_integrated.csv",
    "PORTFOLIO_RETURNS_PATH": CONFIG["INTEGRATED_DIR"] / "integrated_portfolios.csv",
})
for dir_path in [CONFIG["CRYPTO_DATA_DIR"], CONFIG["NEWS_DATA_DIR"], CONFIG["INTEGRATED_DIR"], CONFIG["FIGURES_DIR"]]:
    dir_path.mkdir(parents=True, exist_ok=True)


# --- Main Orchestrator ---
def run_complete_pipeline():
    try:
        # --- STAGE 1: DATA COLLECTION ---
        logging.info("="*20 + " STAGE 1: DATA COLLECTION " + "="*20)
        if not CONFIG["CRYPTO_RAW_PATH"].exists():
            logging.info("Raw crypto data not found. Collecting...")
            crypto_stage1(api_key=CONFIG["API_KEY"], pages=CONFIG["CRYPTO_PAGES"], top_limit=CONFIG["CRYPTO_TOP_LIMIT"], history_limit=CONFIG["CRYPTO_HISTORY_LIMIT"], data_dir=CONFIG["CRYPTO_DATA_DIR"], filename=CONFIG["CRYPTO_RAW_PATH"].name)
        else:
            logging.info("Raw crypto data found. Skipping collection.")
        if not CONFIG["NEWS_RAW_PATH"].exists():
            logging.info("Raw news data not found. Collecting...")
            start_date = datetime.strptime(CONFIG["NEWS_START_DATE"], "%Y-%m-%d")
            end_date = datetime.strptime(CONFIG["NEWS_END_DATE"], "%Y-%m-%d")
            sentiment_stage1(api_key=None, start_dt=start_date, end_dt=end_date, data_dir=CONFIG["NEWS_DATA_DIR"], filename=CONFIG["NEWS_RAW_PATH"].name)
        else:
            logging.info("Raw news data found. Skipping collection.")

        # --- STAGE 2: FEATURE ENGINEERING & INTEGRATION ---
        logging.info("="*20 + " STAGE 2: FEATURE ENGINEERING & INTEGRATION " + "="*20)
        if not CONFIG["INTEGRATED_FEATURES_PATH"].exists():
            logging.info("Integrated features not found. Generating...")
            crypto_raw_df = pd.read_csv(CONFIG["CRYPTO_RAW_PATH"])
            news_raw_df = pd.read_csv(CONFIG["NEWS_RAW_PATH"])
            crypto_stage2(tidy_prices=crypto_raw_df, data_dir=CONFIG["CRYPTO_DATA_DIR"], filename=CONFIG["CRYPTO_FEATURES_PATH"].name)
            
            clean_news_df = sentiment_stage2(news_raw_df)
            
            dirs_for_sentiment = {'news_data_dir': CONFIG["NEWS_DATA_DIR"], 'fig_dir': CONFIG["FIGURES_DIR"]}
            sentiment_stage3(clean_news_df, dirs_for_sentiment, resample_rule='D')
            
            integrate_features(crypto_path=CONFIG["CRYPTO_FEATURES_PATH"], sentiment_path=CONFIG["SENTIMENT_TIMESERIES_PATH"], output_dir=CONFIG["INTEGRATED_DIR"])
        else:
            logging.info("Integrated features found. Skipping generation.")

        # --- STAGE 3: MACHINE LEARNING PREDICTION ---
        logging.info("="*20 + " STAGE 3: MACHINE LEARNING PREDICTION " + "="*20)
        if not CONFIG["ML_PREDICTIONS_PATH"].exists():
            logging.info("ML predictions not found. Training models...")
            integrated_df = pd.read_csv(CONFIG["INTEGRATED_FEATURES_PATH"])
            run_ml_training(integrated_features_df=integrated_df, out_csv=CONFIG["ML_PREDICTIONS_PATH"], init_train_days=CONFIG["ML_INITIAL_TRAIN_DAYS"], retrain_every=CONFIG["ML_RETRAIN_EVERY_DAYS"])
        else:
            logging.info("ML predictions found. Skipping training.")

        # --- STAGE 4: PORTFOLIO CONSTRUCTION & BACKTESTING ---
        logging.info("="*20 + " STAGE 4: PORTFOLIO CONSTRUCTION & BACKTESTING " + "="*20)
        if not CONFIG["PORTFOLIO_RETURNS_PATH"].exists():
            logging.info("Portfolio results not found. Running backtest...")
            run_portfolio_construction(panel_path=CONFIG["INTEGRATED_FEATURES_PATH"], ml_path=CONFIG["ML_PREDICTIONS_PATH"], out_csv=CONFIG["PORTFOLIO_RETURNS_PATH"], fig_dir=CONFIG["FIGURES_DIR"], n_qtiles=CONFIG["PORTFOLIO_QUANTILES"])
        else:
            logging.info("Portfolio results found. Skipping backtest.")

        # --- NEW STAGE 5: CAUSALITY ANALYSIS ---
        logging.info("="*20 + " STAGE 5: CAUSALITY ANALYSIS " + "="*20)
        run_granger_causality_analysis(
            integrated_results_dir=CONFIG["INTEGRATED_DIR"],
            portfolio_returns_path=CONFIG["PORTFOLIO_RETURNS_PATH"],
            figures_dir=CONFIG["FIGURES_DIR"]
        )
        
        logging.info("\n====== PIPELINE HAS COMPLETED SUCCESSFULLY! ======")
        logging.info(f"All outputs are located in the '{CONFIG['BASE_DIR']}' directory.")

    except Exception as e:
        logging.error("An error occurred during the pipeline execution: %s", e, exc_info=True)
        logging.error("Pipeline aborted.")

# --- Execution Entry Point ---
if __name__ == "__main__":
    run_complete_pipeline()