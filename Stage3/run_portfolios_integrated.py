# run_portfolios_integrated.py
# FINAL VERSION with enhanced strategies and advanced backtesting
# MODIFIED to add data export for Station 4

#!/usr/bin/env python
from sklearn.metrics import r2_score
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyBondLab as pbl
import logging
from typing import Dict, Tuple, Optional, List

# Local imports
from portfolio_optimizer import PortfolioOptimizer

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True})
logger = logging.getLogger(__name__)

# --- Helper Functions (No changes here) ---
def load_and_merge(panel_path: Path, ml_path: Path) -> pd.DataFrame:
    df_panel = pd.read_csv(panel_path)
    df_panel["date"] = pd.to_datetime(df_panel["date"])
    if "index" in df_panel.columns:
        df_panel = df_panel.drop(columns=["index"])
    
    df_ml = pd.read_csv(ml_path)
    df_ml["date"] = pd.to_datetime(df_ml["date"])
    
    df = df_panel.merge(df_ml, on=["date", "symbol"], how="inner")
    
    df['close'] = df_panel.set_index(['date', 'symbol'])['close'].loc[df.set_index(['date', 'symbol']).index].values

    if "usd_volume" in df.columns:
        df["ID"] = pd.factorize(df["symbol"])[0] + 1
        df = (df.sort_values(["symbol", "date"])
              .rename(columns={"return": "ret"})
              .assign(VW=lambda d: d["usd_volume"], RATING_NUM=1)
              .sort_values(["ID", "date"])
              .reset_index(drop=True))
    else:
        df = df.sort_values(["symbol", "date"]).rename(columns={"return": "ret"})
    
    return df

def build_single_sort(df: pd.DataFrame, signal_col: str, n_q: int) -> pd.Series:
    if not all(col in df.columns for col in ['ID', 'VW', 'RATING_NUM']):
        raise ValueError("Missing required columns for PyBondLab.")
    
    ss = pbl.SingleSort(1, signal_col, n_q)
    res = pbl.StrategyFormation(df, strategy=ss, rating=None).fit()
    ls_df = pd.concat(res.get_long_short(), axis=1)
    ewea_cols = [c for c in ls_df.columns if c.startswith("EWEA_")]
    
    if not ewea_cols:
        raise ValueError(f"EWEA series not found for {signal_col}")
    
    return ls_df[ewea_cols[0]]

def calculate_max_drawdown(returns: pd.Series) -> float:
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_turnover_and_costs(portfolios: pd.DataFrame, transaction_cost: float = 0.001) -> Tuple[pd.DataFrame, pd.Series]:
    net_returns = {}
    turnover_rates = {}
    
    for strategy in portfolios.columns:
        position_changes = np.sign(portfolios[strategy]).diff().abs()
        turnover_rates[strategy] = position_changes.mean() * 252
        costs = position_changes * transaction_cost
        net_returns[strategy] = portfolios[strategy] - costs
    
    return pd.DataFrame(net_returns), pd.Series(turnover_rates)

def build_optimized_portfolios(df: pd.DataFrame, lookback_days: int = 60) -> Dict[str, pd.Series]:
    returns_df = df.pivot(index='date', columns='symbol', values='ret')
    predictions_df = df.pivot(index='date', columns='symbol', values='ensemble_return_pred')
    
    optimizer = PortfolioOptimizer(
        max_position=0.10,
        target_vol=0.40,
        lookback_days=lookback_days
    )
    
    optimized_portfolios = {
        'MV_OPT': [],
        'RISK_PARITY_ML': [],
        'RISK_PARITY_EW': [],
        'VOL_TARGET_EW': [],
        'VOL_TARGET_ML': [],
    }
    
    dates = returns_df.index[lookback_days:]
    
    for date in dates:
        hist_end_idx = returns_df.index.get_loc(date)
        hist_start_idx = max(0, hist_end_idx - lookback_days)
        hist_returns = returns_df.iloc[hist_start_idx:hist_end_idx]
        
        if len(hist_returns) < 20: continue
        
        valid_assets = hist_returns.columns[hist_returns.count() > len(hist_returns) * 0.8]
        hist_returns_clean = hist_returns[valid_assets].fillna(0)
        
        if len(valid_assets) < 5: continue
        
        try:
            cov_matrix_rp = hist_returns_clean.cov()
            if cov_matrix_rp.isnull().any().any():
                continue

            # 1. Mean-Variance Optimization (ML-driven)
            if date in predictions_df.index:
                expected_returns_mv = predictions_df.loc[date][valid_assets].dropna()
                if len(expected_returns_mv) > 0:
                    cov_matrix = hist_returns_clean[expected_returns_mv.index].cov()
                    if not cov_matrix.isnull().any().any():
                        mv_weights = optimizer.mean_variance_optimization(expected_returns_mv, cov_matrix)
                        portfolio_return = (returns_df.loc[date][mv_weights.index] * mv_weights).sum()
                        optimized_portfolios['MV_OPT'].append({'date': date, 'return': portfolio_return})
            
            # 2. ML-Enhanced Risk Parity
            expected_returns_rp = predictions_df.loc[date][valid_assets].dropna() if date in predictions_df.index else pd.Series()
            if not expected_returns_rp.empty:
                rp_ml_weights = optimizer.risk_parity_optimization(cov_matrix_rp, expected_returns=expected_returns_rp, alpha=0.7)
                portfolio_return_ml = (returns_df.loc[date][rp_ml_weights.index] * rp_ml_weights).sum()
                optimized_portfolios['RISK_PARITY_ML'].append({'date': date, 'return': portfolio_return_ml})

            # 3. Pure Risk Parity (Equal Weight version, no ML signal)
            rp_ew_weights = optimizer.risk_parity_optimization(cov_matrix_rp)
            portfolio_return_ew = (returns_df.loc[date][rp_ew_weights.index] * rp_ew_weights).sum()
            optimized_portfolios['RISK_PARITY_EW'].append({'date': date, 'return': portfolio_return_ew})
            
            # 4. Volatility Targeted Equal Weight
            equal_weights = pd.Series(1/len(valid_assets), index=valid_assets)
            vol_adj_ew_weights, _ = optimizer.apply_volatility_target(equal_weights, hist_returns_clean)
            portfolio_return = (returns_df.loc[date][vol_adj_ew_weights.index] * vol_adj_ew_weights).sum()
            optimized_portfolios['VOL_TARGET_EW'].append({'date': date, 'return': portfolio_return})
            
            # 5. Volatility Targeted on ML Portfolio
            if date in predictions_df.index:
                expected_returns_ml = predictions_df.loc[date][valid_assets].dropna()
                if len(expected_returns_ml) > 0:
                    ml_weights = expected_returns_ml.clip(lower=0)
                    if ml_weights.sum() > 1e-8:
                        ml_weights = ml_weights / ml_weights.sum()
                        vol_adj_ml_weights, _ = optimizer.apply_volatility_target(ml_weights, hist_returns_clean)
                        portfolio_return = (returns_df.loc[date][vol_adj_ml_weights.index] * vol_adj_ml_weights).sum()
                        optimized_portfolios['VOL_TARGET_ML'].append({'date': date, 'return': portfolio_return})
            
        except Exception as e:
            logger.warning(f"Optimization failed for date {date}: {str(e)}")
            continue
    
    result = {}
    for strategy, returns_list in optimized_portfolios.items():
        if returns_list:
            result[strategy] = pd.DataFrame(returns_list).set_index('date')['return']
    
    return result

# --- NEW FUNCTION: Export chart data for Station 4 ---
def export_chart_data(portfolios: pd.DataFrame, out_path: Path, metrics: pd.DataFrame):
    """Export all chart data as CSV files for Station 4 to use"""
    
    logger.info("Exporting chart data for Station 4...")
    
    # 1. Export cumulative returns data
    logger.info("Exporting cumulative returns data...")
    cumulative_returns = (1 + portfolios).cumprod()
    cumulative_returns.index.name = 'date'
    cumulative_returns.to_csv(out_path / 'cumulative_returns_data.csv')
    
    # 2. Export cumulative log returns data
    logger.info("Exporting cumulative log returns data...")
    log_returns = np.log(1 + portfolios)
    cumulative_log_returns = log_returns.cumsum()
    cumulative_log_returns.index.name = 'date'
    cumulative_log_returns.to_csv(out_path / 'cumulative_log_returns_data.csv')
    
    # 3. 删除重复计算，直接使用传入的 metrics
    # 不需要 performance_metrics_data.csv，因为我们已经有 metrics_table_sorted.csv
    
    # 4. Export monthly returns data (for heatmap)
    logger.info("Exporting monthly returns data...")
    monthly_returns = portfolios.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
    monthly_returns.index.name = 'month'
    monthly_returns.to_csv(out_path / 'monthly_returns_data.csv')
    
    # 5. Export correlation matrix data
    logger.info("Exporting correlation matrix data...")
    correlation_matrix = portfolios.corr()
    correlation_matrix.to_csv(out_path / 'correlation_matrix_data.csv')
    
    # 6. 删除 strategy_rankings_data.csv - 使用 metrics_table_sorted.csv 即可
    
    # 7. Export daily returns data (raw portfolio returns)
    logger.info("Exporting daily returns data...")
    daily_returns = portfolios.copy()
    daily_returns.index.name = 'date'
    daily_returns.to_csv(out_path / 'daily_returns_data.csv')
    
    # 8. Export drawdown timeline data
    logger.info("Exporting drawdown timeline data...")
    drawdown_data = pd.DataFrame(index=portfolios.index)
    
    for strategy in portfolios.columns:
        cumulative = (1 + portfolios[strategy]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        drawdown_data[strategy] = drawdown
    
    drawdown_data.index.name = 'date'
    drawdown_data.to_csv(out_path / 'drawdown_timeline_data.csv')
    
    logger.info("All chart data exported successfully!")

def run_portfolio_construction(panel_path: Path, ml_path: Path, out_csv: Path, fig_dir: Path, n_qtiles: int):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    df = load_and_merge(panel_path, ml_path)
    
    logger.info("Building traditional quantile-based strategies...")
    ridge_ls = build_single_sort(df, "ridge_return_pred", n_qtiles).rename("RIDGE")
    boost_ls = build_single_sort(df, "hgb_return_pred", n_qtiles).rename("BOOST")
    ols_ls = build_single_sort(df, "ols_return_pred", n_qtiles).rename("OLS")
    ensemble_ls = build_single_sort(df, "ensemble_return_pred", n_qtiles).rename("ENSEMBLE")
    
    ew_ret = df.groupby("date")["ret"].mean().rename("EW")
    btc_df = df[df['symbol'] == 'BTC'].set_index('date')
    if not btc_df.empty:
        btc_ret = btc_df['ret'].rename("BTC")
    else:
        logger.warning("BTC data not found...")
        btc_ret = pd.Series(dtype=float, name="BTC")
    
    logger.info("Building optimized portfolios...")
    optimized_portfolios = build_optimized_portfolios(df, lookback_days=60)
    
    # Consolidate all strategies
    all_strategies = [ridge_ls, boost_ls, ols_ls, ensemble_ls, ew_ret, btc_ret]
    for name, returns in optimized_portfolios.items():
        all_strategies.append(returns.rename(name))
    
    portfolios = pd.concat(all_strategies, axis=1)
    portfolios.index = pd.to_datetime(portfolios.index)
    portfolios = portfolios.dropna()
    
    portfolios.to_csv(out_csv, index=True)
    
    # Calculate metrics
    net_portfolios, turnover_rates = calculate_turnover_and_costs(portfolios)
    mu_daily = portfolios.mean()
    vol_daily = portfolios.std()
    sharpe = mu_daily / vol_daily * np.sqrt(365)
    ann_ret = mu_daily * 365
    ann_vol = vol_daily * np.sqrt(365)
    max_drawdowns = {col: calculate_max_drawdown(portfolios[col]) for col in portfolios.columns}
    net_sharpe = net_portfolios.mean() / net_portfolios.std() * np.sqrt(365)
    
    ir_vs_btc, ir_vs_ew = {}, {}
    benchmarks = {'BTC': portfolios.get('BTC'), 'EW': portfolios.get('EW')}
    for b_name, b_returns in benchmarks.items():
        if b_returns is None: continue
        current_ir_dict = ir_vs_btc if b_name == 'BTC' else ir_vs_ew
        for col in portfolios.columns:
            if col == b_name: continue
            active_return = portfolios[col] - b_returns
            tracking_error = active_return.std()
            if tracking_error > 1e-8:
                current_ir_dict[col] = (active_return.mean() / tracking_error) * np.sqrt(365)
            else: current_ir_dict[col] = np.nan
    
    metrics = pd.DataFrame({
        "Ave.Return(%)": ann_ret * 100,
        "Volatility(%)": ann_vol * 100,
        "Sharpe": sharpe,
        "Net_Sharpe": net_sharpe,
        "Max_Drawdown(%)": pd.Series(max_drawdowns) * 100,
        "Turnover(Annual)": turnover_rates,
        "Info Ratio (vs BTC)": pd.Series(ir_vs_btc),
        "Info Ratio (vs EW)": pd.Series(ir_vs_ew)
    }).round(2)
    
    # Define a logical sort order for the output table
    strategy_order = [
        # Benchmarks
        'BTC', 'EW',
        # Simple ML Strategies
        'OLS', 'RIDGE', 'BOOST', 'ENSEMBLE',
        # Advanced Non-ML Strategies (Risk-Managed Benchmarks)
        'VOL_TARGET_EW', 'RISK_PARITY_EW',
        # Advanced ML-Driven Strategies
        'VOL_TARGET_ML', 'RISK_PARITY_ML', 'MV_OPT',
    ]
    
    # Filter out any strategies that might not have been generated
    existing_strategies_in_order = [s for s in strategy_order if s in metrics.index]
    metrics_sorted = metrics.reindex(existing_strategies_in_order)
    
    # Print and save the sorted table
    print("\n" + "="*60)
    print("PERFORMANCE METRICS (Annualized & Sorted)")
    print("="*60)
    print(metrics_sorted.to_string())
    metrics_path = fig_dir / "metrics_table.txt"
    metrics_sorted.to_csv(fig_dir / "metrics_table_sorted.csv")
    metrics_path.write_text(metrics_sorted.to_string())
    
    # --- NEW: Export chart data before creating plots ---
    export_chart_data(portfolios, out_csv.parent, metrics_sorted)
    
    # Create plots and other outputs
    create_portfolio_summary_plots(portfolios, fig_dir)
    create_monthly_heatmap(portfolios, fig_dir)
    create_correlation_matrix(portfolios, fig_dir)
    create_risk_return_plot(portfolios, fig_dir)
    create_annual_return_barchart(portfolios, fig_dir)
    calculate_statistical_validation(df, fig_dir)
    create_drawdown_timeline(portfolios, fig_dir)
    
    print(f"\n✅ Portfolio construction complete with optimization strategies!")
    print(f"📁 Portfolio returns CSV saved to {out_csv.resolve()}")
    print(f"📊 Added strategies: {', '.join(optimized_portfolios.keys())}")
    print(f"📊 Chart data exported for Station 4 use!")

# --- Plotting and other functions (No changes below this line) ---
def create_monthly_heatmap(portfolios: pd.DataFrame, fig_dir: Path):
    monthly_returns = portfolios.resample('M').apply(lambda x: (1 + x).prod() - 1)
    if len(monthly_returns) > 12:
        monthly_returns = monthly_returns.tail(12)
    monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
    plt.figure(figsize=(12, 8))
    sns.heatmap((monthly_returns.T * 100), annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Monthly Return (%)'})
    plt.title('Monthly Returns Heatmap (%) - Last 12 Months', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Strategy')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    portfolio_fig_dir = fig_dir / "portfolio"
    portfolio_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(portfolio_fig_dir / "monthly_returns_heatmap.png")
    plt.close()
    print("✅ Monthly heatmap plot (Last 12 Months) created.")
    
def create_drawdown_timeline(portfolios: pd.DataFrame, fig_dir: Path):
    """Create a timeline chart showing drawdowns for all strategies"""
    
    plt.figure(figsize=(14, 8))
    
    # Calculate drawdowns for each strategy
    for i, strategy in enumerate(portfolios.columns):
        cumulative = (1 + portfolios[strategy]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100  # Convert to percentage
        
        # Plot with different colors for each strategy
        plt.plot(portfolios.index, drawdown, label=strategy, linewidth=1.5, alpha=0.8)
    
    plt.fill_between(portfolios.index, 0, -100, color='red', alpha=0.1)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=-20, color='orange', linestyle='--', linewidth=0.5, alpha=0.5, label='20% Drawdown')
    plt.axhline(y=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='50% Drawdown')
    
    plt.title('Drawdown Timeline - All Strategies', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    portfolio_fig_dir = fig_dir / "portfolio"
    portfolio_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(portfolio_fig_dir / "drawdown_timeline.png", dpi=150)
    plt.close()
    print("✅ Drawdown timeline plot created.")
    

def create_correlation_matrix(portfolios: pd.DataFrame, fig_dir: Path):
    corr_matrix = portfolios.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title('Portfolio Strategy Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    portfolio_fig_dir = fig_dir / "portfolio"
    portfolio_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(portfolio_fig_dir / "correlation_matrix.png")
    plt.close()
    print("✅ Correlation matrix plot created.")
    
def create_risk_return_plot(portfolios: pd.DataFrame, fig_dir: Path):
    ann_ret = portfolios.mean() * 365 * 100
    ann_vol = portfolios.std() * np.sqrt(365) * 100
    plt.figure(figsize=(10, 7))
    plt.scatter(ann_vol, ann_ret, s=150, alpha=0.7)
    for i, txt in enumerate(portfolios.columns):
        plt.annotate(txt, (ann_vol[i], ann_ret[i]), xytext=(5, 5), textcoords='offset points', fontsize=12)
    plt.xlabel('Annualized Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    plt.title('Risk-Return Profile', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    portfolio_fig_dir = fig_dir / "portfolio"
    portfolio_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(portfolio_fig_dir / "risk_return_profile.png")
    plt.close()
    print("✅ Risk-Return profile plot created.")
    
def create_annual_return_barchart(portfolios: pd.DataFrame, fig_dir: Path):
    ann_ret = portfolios.mean() * 365 * 100
    plt.figure(figsize=(10, 7))
    ann_ret.sort_values(ascending=False).plot(kind='bar', color='c', alpha=0.7)
    plt.ylabel('Average Annual Return (%)')
    plt.xlabel('Strategy')
    plt.title('Average Annual Return (%)', fontsize=16)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    portfolio_fig_dir = fig_dir / "portfolio"
    portfolio_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(portfolio_fig_dir / "average_annual_return.png")
    plt.close()
    print("✅ Average Annual Return bar chart created.")
    
def create_portfolio_summary_plots(portfolios: pd.DataFrame, fig_dir: Path):
    portfolio_fig_dir = fig_dir / "portfolio"
    portfolio_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    cumlog = np.log1p(portfolios).cumsum()
    cumlog.plot(ax=plt.gca(), linewidth=2)
    plt.title("Cumulative Log Returns - All Strategies", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(portfolio_fig_dir / "cumulative_log_returns.png")
    plt.close()
    sharpe = portfolios.mean() / portfolios.std() * np.sqrt(365)
    plt.figure(figsize=(10, 6))
    sharpe.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.title('Annualized Sharpe Ratios - All Strategies', fontsize=16)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(portfolio_fig_dir / "sharpe_ratios.png")
    plt.close()
    print(f"✅ Core portfolio summary plots saved to '{portfolio_fig_dir.resolve()}'")

def calculate_statistical_validation(df_merged: pd.DataFrame, fig_dir: Path):
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION METRICS")
    print("="*60)
    
    if 'close' not in df_merged.columns:
        print("[ERROR] 'close' price column is missing. Cannot calculate multi-day returns.")
        return

    df_merged['actual_return_5d'] = df_merged.groupby('symbol')['close'].transform(
        lambda x: (x.shift(-5) / x - 1)
    )
    
    prediction_cols = [col for col in df_merged.columns if col.endswith('_pred')]
    results = []
    
    for pred_col in prediction_cols:
        eval_df = df_merged[['actual_return_5d', pred_col]].dropna()
        if eval_df.empty:
            print(f"Skipping {pred_col} due to no overlapping data.")
            continue
            
        y_true = eval_df['actual_return_5d']
        y_pred = eval_df[pred_col]
        
        r2 = r2_score(y_true, y_pred)
        correct_direction = np.sign(y_pred) == np.sign(y_true)
        correct_direction[y_true == 0] = True
        dir_accuracy = correct_direction.mean()
        
        model_name = pred_col.replace('_return_pred', '').upper()
        results.append({
            'Model': model_name, 
            'Out-of-Sample R2': f"{r2:.4f}", 
            'Directional Accuracy (%)': f"{dir_accuracy:.2%}"
        })
    
    if results:
        results_df = pd.DataFrame(results)
        results_str = results_df.to_string(index=False)
        print(results_str)
        output_path = fig_dir / "statistical_validation_metrics.txt"
        output_path.write_text(results_str)
        print(f"✅ Statistical validation metrics saved to '{output_path.resolve()}'")
    else:
        print("No prediction columns found to evaluate.")

# --- Execution Entry Point (unchanged) ---
if __name__ == "__main__":
    crypto_features_path = Path("./results/integrated_results/integrated_features.csv")
    ml_predictions_path = Path("./results/integrated_results/ml_predictions_integrated.csv")
    output_returns_path = Path("./results/integrated_results/integrated_portfolios.csv")
    figures_dir = Path("./results/figures")
    run_portfolio_construction(
        panel_path=crypto_features_path,
        ml_path=ml_predictions_path,
        out_csv=output_returns_path,
        fig_dir=figures_dir,
        n_qtiles=5
    )