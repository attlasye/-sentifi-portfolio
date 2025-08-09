# news_pipeline.py (The absolute final, complete, and fixed version)

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Internal Helper Functions (All your original functions are here) ---

def _ensure_diag_dir(figures_base_dir: Path) -> Path:
    """Helper to create and return the causality diagnostics sub-directory."""
    d = figures_base_dir / "causality" / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    return d

def compute_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Adds first differences of sentiment indices."""
    out = df.copy()
    out["d_Overall"] = out["Overall"].diff()
    out["d_BTC"]     = out["BTC"].diff()
    return out.dropna()

def _plot_granger_stats(stats_df: pd.DataFrame, cause: str, effect: str, out_dir: Path) -> None:
    """Line plot of F-statistics across lags."""
    plt.figure(figsize=(8, 5))
    plt.plot(stats_df["lag"], stats_df["F_stat"], marker="o", linewidth=2, color="#2E86AB")
    plt.axhline(4.0, color="red", linestyle="--", alpha=0.7, label="approx 5% critical value")
    plt.title(f"Granger causality: {cause} -> {effect}")
    plt.xlabel("Lag")
    plt.ylabel("F-statistic")
    plt.grid(alpha=0.3)
    plt.legend()
    fname = f"granger_{cause}_to_{effect}.png"
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()

def run_granger_tests(df: pd.DataFrame, max_lag: int, figures_dir: Path) -> dict:
    """Granger-causality tests in both directions."""
    pairs = [
        ("d_Overall", "ret_btc"),
        ("d_BTC",     "ret_btc"),
        ("ret_btc",   "d_Overall"),
        ("ret_btc",   "d_BTC"),
    ]
    causality_plots_dir = figures_dir / "causality"
    causality_plots_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for cause, effect in pairs:
        logging.info("Running Granger test: %s -> %s", cause, effect)
        data = df[[effect, cause]].dropna()
        gtest = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        rows = [(lag, res[0]["ssr_ftest"][0], res[0]["ssr_ftest"][1]) for lag, res in gtest.items()]
        stats_df = pd.DataFrame(rows, columns=["lag", "F_stat", "p_value"])

        best = stats_df.loc[stats_df["p_value"].idxmin()]
        summary = (f"{cause} Granger-causes {effect} (p={best['p_value']:.4f}, lag={int(best['lag'])})"
                   if best["p_value"] < 0.05
                   else f"No strong evidence that {cause} Granger-causes {effect} (min p={best['p_value']:.4f})")
        
        results[f"{cause}->{effect}"] = {"table": stats_df, "summary": summary}
        _plot_granger_stats(stats_df, cause=cause, effect=effect, out_dir=causality_plots_dir)

    return results

def plot_ccf_delta_sent_ret(df: pd.DataFrame, figures_dir: Path, max_lag: int = 8) -> None:
    """Stem plot of the cross-correlation function."""
    diag_dir = _ensure_diag_dir(figures_dir)
    x, y = df["d_Overall"] - df["d_Overall"].mean(), df["ret_btc"] - df["ret_btc"].mean()
    lags = list(range(-max_lag, max_lag + 1))
    ccf  = [x.corr(y.shift(-k)) for k in lags]
    
    plt.figure(figsize=(10, 5))
    markerline, stemlines, baseline = plt.stem(lags, ccf, basefmt=" ")
    plt.setp(markerline, marker="o", markersize=6, color="#2E86AB")
    plt.setp(stemlines, linewidth=1.2, color="#2E86AB")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cross-Correlation: ΔOverall vs ret_btc")
    plt.xlabel("Lag (days; positive = ΔOverall leads)")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(diag_dir / "ccf_deltaOverall_ret_btc.png", dpi=150)
    plt.close()

def plot_corr_matrix_lags(df: pd.DataFrame, figures_dir: Path, max_lag: int = 4) -> None:
    """Heat-map: correlation of ret_btc with sentiment lags."""
    diag_dir = _ensure_diag_dir(figures_dir)
    rows, cols = ["d_Overall", "d_BTC"], [f"Lag_{k}" for k in range(1, max_lag + 1)]
    mat  = np.zeros((len(rows), max_lag))
    for r_idx, r in enumerate(rows):
        for k in range(1, max_lag + 1):
            mat[r_idx, k - 1] = df[r].shift(k).corr(df["ret_btc"])
            
    plt.figure(figsize=(8, 3))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", yticklabels=rows, xticklabels=cols, center=0.0, cbar_kws={"label": "Corr"})
    plt.title("ret_btc vs sentiment lags (positive lag = sentiment leads)")
    plt.tight_layout()
    plt.savefig(diag_dir / "corr_matrix_ret_btc_vs_sent_lags.png", dpi=150)
    plt.close()

def plot_dual_axis(df: pd.DataFrame, figures_dir: Path) -> None:
    """Dual-axis plot: z-scored sentiment vs returns."""
    diag_dir = _ensure_diag_dir(figures_dir)
    z_dOverall = ss.zscore(df["d_Overall"].values, nan_policy="omit")
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["date"], z_dOverall, color="#2E86AB", label="z-score ΔOverall")
    ax1.set_ylabel("z-score ΔOverall", color="#2E86AB")
    ax1.tick_params(axis="y", labelcolor="#2E86AB")
    
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["ret_btc"], color="#A23B72", label="ret_btc")
    ax2.set_ylabel("ret_btc", color="#A23B72")
    ax2.tick_params(axis="y", labelcolor="#A23B72")
    
    fig.suptitle("ΔOverall (standardised) vs BTC Daily Return")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "dual_axis_zDeltaOverall_ret_btc.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main function to be called by the orchestrator
# --------------------------------------------------------------------------
def run_granger_causality_analysis(
    integrated_results_dir: Path,
    portfolio_returns_path: Path,
    figures_dir: Path
):
    """
    Main function to run the Granger Causality analysis pipeline.
    This will be imported and called by main.py.
    """
    logging.info("="*20 + " EXTRA ANALYSIS: GRANGER CAUSALITY " + "="*20)

    sentiment_path = integrated_results_dir.parent / "news_data" / "compound_timeseries_daily.csv"
    returns_path = portfolio_returns_path

    if not sentiment_path.exists() or not returns_path.exists():
        logging.error(f"Cannot run Granger analysis. Missing input files: {sentiment_path} or {returns_path}")
        return

    df_sent = pd.read_csv(sentiment_path, parse_dates=["date"])[["date", "Overall", "BTC"]]

    # --- FIX for ValueError STARTS HERE ---
    # Load the returns CSV, telling pandas the first column is the index and should be parsed as dates
    df_ret = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    # Reset the index to turn the date index back into a 'date' column for merging
    df_ret = df_ret.reset_index().rename(columns={'index': 'date'})
    # --- FIX for ValueError ENDS HERE ---

    if "BTC" not in df_ret.columns:
        logging.error("BTC return column not found in portfolio returns file. Cannot run Granger analysis.")
        return

    df_ret = df_ret[["date", "BTC"]].rename(columns={"BTC": "ret_btc"})

    # Merge and prepare data
    df = pd.merge(df_sent, df_ret, on="date", how="inner").sort_values("date")
    df = compute_differences(df)

    if df.empty:
        logging.warning("Dataframe is empty after merging and differencing. Skipping Granger analysis.")
        return

    # Run tests and save summary
    logging.info("Running Granger causality tests...")
    results = run_granger_tests(df, max_lag=4, figures_dir=figures_dir)
    summary_path = figures_dir / "granger_causality_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Granger Causality Analysis Summary\n" + "="*40 + "\n")
        for key, value in results.items():
            f.write(f"Test: {key}\nInterpretation: {value['summary']}\n")
            f.write("--- Detailed p-values ---\n")
            f.write(value['table'].to_string(index=False) + "\n\n")

    # Run diagnostic plots
    logging.info("Generating diagnostic plots for causality analysis...")
    plot_ccf_delta_sent_ret(df, figures_dir=figures_dir)
    plot_corr_matrix_lags(df, figures_dir=figures_dir)
    plot_dual_axis(df, figures_dir=figures_dir)

    logging.info(f"Granger causality analysis complete. Plots are in '{figures_dir}/causality'. Summary saved to {summary_path}")