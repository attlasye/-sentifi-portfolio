# sentiment_pipeline.py (Final Version with Subfolder Organization)

from __future__ import annotations
import logging, time
from datetime import timedelta, datetime
from pathlib import Path
from typing import Dict
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
import re
from vader_custom_lexicon import custom_words

# --- Logging and Directory setup (unchanged) ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
tqdm.pandas()
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
def build_sentiment_dirs(base_dir: str | Path = ".") -> Dict[str, Path]:
    root = Path(base_dir).resolve()
    results_root = root / "results"
    dirs = {"data_dir": results_root / "sentiment_analysis", "fig_dir": results_root / "figures", "news_raw_dir": results_root / "news_data"}
    for path in dirs.values(): path.mkdir(parents=True, exist_ok=True)
    return dirs

# --- Stage 1 (Data Collection with retries - unchanged from our previous fix) ---
def fetch_news_range(
    api_key: str | None,
    start_dt: datetime,
    end_dt: datetime,
    lang: str = "EN",
    max_retries: int = 5,
    retry_delay: int = 10
) -> pd.DataFrame:
    """
    Pull CoinDesk news between start_dt and end_dt (inclusive).
    Includes a retry mechanism and the fix for the 'ID' vs 'id' KeyError.
    """
    url = "https://data-api.coindesk.com/news/v1/article/list"
    out: list[pd.DataFrame] = []

    while end_dt > start_dt:
        query_ts = int(end_dt.timestamp())
        query_day = end_dt.strftime("%Y-%m-%d")
        
        resp = None
        for attempt in range(max_retries):
            try:
                logging.info("Requesting articles up to %s (UTC) - Attempt %d/%d", query_day, attempt + 1, max_retries)
                resp = requests.get(f"{url}?lang={lang}&to_ts={query_ts}", timeout=30)
                
                if resp.ok:
                    break
                if 500 <= resp.status_code < 600:
                    logging.warning("Server error (status %s). Retrying in %d seconds...", resp.status_code, retry_delay)
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error("Client error (status %s). Aborting.", resp.status_code)
                    break
            except requests.exceptions.RequestException as e:
                logging.error("A network error occurred: %s. Retrying in %d seconds...", e, retry_delay)
                time.sleep(retry_delay)

        if not resp or not resp.ok:
            logging.error("Failed to fetch data after %d retries for date %s. Stopping collection.", max_retries, query_day)
            break

        data = resp.json()
        if "Data" not in data or not data["Data"]:
            logging.warning("No 'Data' field or empty data returned for %s. Stopping loop.", query_day)
            break
            
        d = pd.DataFrame(data["Data"])
        # Important check: Make sure the required 'ID' and 'PUBLISHED_ON' columns exist
        if 'ID' not in d.columns or 'PUBLISHED_ON' not in d.columns:
            logging.warning("Response missing 'ID' or 'PUBLISHED_ON' column. Skipping this batch.")
            # Step backward by one day to avoid getting stuck in a loop
            end_dt -= timedelta(days=1)
            continue
            
        d["date"] = pd.to_datetime(d["PUBLISHED_ON"], unit="s")
        
        new_articles = d[d["date"] >= start_dt]
        if new_articles.empty and not d.empty:
            break
        
        out.append(new_articles)
        end_dt = datetime.utcfromtimestamp(d["PUBLISHED_ON"].min() - 1)

    if not out:
        logging.warning("No articles were collected in the given date range.")
        return pd.DataFrame()

    # --- FIX: Changed subset from ['id'] to ['ID'] to match the API's raw column name ---
    news = pd.concat(out, ignore_index=True).drop_duplicates(subset=['ID']).sort_values('date', ascending=False)
    
    logging.info("Fetched %d unique articles in total", len(news))
    return news

# --- Stage 1 & 2 main functions (unchanged) ---
def stage1_collect_news(api_key: str | None, start_dt: datetime, end_dt: datetime, data_dir: Path, filename: str = "stage_1_news_raw.csv") -> pd.DataFrame:
    # This function remains the same
    tic = time.time()
    logging.info("Stage 1 - downloading news from CoinDesk API...")
    df = fetch_news_range(api_key, start_dt, end_dt)
    if df.empty:
        logging.warning("No news data was collected.")
        return df
    drop_cols = ["GUID", "PUBLISHED_ON_NS", "IMAGE_URL", "SUBTITLE", "AUTHORS", "URL", "UPVOTES", "DOWNVOTES", "SCORE", "CREATED_ON", "UPDATED_ON", "SOURCE_DATA", "CATEGORY_DATA", "STATUS", "SOURCE_ID", "TYPE"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.columns = df.columns.str.lower()
    essential_cols = ["date", "id", "title", "body", "tags", "keywords"]
    other_cols = [c for c in df.columns if c not in essential_cols + ["date", "id"]]
    df = df[["date", "id"] + [c for c in essential_cols[2:] if c in df.columns] + other_cols]
    if "sentiment" in df.columns: df["positive"] = np.where(df["sentiment"].str.upper() == "POSITIVE", 1, 0); df = df.drop(columns="sentiment")
    else: df["positive"] = np.nan
    out_path = data_dir / filename
    df.to_csv(out_path, index=False)
    logging.info("Saved raw news -> %s (%d articles in %.2f s)", out_path.name, len(df), time.time() - tic)
    return df

def stage2_add_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    # This function remains the same
    neutral_stopwords = {"the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "being", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "you", "your", "yours", "i", "me", "my", "mine", "in", "on", "at", "by", "with", "of", "for", "to", "from", "this", "that", "these", "those", "and", "or", "if", "then", "when", "while", "where", "which", "who", "whom", "whose", "what", "how", "because", "since", "as", "so"}
    def _remove_neutral_words(text: str) -> str:
        tokens = text.split(); cleaned = []
        for tok in tokens:
            core = re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", tok)
            if core.lower() not in neutral_stopwords: cleaned.append(tok)
        return " ".join(cleaned)
    crypto_map = {"BTC":  ["BTC", "BITCOIN", "Bitcoin"], "ETH":  ["ETH", "ETHEREUM", "Ethereum"], "XRP":  ["XRP", "RIPPLE", "Ripple"], "USDT": ["USDT", "TETHER", "Tether"], "BNB":  ["BNB", "BINANCE COIN", "Binance Coin"], "SOL":  ["SOL", "SOLANA", "Solana"], "USDC": ["USDC", "USD COIN", "USD Coin"], "ADA":  ["ADA", "CARDANO", "Cardano"], "DOGE": ["DOGE", "DOGECOIN", "Dogecoin"], "TON":  ["TON", "TONCOIN", "Toncoin"]}
    meme_terms = {"MEME", "SHIBA", "SHIB", "PEPE", "FLOKI", "MOG", "BONK", "DOGE", "DOGECOIN","Meme"}
    def _tag_keywords(kw: str) -> str | None:
        if pd.isna(kw): return None
        kw_up = kw.upper(); hits = set()
        for tag, terms in crypto_map.items():
            if any(t in kw_up for t in terms): hits.add(tag)
        if any(m in kw_up for m in meme_terms): hits.add("Meme")
        if len(hits) == 0: return None
        if len(hits) == 1: return hits.pop()
        return "Mixed"
    df = df_raw.copy()
    df = df.rename(columns={"date": "time"}); df["time"] = pd.to_datetime(df["time"]); df.insert(1, "date", df["time"].dt.date)
    df["all_text"] = (df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).str.strip()
    df["all_text"] = df["all_text"].apply(_remove_neutral_words)
    df["n_words"] = df["all_text"].astype(str).str.split().str.len()
    df["crypto_tag"] = df["keywords"].astype(str).apply(_tag_keywords)
    cols = list(df.columns)
    cols.insert(cols.index("body") + 1, cols.pop(cols.index("all_text")))
    cols.insert(cols.index("keywords") + 1, cols.pop(cols.index("crypto_tag")))
    return df[cols]

# --- Helper functions for VADER and resampling (unchanged) ---
_VADER = SentimentIntensityAnalyzer(); _VADER.lexicon.update(custom_words)
def _vader_scores(txt: str) -> pd.Series: return pd.Series(_VADER.polarity_scores(txt))
def _add_word_filters(df: pd.DataFrame, pctl: float = 25.0) -> pd.DataFrame:
    out = df.copy(); out["n_words"] = out["all_text"].astype(str).str.split().str.len()
    thresh = out.groupby("date")["n_words"].transform(lambda x: np.percentile(x, pctl))
    out["below_pctl"] = out["n_words"] < thresh
    return out
def _rescale(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out["compound_pct"] = (out["compound"] + 1.0) * 50.0
    return out
def _wmean(g: pd.DataFrame, weight_col: str | None) -> float:
    if weight_col and weight_col in g.columns:
        w = g[weight_col].to_numpy()
        return np.average(g["compound_pct"].to_numpy(), weights=w) if w.sum() else np.nan
    return g["compound_pct"].mean()
def _export_timeseries(df: pd.DataFrame, out_dir: Path, *, weight_col: str | None = None, rule: str | None = None) -> None:
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    tag_daily = (df.groupby(["date", "crypto_tag"]).apply(lambda g: _wmean(g, weight_col)).reset_index(name="compound_pct").pivot(index="date", columns="crypto_tag", values="compound_pct"))
    overall = (df.groupby("date").apply(lambda g: _wmean(g, weight_col)).rename("Overall").to_frame())
    daily_tbl = pd.concat([overall, tag_daily], axis=1).sort_index()
    daily_tbl.to_csv(out_dir / "compound_timeseries_daily.csv", float_format="%.4f")
    if rule:
        resampled = daily_tbl.resample(rule).mean()
        resampled.to_csv(out_dir / f"compound_timeseries_{rule}.csv", float_format="%.4f")

# ===================================================================
# --- FIX: All plotting functions now save to the 'sentiment' subfolder ---
# ===================================================================
def _get_sentiment_fig_dir(fig_dir_base: Path) -> Path:
    """Helper to create and return the sentiment figures sub-directory."""
    sentiment_dir = fig_dir_base / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    return sentiment_dir

def _hist_grid(df: pd.DataFrame, fig_dir: Path, title: str, suffix: str = "") -> None:
    sentiment_fig_dir = _get_sentiment_fig_dir(fig_dir)
    fname = sentiment_fig_dir / f"hist_{suffix}.png"
    # ... (plotting code is the same)
    cols = ["compound", "pos", "neg", "neu"]; labs = ["Compound", "Positive", "Negative", "Neutral"]; clrs = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10)); fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    for i, (c, l, col) in enumerate(zip(cols, labs, clrs)):
        r, cc = divmod(i, 2); ax[r, cc].hist(df[c], bins=50, color=col, alpha=0.7, edgecolor="white"); ax[r, cc].set_title(l); ax[r, cc].grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def _daily_line(df: pd.DataFrame, fig_dir: Path, weight_col: str | None = None, suffix: str = "") -> None:
    sentiment_fig_dir = _get_sentiment_fig_dir(fig_dir)
    fname = sentiment_fig_dir / f"daily_avg_sentiment_{suffix}.png"
    # ... (plotting code is the same)
    daily = (df.groupby("date").apply(lambda g: _wmean(g, weight_col)).reset_index(name="compound_pct"))
    daily["date"] = pd.to_datetime(daily["date"]).sort_values()
    plt.figure(figsize=(15, 8)); plt.plot(daily["date"], daily["compound_pct"], linewidth=1.5, color="#2E86AB"); plt.fill_between(daily["date"], daily["compound_pct"], color="#2E86AB", alpha=0.3)
    plt.axhline(50, color="red", linestyle="--", alpha=0.7, label="Neutral (50)"); plt.title("Daily Weighted Average Sentiment (0–100)"); plt.xlabel("Date"); plt.ylabel("Weighted Avg Compound (%)")
    plt.grid(alpha=0.3, linewidth=0.5); plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()

def _heatmap(df: pd.DataFrame, fig_dir: Path, suffix: str = "") -> None:
    sentiment_fig_dir = _get_sentiment_fig_dir(fig_dir)
    fname = sentiment_fig_dir / f"monthly_avg_heatmap_{suffix}.png"
    # ... (plotting code is the same)
    daily = df.groupby("date")["compound_pct"].mean().reset_index(); daily["date"] = pd.to_datetime(daily["date"]); daily["year"] = daily["date"].dt.year; daily["month"] = daily["date"].dt.month
    pivot = (daily.groupby(["year", "month"])["compound_pct"].mean().unstack(fill_value=np.nan).sort_index(ascending=False))
    plt.figure(figsize=(12, 8)); sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlBu_r", center=50, cbar_kws={"label": "Avg Sentiment (%)"}, linewidths=0.5, linecolor="white")
    plt.title("Monthly Average Sentiment Heat-map"); plt.xlabel("Month"); plt.ylabel("Year"); plt.xticks(ticks=np.arange(12) + 0.5, labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def _fear_greed_gauge(df: pd.DataFrame, fig_dir: Path, suffix: str = "") -> None:
    sentiment_fig_dir = _get_sentiment_fig_dir(fig_dir)
    fname = sentiment_fig_dir / f"fear_greed_gauge_{suffix}.png"
    # ... (plotting code is the same)
    recent = df[pd.to_datetime(df["date"]) >= pd.to_datetime(df["date"]).max() - timedelta(days=6)]; avg = recent["compound_pct"].mean()
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection="polar")); colors = ["#8B0000", "#FF4500", "#FFD700", "#90EE90", "#006400"]; bounds = [0, 20, 40, 60, 80, 100]
    for i in range(5): t0, t1 = np.pi * (bounds[i] / 100), np.pi * (bounds[i + 1] / 100); ax.fill_between(np.linspace(t0, t1, 20), 0.5, 1, color=colors[i], alpha=0.8)
    for sc in [0, 25, 50, 75, 100]: ang = np.pi * (sc / 100); ax.plot([ang, ang], [0.5, 0.55], "k-", lw=1); ax.text(ang, 0.6, f"{sc}", ha="center", va="center", fontsize=10)
    needle = np.pi * (avg / 100); ax.plot([needle, needle], [0, 0.9], "k-", lw=8); ax.plot(needle, 0, "ko", ms=15)
    cat, col = _cat(avg); ax.text(np.pi / 2, 0.2, f"{avg:.0f}", ha="center", va="center", fontsize=60, weight="bold")
    plt.figtext(0.5, 0.15, "Last 7-day Average", ha="center", fontsize=13); plt.figtext(0.5, 0.10, f"Current Status: {cat}", ha="center", fontsize=15, weight="bold", color=col)
    ax.set_ylim(0, 1.3); ax.set_xlim(0, np.pi); ax.set_theta_zero_location("W"); ax.set_theta_direction(1); ax.grid(False); ax.set_rticks([]); ax.set_thetagrids([])
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def _cat(v: float) -> tuple[str, str]:
    if v < 20: return "Extreme Fear", "#8B0000"
    if v < 40: return "Fear", "#FF4500"
    if v < 60: return "Neutral", "#FFD700"
    if v < 80: return "Greed", "#90EE90"
    return "Extreme Greed", "#006400"

# --- Main Stage 3 function (modified to pass correct paths) ---
def stage3_sentiment_and_plots(df_clean: pd.DataFrame, dirs: Dict[str, Path], *, thr: float = 0.05, word_pctl: float = 25.0, weight_col: str | None = "n_words", resample_rule: str | None = "D") -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 3 – VADER, filters, plots")
    df = df_clean.copy()
    df[["neg", "neu", "pos", "compound"]] = df["all_text"].progress_apply(_vader_scores)
    df_thr = df.loc[df["compound"].abs() >= thr].copy()
    df_final = (_add_word_filters(df_thr, word_pctl).loc[lambda d: ~d["below_pctl"]].copy())
    df_final["sentiment"] = np.where(df_final["compound"] >= thr, "positive", "negative")
    df_final = _rescale(df_final)

    # Plots - overall
    _hist_grid(df, dirs["fig_dir"], "Sentiment – raw sample", "raw")
    _hist_grid(df_thr, dirs["fig_dir"], f"Sentiment – |compound| ≥ {thr}", "thr")
    _hist_grid(df_final, dirs["fig_dir"], "Sentiment – final sample", "final")
    _daily_line(df_final, dirs["fig_dir"], weight_col)
    _heatmap(df_final, dirs["fig_dir"])
    _fear_greed_gauge(df_final, dirs["fig_dir"])

    # Plots - per crypto_tag
    tags = sorted([t for t in df_final["crypto_tag"].dropna().unique()])
    for tag in tags:
        m_final = df_final["crypto_tag"] == tag
        if not m_final.any(): continue
        suff = re.sub(r"[^A-Za-z0-9]+", "_", tag)
        _hist_grid(df_final.loc[m_final], dirs["fig_dir"], f"Sentiment – final ({tag})", f"final_{suff}")
        _daily_line(df_final.loc[m_final], dirs["fig_dir"], weight_col, suff)
        _heatmap(df_final.loc[m_final], dirs["fig_dir"], suff)
        _fear_greed_gauge(df_final.loc[m_final], dirs["fig_dir"], suff)

    _export_timeseries(df_final, dirs["news_data_dir"], weight_col=weight_col, rule=resample_rule)
    out_csv = dirs["news_data_dir"] / "clean_news_timeseries.csv"
    df_final.to_csv(out_csv, index=False)
    logging.info("Wrote final time-series -> %s  (%.2f s)", out_csv.name, time.time() - tic)
    return df_final