import os
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import tweepy
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------------------------------------------------------
# Config (from env/GitHub secrets)
# -----------------------------------------------------------------------------
TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# -----------------------------------------------------------------------------
# Clients
# -----------------------------------------------------------------------------
twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER) if TWITTER_BEARER else None
analyzer = SentimentIntensityAnalyzer()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def extract_tickers(text: str) -> List[str]:
    """Extract tickers noted like $TSLA $AAPL (1-5 uppercase letters)."""
    return re.findall(r"\$([A-Z]{1,5})", text.upper())

# -----------------------------------------------------------------------------
# Twitter
# -----------------------------------------------------------------------------
def fetch_latest_post(username: str = "AceOfWallSt") -> str:
    """
    Fetch latest tweet from @AceOfWallSt containing 'Premarket Top % Gainers'.
    Retries to handle transient API rate limits.
    """
    if not twitter_client:
        raise RuntimeError("TWITTER_BEARER_TOKEN not configured")
    query = f'from:{username} "Premarket Top % Gainers"'
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = twitter_client.search_recent_tweets(
                query=query, max_results=10, tweet_fields=["created_at"]
            )
            if not resp.data:
                raise ValueError("No premarket post found.")
            # Choose the most recent tweet text
            sorted_tweets = sorted(resp.data, key=lambda t: t.created_at, reverse=True)
            return sorted_tweets[0].text
        except tweepy.TooManyRequests as e:
            last_err = e
            # wait 60s then retry
            time.sleep(60)
        except Exception as e:
            last_err = e
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch the premarket post: {last_err}")

# -----------------------------------------------------------------------------
# Finance helpers
# -----------------------------------------------------------------------------
def get_premarket_change_pct(ticker: str) -> float:
    """
    Approximate premarket % change using current/last price vs previous close.
    Falls back gracefully to 0.0 if data is unavailable.
    """
    try:
        tk = yf.Ticker(ticker)
        info = getattr(tk, "fast_info", None)
        if info and "last_price" in info and "previous_close" in info and info["previous_close"]:
            last_price = float(info["last_price"])
            prev_close = float(info["previous_close"])
            return ((last_price - prev_close) / prev_close) * 100.0
        # Fallback using recent 1d history
        hist = tk.history(period="2d", interval="1m")
        if not hist.empty:
            last = float(hist["Close"].dropna().iloc[-1])
            # previous close from 1d daily
            daily = tk.history(period="5d", interval="1d").dropna()
            if not daily.empty:
                prev_close = float(daily["Close"].iloc[-2] if len(daily) >= 2 else daily["Close"].iloc[-1])
                if prev_close:
                    return ((last - prev_close) / prev_close) * 100.0
        return 0.0
    except Exception:
        return 0.0

def get_sentiment_score(ticker: str) -> float:
    """
    Simple sentiment: average VADER compound on latest headlines if available.
    Range [-1, 1]. Returns 0.0 on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        news = getattr(tk, "news", None) or []
        if not news:
            return 0.0
        texts = [n.get("title", "") for n in news[:5] if n.get("title")]
        if not texts:
            return 0.0
        scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
        return float(sum(scores) / len(scores)) if scores else 0.0
    except Exception:
        return 0.0

def get_runner_score(ticker: str) -> float:
    """
    Crude 'runner' score: scale 1-month daily volatility into [0, 1].
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1mo", interval="1d").dropna()
        if len(hist) < 5:
            return 0.0
        returns = hist["Close"].pct_change().dropna().abs()
        vol = float(returns.mean())
        # Heuristic scaling: 0% -> 0, 10% -> ~1
        return max(0.0, min(1.0, vol / 0.10))
    except Exception:
        return 0.0

# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------
def rank_tickers(tickers: List[str]) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      Ticker | Premarket% | Sentiment | Runner | Composite
    and sorted by Composite desc.
    """
    rows: List[Dict] = []
    for t in sorted(set(tickers)):
        try:
            pm = round(get_premarket_change_pct(t), 2)
            sent = round(get_sentiment_score(t), 3)
            runner = round(get_runner_score(t), 3)
            comp = round(pm * 0.6 + sent * 20.0 * 0.2 + runner * 100.0 * 0.2, 3)
            rows.append(
                {
                    "Ticker": t,
                    "Premarket%": pm,
                    "Sentiment": sent,
                    "Runner": runner,
                    "Composite": comp,
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Composite", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def send_to_telegram(df: pd.DataFrame, message: str) -> None:
    """
    Send ranked table as Markdown to Telegram. If secrets are missing,
    print the message to stdout instead.
    """
    try:
        table_md = df.to_markdown(index=False) if not df.empty else "_(no data)_"
    except Exception:
        table_md = str(df)

    full_msg = f"{message}\n\n*Ranked Premarket Gainers*\n{table_md}"
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": full_msg, "parse_mode": "Markdown"}
        try:
            import requests  # local import to keep import surface minimal
            requests.post(url, json=payload, timeout=20)
        except Exception as e:
            print(f"[warn] Telegram send failed: {e}. Falling back to stdout.")
            print(full_msg)
    else:
        print("[info] TELEGRAM_TOKEN/CHAT_ID not set. Printing output instead:")
        print(full_msg)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_pipeline() -> None:
    post_text = fetch_latest_post()
    tickers = extract_tickers(post_text)
    if not tickers:
        raise ValueError("No tickers extracted.")
    ranked_df = rank_tickers(tickers)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    message = f"Ace's Premarket List Ranked\n\nDate: {timestamp}\nTickers Found: {len(tickers)}"
    send_to_telegram(ranked_df, message)

if __name__ == "__main__":
    try:
        run_pipeline()
        print("Pipeline complete.")
    except Exception as e:
        err = f"Pipeline failed: {e}"
        try:
            send_to_telegram(pd.DataFrame(), err)
        finally:
            print(err)
