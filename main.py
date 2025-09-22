import os
import time
import re
import io
import argparse
from datetime import datetime, time as dtime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import tweepy
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# Config (from env/GitHub secrets)
# -----------------------------------------------------------------------------
TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TZ_ET = ZoneInfo("America/New_York")

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

def parse_tickers_with_pct(text: str) -> List[Tuple[str, float]]:
    """
    Extract pairs of (ticker, pct_change) from text like:
      '$AGMH +230%, $CJET +150%'
    Handles optional '+' and decimals. Returns [] if none found.
    """
    norm = text.upper().replace("＋", "+").replace("−", "-").replace("％", "%")
    pattern = re.compile(r"\$([A-Z]{1,5})\s*([+\-]?\d+(?:\.\d+)?)\s*%")
    results = []
    for m in pattern.finditer(norm):
        tkr = m.group(1)
        try:
            pct = float(m.group(2))
        except Exception:
            continue
        results.append((tkr, pct))
    # Deduplicate keep first
    seen, out = set(), []
    for t, p in results:
        if t not in seen:
            seen.add(t)
            out.append((t, p))
    return out

def now_et() -> datetime:
    return datetime.now(tz=TZ_ET)

def within_window(start_hm: str, end_hm: str) -> bool:
    """
    Check if current ET time is within [start_hm, end_hm] inclusive.
    hm format 'HH:MM' 24h.
    """
    now = now_et()
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start_dt <= now <= end_dt

def sleep_until_window(start_hm: str, end_hm: str, max_wait_min: int = 180):
    """
    If now is before the window, sleep until start or max_wait_min reached.
    If now is after end, return immediately.
    """
    now = now_et()
    sh, sm = map(int, start_hm.split(":"))
    start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    if now < start_dt:
        wait = min(max_wait_min * 60, max(0, int((start_dt - now).total_seconds())))
        if wait > 0:
            print(f"[info] Waiting {wait//60}m until window {start_hm} ET...")
            time.sleep(wait)

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
            sorted_tweets = sorted(resp.data, key=lambda t: t.created_at, reverse=True)
            return sorted_tweets[0].text
        except tweepy.TooManyRequests as e:
            last_err = e
            time.sleep(60)
        except Exception as e:
            last_err = e
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch the premarket post: {last_err}")

# -----------------------------------------------------------------------------
# Finance helpers
# -----------------------------------------------------------------------------
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
        return max(0.0, min(1.0, vol / 0.10))
    except Exception:
        return 0.0

# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------
def rank_tickers(parsed: List[Tuple[str, float]]) -> pd.DataFrame:
    """
    Accepts a list of (Ticker, Premarket%) parsed from the source tweet.
    Enriches with Sentiment and Runner, then computes Composite score:
      Composite = Premarket% * 0.7 + (Sentiment * 100) * 0.15 + (Runner * 100) * 0.15
    """
    rows: List[Dict] = []
    for t, pm in parsed:
        try:
            sent = round(get_sentiment_score(t), 3)
            runner = round(get_runner_score(t), 3)
            comp = round(pm * 0.70 + (sent * 100.0) * 0.15 + (runner * 100.0) * 0.15, 3)
            rows.append({"Ticker": t, "Premarket%": round(pm, 2), "Sentiment": sent, "Runner": runner, "Composite": comp})
        except Exception:
            rows.append({"Ticker": t, "Premarket%": round(pm, 2), "Sentiment": 0.0, "Runner": 0.0, "Composite": round(pm * 0.70, 3)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Composite", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def _telegram_send_text(text: str) -> None:
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
        try:
            import requests
            requests.post(url, json=payload, timeout=20)
        except Exception as e:
            print(f"[warn] Telegram send failed: {e}. Falling back to stdout.")
            print(text)
    else:
        print("[info] TELEGRAM_TOKEN/CHAT_ID not set. Printing output instead:")
        print(text)

def send_to_telegram(df: pd.DataFrame, message: str) -> None:
    try:
        table_md = df.to_markdown(index=False) if not df.empty else "_(no data)_"
    except Exception:
        table_md = str(df)
    full_msg = f"{message}\n\n*Ranked Premarket Gainers*\n{table_md}"
    _telegram_send_text(full_msg)

def send_table_image(df: pd.DataFrame, caption: str, filename: str = "ranked.png") -> None:
    """
    Render df as a table PNG and send via Telegram sendPhoto.
    """
    # Basic matplotlib table render
    fig, ax = plt.subplots(figsize=(10, min(12, 1 + 0.5 * max(3, len(df)))))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': ('ranked.png', buf, 'image/png')}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        try:
            import requests
            requests.post(url, files=files, data=data, timeout=30)
        except Exception as e:
            print(f"[warn] Telegram image send failed: {e}. Falling back to text table.")
            send_to_telegram(df, caption)
    else:
        # Save locally so GH logs show a path
        with open(filename, "wb") as f:
            f.write(buf.getvalue())
        print(f"[info] Saved image to {filename}")
        send_to_telegram(df, caption)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_pipeline() -> Tuple[pd.DataFrame, str]:
    post_text = fetch_latest_post()
    parsed = parse_tickers_with_pct(post_text)
    if not parsed:
        tickers = extract_tickers(post_text)
        parsed = [(t, 0.0) for t in tickers]
    ranked_df = rank_tickers(parsed)
    timestamp = now_et().strftime("%Y-%m-%d %H:%M ET")
    message = f"Ace's Premarket List Ranked\n\nDate: {timestamp}\nTickers Found: {len(parsed)}"
    return ranked_df, message

def run_pipeline_with_poll(max_minutes: int = 0, interval_sec: int = 60, quick_first: bool = False,
                           window_start: Optional[str] = None, window_end: Optional[str] = None) -> None:
    """
    Poll for a tweet with % values. If quick_first is True, immediately send a raw tickers-only
    message first (if % not present yet), then keep polling until % appears or timeout.
    If a window is provided, sleep until the window opens and only poll within it.
    """
    if window_start and window_end:
        if not within_window(window_start, window_end):
            sleep_until_window(window_start, window_end)
        if not within_window(window_start, window_end):
            print("[info] Outside window; exiting without polling.")
            return

    # One attempt before polling to decide quick-first behavior
    try:
        post_text = fetch_latest_post()
        parsed = parse_tickers_with_pct(post_text)
        if parsed:
            df = rank_tickers(parsed)
            ts = now_et().strftime("%Y-%m-%d %H:%M ET")
            msg = f"Ace's Premarket List Ranked\n\nDate: {ts}\nTickers Found: {len(parsed)}"
            send_to_telegram(df, msg)
            send_table_image(df, "*Ranked Premarket Gainers*")
            print("Pipeline complete (found on first try).")
            return
        elif quick_first:
            # Send raw tickers immediately
            tickers = extract_tickers(post_text)
            ts = now_et().strftime("%Y-%m-%d %H:%M ET")
            raw_msg = f"Ace's Premarket — initial list (no % yet)\n\nDate: {ts}\nTickers: {', '.join(tickers) if tickers else 'None'}"
            _telegram_send_text(raw_msg)
    except Exception as e:
        print(f"[warn] Initial attempt failed: {e}")

    if max_minutes <= 0:
        # Final fallback one-shot
        df, msg = run_pipeline()
        send_to_telegram(df, msg)
        send_table_image(df, "*Ranked Premarket Gainers*")
        print("Pipeline complete (fallback one-shot).")
        return

    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        try:
            post_text = fetch_latest_post()
            parsed = parse_tickers_with_pct(post_text)
            if parsed:
                df = rank_tickers(parsed)
                ts = now_et().strftime("%Y-%m-%d %H:%M ET")
                msg = f"Ace's Premarket List Ranked\n\nDate: {ts}\nTickers Found: {len(parsed)}"
                send_to_telegram(df, msg)
                send_table_image(df, "*Ranked Premarket Gainers*")
                print("Pipeline complete (polled).")
                return
            else:
                print("No % values yet; polling again...")
        except Exception as e:
            print(f"Poll attempt error: {e}")
        time.sleep(interval_sec)

    # After polling window expires
    df, msg = run_pipeline()
    send_to_telegram(df, msg)
    send_table_image(df, "*Ranked Premarket Gainers*")
    print("Pipeline complete (timeout fallback).")

def _cli():
    parser = argparse.ArgumentParser(description="ACE Alerts Runner")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of Telegram")
    parser.add_argument("--poll-minutes", type=int, default=int(os.getenv("POLL_MINUTES", "0")),
                        help="If >0, poll for this many minutes until a post with % is found.")
    parser.add_argument("--poll-interval", type=int, default=int(os.getenv("POLL_INTERVAL_SEC", "60")),
                        help="Seconds between polls when --poll-minutes > 0.")
    parser.add_argument("--quick-first", action="store_true", help="Send raw tickers immediately, then follow up with ranked when % appears.")
    parser.add_argument("--window-start", type=str, default=os.getenv("WINDOW_START", ""),
                        help="ET start of polling window, e.g., 07:00")
    parser.add_argument("--window-end", type=str, default=os.getenv("WINDOW_END", ""),
                        help="ET end of polling window, e.g., 09:45")
    args = parser.parse_args()

    # Dry-run: temporarily unset Telegram env so send_to_telegram prints instead of sending
    if args.dry_run:
        os.environ.pop("TELEGRAM_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)

    window_start = args.window_start or None
    window_end = args.window_end or None

    run_pipeline_with_poll(
        max_minutes=args.poll_minutes,
        interval_sec=args.poll_interval,
        quick_first=args.quick_first,
        window_start=window_start,
        window_end=window_end,
    )

if __name__ == "__main__":
    try:
        _cli()
        print("Pipeline complete.")
    except Exception as e:
        err = f"Pipeline failed: {e}"
        try:
            send_to_telegram(pd.DataFrame(), err)
        finally:
            print(err)
