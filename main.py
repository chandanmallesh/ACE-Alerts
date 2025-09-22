import os
import io
import re
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import tweepy
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TZ_ET = ZoneInfo("America/New_York")

twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER) if TWITTER_BEARER else None
analyzer = SentimentIntensityAnalyzer()

# -----------------------------------------------------------------------------
# Helpers: time/window
# -----------------------------------------------------------------------------
def now_et() -> datetime:
    return datetime.now(tz=TZ_ET)

def within_window(start_hm: Optional[str], end_hm: Optional[str]) -> bool:
    if not start_hm or not end_hm:
        return True
    now = now_et()
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start_dt <= now <= end_dt

def sleep_until_window(start_hm: str):
    now = now_et()
    sh, sm = map(int, start_hm.split(":"))
    start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    if now < start_dt:
        secs = int((start_dt - now).total_seconds())
        print(f"[info] Sleeping {secs}s until window {start_hm} ET...")
        time.sleep(secs)

# -----------------------------------------------------------------------------
# Helpers: parsing
# -----------------------------------------------------------------------------
def extract_tickers(text: str) -> List[str]:
    return re.findall(r"\$([A-Z]{1,5})", text.upper())

def parse_tickers_with_pct(text: str) -> List[Tuple[str, float]]:
    norm = text.upper().replace("＋", "+").replace("−", "-").replace("％", "%")
    pattern = re.compile(r"\$([A-Z]{1,5})\s*([+\-]?\d+(?:\.\d+)?)\s*%")
    seen, pairs = set(), []
    for m in pattern.finditer(norm):
        t = m.group(1)
        try:
            p = float(m.group(2))
        except Exception:
            continue
        if t not in seen:
            seen.add(t)
            pairs.append((t, p))
    return pairs

# -----------------------------------------------------------------------------
# Rate-limit handling
# -----------------------------------------------------------------------------
def _sleep_until_reset(e, default_sleep: int = 60):
    """Sleep until Twitter rate-limit reset if available."""
    try:
        reset = None
        resp = getattr(e, "response", None)
        if resp is not None and hasattr(resp, "headers"):
            hdr = resp.headers.get("x-rate-limit-reset")
            if hdr:
                reset = int(hdr)
        now = int(time.time())
        if reset and reset > now:
            sleep_for = min(int(os.getenv("RATE_LIMIT_MAX_SLEEP", "600")), reset - now + 5)
        else:
            sleep_for = int(os.getenv("RATE_LIMIT_DEFAULT_SLEEP", str(default_sleep)))
        print(f"[rate-limit] Sleeping {sleep_for}s until reset...")
        time.sleep(sleep_for)
    except Exception:
        time.sleep(default_sleep)

# -----------------------------------------------------------------------------
# Twitter fetch
# -----------------------------------------------------------------------------
def _resolve_user_id(username: str) -> str:
    if not twitter_client:
        raise RuntimeError("TWITTER_BEARER_TOKEN not configured")
    last_err = None
    for _ in range(3):
        try:
            u = twitter_client.get_user(username=username, user_fields=["id"])
            if not u.data:
                raise ValueError(f"User not found: {username}")
            return u.data.id
        except tweepy.TooManyRequests as e:
            last_err = e
            _sleep_until_reset(e, default_sleep=60)
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Failed to resolve user id: {last_err}")

def _fetch_matching_tweets(user_id: str, since_id: Optional[str] = None):
    """Return (matches, newest_id). Each match is (id, text, created_at)."""
    params = dict(id=user_id, max_results=20, tweet_fields=["created_at","text"])
    if since_id:
        params["since_id"] = since_id
    last_err = None
    for _ in range(5):
        try:
            resp = twitter_client.get_users_tweets(**params)
            data = resp.data or []
            matches = []
            for t in data:
                txt = getattr(t, "text", "") or ""
                if "PREMARKET TOP % GAINERS" in txt.upper():
                    matches.append((t.id, txt, getattr(t, "created_at", None)))
            newest_id = None
            try:
                meta = getattr(resp, "meta", {}) or {}
                newest_id = meta.get("newest_id")
            except Exception:
                pass
            return matches, newest_id
        except tweepy.TooManyRequests as e:
            last_err = e
            _sleep_until_reset(e, default_sleep=60)
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch timeline: {last_err}")

def fetch_latest_post(username: str = "AceOfWallSt") -> str:
    """One-shot fetch using timeline first, search as fallback."""
    try:
        user_id = _resolve_user_id(username)
        matches, _ = _fetch_matching_tweets(user_id, since_id=None)
        if matches:
            # Most recent by created_at
            matches.sort(key=lambda x: x[2] or now_et(), reverse=True)
            return matches[0][1]
    except Exception as e:
        print(f"[warn] timeline fetch failed: {e}")
    # Fallback to search
    query = f'from:{username} "Premarket Top % Gainers"'
    last_err = None
    for _ in range(5):
        try:
            s = twitter_client.search_recent_tweets(query=query, max_results=10, tweet_fields=["created_at"])
            if s.data:
                srt = sorted(s.data, key=lambda t: t.created_at, reverse=True)
                return srt[0].text
            else:
                raise ValueError("No premarket post found.")
        except tweepy.TooManyRequests as e:
            last_err = e
            _sleep_until_reset(e, default_sleep=90)
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch the premarket post: {last_err}")

# -----------------------------------------------------------------------------
# Enrichment & ranking
# -----------------------------------------------------------------------------
def get_sentiment_score(ticker: str) -> float:
    try:
        tk = yf.Ticker(ticker)
        news = getattr(tk, "news", None) or []
        if not news:
            return 0.0
        titles = [n.get("title", "") for n in news[:5] if n.get("title")]
        if not titles:
            return 0.0
        scores = [analyzer.polarity_scores(t)["compound"] for t in titles]
        return float(sum(scores) / len(scores)) if scores else 0.0
    except Exception:
        return 0.0

def get_runner_score(ticker: str) -> float:
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

def rank_tickers(parsed: List[Tuple[str, float]]) -> pd.DataFrame:
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
# Telegram output
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
        with open(filename, "wb") as f:
            f.write(buf.getvalue())
        print(f"[info] Saved image to {filename}")
        send_to_telegram(df, caption)

# -----------------------------------------------------------------------------
# Pipelines
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
    if window_start and window_end and not within_window(window_start, window_end):
        sleep_until_window(window_start)
    if max_minutes <= 0:
        df, msg = run_pipeline()
        send_to_telegram(df, msg)
        send_table_image(df, "*Ranked Premarket Gainers*")
        return
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        try:
            text = fetch_latest_post()
            parsed = parse_tickers_with_pct(text)
            if parsed:
                df = rank_tickers(parsed)
                ts = now_et().strftime("%Y-%m-%d %H:%M ET")
                msg = f"Ace's Premarket List Ranked\n\nDate: {ts}\nTickers Found: {len(parsed)}"
                send_to_telegram(df, msg)
                send_table_image(df, "*Ranked Premarket Gainers*")
                return
            elif quick_first:
                tickers = extract_tickers(text)
                ts = now_et().strftime("%Y-%m-%d %H:%M ET")
                raw_msg = f"Ace's Premarket — initial list (no % yet)\n\nDate: {ts}\nTickers: {', '.join(tickers) if tickers else 'None'}"
                _telegram_send_text(raw_msg)
        except tweepy.TooManyRequests as e:
            _sleep_until_reset(e, default_sleep=90)
        except Exception as e:
            print(f"[warn] poll loop error: {e}")
        time.sleep(interval_sec)
    df, msg = run_pipeline()
    send_to_telegram(df, msg)
    send_table_image(df, "*Ranked Premarket Gainers*")

def watch_and_send(minutes: int, interval_sec: int = 900, username: str = "AceOfWallSt",
                   quick_first: bool = False, window_start: Optional[str] = None, window_end: Optional[str] = None):
    if minutes <= 0:
        df, msg = run_pipeline()
        send_to_telegram(df, msg)
        send_table_image(df, "*Ranked Premarket Gainers*")
        return

    if window_start and window_end and not within_window(window_start, window_end):
        sleep_until_window(window_start)
        if not within_window(window_start, window_end):
            print("[info] Outside window; exiting without watching.")
            return

    user_id = _resolve_user_id(username)
    since_id = None
    seen_ids = set()

    end_at = time.time() + minutes * 60
    while time.time() < end_at:
        try:
            matches, newest = _fetch_matching_tweets(user_id, since_id=since_id)
            if newest:
                since_id = newest
            if matches:
                for tid, text, created in sorted(matches, key=lambda x: x[2] or now_et()):
                    if tid in seen_ids:
                        continue
                    parsed = parse_tickers_with_pct(text)
                    ts = now_et().strftime("%Y-%m-%d %H:%M ET")
                    if parsed:
                        df = rank_tickers(parsed)
                        msg = f"Ace's Premarket List Ranked\n\nDate: {ts}\nTickers Found: {len(parsed)}"
                        send_to_telegram(df, msg)
                        send_table_image(df, "*Ranked Premarket Gainers*")
                    elif quick_first:
                        tickers = extract_tickers(text)
                        raw_msg = f"Ace's Premarket — initial list (no % yet)\n\nDate: {ts}\nTickers: {', '.join(tickers) if tickers else 'None'}"
                        _telegram_send_text(raw_msg)
                    seen_ids.add(tid)
            else:
                print("[info] No matching tweets in this interval.")
        except tweepy.TooManyRequests as e:
            _sleep_until_reset(e, default_sleep=90)
        except Exception as e:
            print(f"[warn] watch loop error: {e}")
        print(f"[info] Sleeping {interval_sec}s before next check...")
        time.sleep(max(5, int(interval_sec)))

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="ACE Alerts Runner")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of Telegram")
    parser.add_argument("--poll-minutes", type=int, default=int(os.getenv("POLL_MINUTES", "0")),
                        help="If >0, poll for this many minutes until a post with % is found.")
    parser.add_argument("--poll-interval", type=int, default=int(os.getenv("POLL_INTERVAL_SEC", "60")),
                        help="Seconds between polls when --poll-minutes > 0.")
    parser.add_argument("--watch-minutes", type=int, default=int(os.getenv("WATCH_MINUTES", "0")),
                        help="Continuously watch for this many minutes (default 0 = off).")
    parser.add_argument("--interval-sec", type=int, default=int(os.getenv("WATCH_INTERVAL_SEC", "900")),
                        help="Polling interval seconds when using --watch-minutes (default 900=15 min).")
    parser.add_argument("--quick-first", action="store_true",
                        help="Send raw tickers immediately, then follow up with ranked when % appears.")
    parser.add_argument("--window-start", type=str, default=os.getenv("WINDOW_START", ""),
                        help="ET start of polling window, e.g., 07:00")
    parser.add_argument("--window-end", type=str, default=os.getenv("WINDOW_END", ""),
                        help="ET end of polling window, e.g., 12:00")
    args = parser.parse_args()

    # Dry-run: clear Telegram env so we print instead of sending
    if args.dry_run:
        os.environ.pop("TELEGRAM_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)

    window_start = args.window_start or None
    window_end = args.window_end or None

    if args.watch_minutes > 0:
        watch_and_send(args.watch_minutes, interval_sec=args.interval_sec, quick_first=args.quick_first,
                       window_start=window_start, window_end=window_end)
    else:
        run_pipeline_with_poll(args.poll_minutes, interval_sec=args.poll_interval, quick_first=args.quick_first,
                               window_start=window_start, window_end=window_end)

if __name__ == "__main__":
    try:
        _cli()
        print("Pipeline complete.")
    except Exception as e:
        err = f"Pipeline failed: {e}"
        if "429" in str(e) or "Too Many Requests" in str(e):
            print(err)
        else:
            try:
                send_to_telegram(pd.DataFrame(), err)
            finally:
                print(err)
