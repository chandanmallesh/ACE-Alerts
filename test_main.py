import os
from datetime import datetime
from typing import List, Dict

import pandas as pd

# -----------------------------------------------------------------------------
# Mock data and config
# -----------------------------------------------------------------------------
TEST_POST = "Premarket Top % Gainers: $AGMH +230%, $CJET +150%, $ATMV +68%, $ADAP +19%, $YDKG +4%"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# -----------------------------------------------------------------------------
# Helpers mirroring main.py behavior (but deterministic for CI)
# -----------------------------------------------------------------------------
def extract_tickers(text: str) -> List[str]:
    import re
    return re.findall(r"\$([A-Z]{1,5})", text.upper())

def get_premarket_change_pct(ticker: str) -> float:
    mock = {"AGMH": 230.0, "CJET": 150.0, "ATMV": 68.0, "ADAP": 19.0, "YDKG": 4.0}
    return float(mock.get(ticker, 0.0))

def get_sentiment_score(ticker: str) -> float:
    mock = {"AGMH": 0.85, "CJET": 0.80, "ATMV": 0.75, "ADAP": 0.70, "YDKG": 0.65}
    return float(mock.get(ticker, 0.0))

def get_runner_score(ticker: str) -> float:
    mock = {"AGMH": 0.90, "CJET": 0.88, "ATMV": 0.70, "ADAP": 0.55, "YDKG": 0.40}
    return float(mock.get(ticker, 0.0))

def rank_tickers(tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for t in sorted(set(tickers)):
        pm = round(get_premarket_change_pct(t), 2)
        sent = round(get_sentiment_score(t), 3)
        runner = round(get_runner_score(t), 3)
        comp = round(pm * 0.6 + sent * 20.0 * 0.2 + runner * 100.0 * 0.2, 3)
        rows.append({"Ticker": t, "Premarket%": pm, "Sentiment": sent, "Runner": runner, "Composite": comp})
    df = pd.DataFrame(rows)
    return df.sort_values("Composite", ascending=False).reset_index(drop=True)

def send_to_telegram(df: pd.DataFrame, message: str) -> None:
    try:
        table_md = df.to_markdown(index=False) if not df.empty else "_(no data)_"
    except Exception:
        table_md = str(df)

    full_msg = f"{message}\n\n*Ranked Premarket Gainers*\n{table_md}"
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": full_msg, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=20)
        except Exception as e:
            print(f"[warn] Telegram send failed: {e}. Printing instead.")
            print(full_msg)
    else:
        print("[info] TELEGRAM_TOKEN/CHAT_ID not set. Printing output instead:")
        print(full_msg)

if __name__ == "__main__":
    try:
        post_text = TEST_POST
        tickers = extract_tickers(post_text)
        if not tickers:
            raise ValueError("No tickers extracted from mock data.")

        ranked_df = rank_tickers(tickers)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        message = f"Test: Ace's Premarket List Ranked\nDate: {timestamp}\nTickers Found: {len(tickers)}"

        send_to_telegram(ranked_df, message)
        print("Test pipeline complete.")
    except Exception as e:
        error_msg = f"Test pipeline failed: {e}"
        send_to_telegram(pd.DataFrame(), error_msg)
        print(error_msg)
