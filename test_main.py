import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import requests
from datetime import datetime
import os
from typing import List

# Mock data and config (use same secrets as main.py)
TEST_POST = "Premarket Top % Gainers: $AGMH +230%, $CJET +150%, $ATMV +68%, $ADAP +19%, $YDKG +4%"
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Reuse functions from main.py (simplified for test)
def extract_tickers(text: str) -> List[str]:
    """Extract $TICKERs via regex."""
    import re
    return re.findall(r'\$([A-Z]{1,5})', text.upper())

def get_premarket_change(ticker: str) -> float:
    """Get estimated premarket % change via mock data."""
    test_changes = {'AGMH': 230.0, 'CJET': 150.0, 'ATMV': 68.0, 'ADAP': 19.0, 'YDKG': 4.0}
    return test_changes.get(ticker, 0.0)

def get_sentiment_score(ticker: str) -> float:
    """Quick sentiment from mock data (score -1 to 1)."""
    test_sentiments = {'AGMH': 0.85, 'CJET': 0.80, 'ATMV': 0.75, 'ADAP': 0.70, 'YDKG': 0.65}
    return test_sentiments.get(ticker, 0.0)

def get_runner_score(ticker: str) -> float:
    """Historical runner potential: Mocked volatility score (0-1)."""
    test_histories = {'AGMH': 0.95, 'CJET': 0.60, 'ATMV': 0.50, 'ADAP': 0.80, 'YDKG': 0.40}
    return test_histories.get(ticker, 0.0)

def rank_tickers(tickers: List[str]) -> pd.DataFrame:
    """Analyze and rank top 10 tickers."""
    data = []
    for ticker in tickers[:10]:
        try:
            pm_change = get_premarket_change(ticker)
            sentiment = get_sentiment_score(ticker)
            history = get_runner_score(ticker)
            composite = (pm_change / 100) + (sentiment + 1) / 2 + history
            data.append({
                'Ticker': f"${ticker}",
                'Premarket %': f"{pm_change:.1f}%",
                'Sentiment': f"{sentiment:.2f}",
                'History Score': f"{history:.2f}",
                'Composite': f"{composite:.2f}"
            })
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue
    df = pd.DataFrame(data).sort_values('Composite', ascending=False).reset_index(drop=True)
    return df

def send_to_telegram(df: pd.DataFrame, message: str) -> None:
    """Send ranked table as Markdown to Telegram."""
    table_md = df.to_markdown(index=False)
    full_msg = f"{message}\n\n**Ranked Premarket Gainers**\n{table_md}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': full_msg, 'parse_mode': 'Markdown'}
    requests.post(url, json=payload)

if __name__ == "__main__":
    try:
        post_text = TEST_POST
        tickers = extract_tickers(post_text)
        if not tickers:
            raise ValueError("No tickers extracted from mock data.")

        ranked_df = rank_tickers(tickers)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        message = f"Test: Ace's Premarket List Ranked\n\nDate: {timestamp}\nTickers Found: {len(tickers)}"

        send_to_telegram(ranked_df, message)
        print("Test pipeline complete. Sent to Telegram.")
    except Exception as e:
        error_msg = f"Test pipeline failed: {str(e)}"
        send_to_telegram(pd.DataFrame(), error_msg)
        print(error_msg)
