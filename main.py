import tweepy
import yfinance as yf
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import re
import os
from typing import List, Dict
import time

# Config (from env/GitHub secrets)
TWITTER_BEARER = os.getenv('TWITTER_BEARER_TOKEN')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Initialize clients
twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER)
analyzer = SentimentIntensityAnalyzer()

def fetch_latest_post(username: str = 'AceOfWallSt') -> str:
    """Fetch latest tweet from @AceOfWallSt containing 'Premarket Top % Gainers' with retry logic."""
    query = f'from:{username} "Premarket Top % Gainers"'
    for attempt in range(3):  # Retry 3 times
        try:
            tweets = twitter_client.search_recent_tweets(query=query, max_results=1, tweet_fields=['created_at'])
            if not tweets.data:
                raise ValueError("No premarket post found today.")
            return tweets.data[0].text
        except tweepy.TooManyRequests:
            if attempt == 2:  # Last attempt
                raise  # Re-raise to trigger error handling
            time.sleep(60 * (attempt + 1))  # Wait 1, 2, 3 mins
            continue
        except Exception as e:
            raise e
    raise ValueError("Rate limit exceeded after retries.")

def extract_tickers(text: str) -> List[str]:
    """Extract $TICKERs via regex."""
    return re.findall(r'\$([A-Z]{1,5})', text.upper())

def get_premarket_change(ticker: str) -> float:
    """Get estimated premarket % change via yfinance (premarket flag)."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1d', prepost=True)
    if len(hist) < 2:
        return 0.0
    current = hist['Close'][-1]
    prev = hist['Close'][-2]
    return ((current - prev) / prev) * 100 if prev > 0 else 0.0

def get_sentiment_score(ticker: str) -> float:
    """Quick sentiment from recent X posts (score -1 to 1)."""
    query = f'${ticker} lang:en -is:retweet'
    tweets = twitter_client.search_recent_tweets(query=query, max_results=20)
    if not tweets.data:
        return 0.0
    scores = [analyzer.polarity_scores(tweet.text)['compound'] for tweet in tweets.data]
    return sum(scores) / len(scores)

def get_runner_score(ticker: str) -> float:
    """Historical runner potential: Volatility score (0-1) from 52w range."""
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period='1y')
    if hist.empty or 'fiftyTwoWeekHigh' not in info or 'fiftyTwoWeekLow' not in info:
        return 0.0
    range_pct = ((info['fiftyTwoWeekHigh'] - info['fiftyTwoWeekLow']) / info['fiftyTwoWeekLow']) * 100 if info['fiftyTwoWeekLow'] > 0 else 0
    return min(range_pct / 100, 1.0)  # Cap at 1 for >100% volatility

def rank_tickers(tickers: List[str]) -> pd.DataFrame:
    """Analyze and rank top 10 tickers."""
    data = []
    for ticker in tickers[:10]:  # Limit for speed
        try:
            pm_change = get_premarket_change(ticker)
            sentiment = get_sentiment_score(ticker)
            history = get_runner_score(ticker)
            composite = (pm_change / 100) + (sentiment + 1) / 2 + history  # Normalize to 0-3 scale
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
        post_text = fetch_latest_post()
        tickers = extract_tickers(post_text)
        if not tickers:
            raise ValueError("No tickers extracted.")
        
        ranked_df = rank_tickers(tickers)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M ET")
        message = f"Ace's Premarket List Ranked\n\nDate: {timestamp}\nTickers Found: {len(tickers)}"
        
        send_to_telegram(ranked_df, message)
        print("Pipeline complete. Sent to Telegram.")
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        send_to_telegram(pd.DataFrame(), error_msg)  # Send error notification
        print(error_msg)
