# ACE Alerts (15-min Watch)

- Parses Ace's "Premarket Top % Gainers" tweet.
- Ranks tickers by % plus sentiment/runner enrichments.
- Sends both Markdown table and PNG image to Telegram.
- 15-minute watch loop via CLI or GitHub Actions.

## Local (dry run)
```
python main.py --dry-run --watch-minutes 60 --interval-sec 900 --quick-first --window-start 07:00 --window-end 12:00
```

## Secrets
- `TWITTER_BEARER_TOKEN`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`
