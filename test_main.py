import pandas as pd
from datetime import datetime

TEST_POST = "Premarket Top % Gainers: $AGMH +230%, $CJET +150%, $ATMV +68%, $ADAP +19%, $YDKG +4%"

def extract_tickers(text):
    import re
    return re.findall(r"\$([A-Z]{1,5})", text.upper())

def parse_tickers_with_pct(text):
    import re
    norm = text.upper().replace("＋","+").replace("−","-").replace("％","%")
    pattern = re.compile(r"\$([A-Z]{1,5})\s*([+\-]?\d+(?:\.\d+)?)\s*%")
    out = []
    seen = set()
    for m in pattern.finditer(norm):
        t = m.group(1); p = float(m.group(2))
        if t not in seen:
            seen.add(t); out.append((t,p))
    return out

def rank_tickers(pairs):
    rows = []
    for t, pm in pairs:
        comp = round(pm * 0.70, 3)  # simplified for test determinism
        rows.append({"Ticker": t, "Premarket%": round(pm, 2), "Sentiment": 0.0, "Runner": 0.0, "Composite": comp})
    return pd.DataFrame(rows).sort_values("Composite", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    pairs = parse_tickers_with_pct(TEST_POST)
    df = rank_tickers(pairs)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    msg = f"Test: Ace's Premarket List Ranked\nDate: {ts}\nTickers Found: {len(pairs)}"
    print(msg); print(df.to_markdown(index=False))
