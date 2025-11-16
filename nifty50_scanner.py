#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTY50 3-Day Scanner: News (OpenAI sentiment) + Technicals on 15-min data
Outputs CSV with signal + entry/target/stop and top headlines.
"""
import os, sys, time, json, math, hashlib, argparse, logging, datetime as dt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import requests
import feedparser
from tqdm import tqdm
from dotenv import load_dotenv

# --------------------------- Config & Constants --------------------------- #
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")

# Static NIFTY50 list (editable). Symbols use Yahoo Finance suffix .NS
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","LT.NS","SBIN.NS",
    # "BHARTIARTL.NS","KOTAKBANK.NS","HINDUNILVR.NS","AXISBANK.NS","HCLTECH.NS","MARUTI.NS",
    # "NTPC.NS","BAJFINANCE.NS","TITAN.NS","ASIANPAINT.NS","SUNPHARMA.NS","ULTRACEMCO.NS",
    # "POWERGRID.NS","WIPRO.NS","ONGC.NS","ADANIENT.NS","ADANIPORTS.NS","NESTLEIND.NS",
    # "BAJAJFINSV.NS","M&M.NS","TATASTEEL.NS","HEROMOTOCO.NS","COALINDIA.NS","LTIM.NS",
    # "GRASIM.NS","INDUSINDBK.NS","DRREDDY.NS","JSWSTEEL.NS","HDFCLIFE.NS","CIPLA.NS",
    # "BRITANNIA.NS","DIVISLAB.NS","EICHERMOT.NS","TECHM.NS","BPCL.NS","UPL.NS","HINDALCO.NS",
    # "APOLLOHOSP.NS","BAJAJ-AUTO.NS","TATAMOTORS.NS","TATACONSUM.NS","SHRIRAMFIN.NS"
]

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
SENTIMENT_CACHE = os.path.join(CACHE_DIR, "sentiments.json")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------- Utilities --------------------------- #
def now_ist() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# --------------------------- Data: Prices --------------------------- #
def fetch_ohlcv(symbol: str, days: int = 3, interval: str = "15m") -> pd.DataFrame:
    """
    Fetch OHLCV for 'days' back with 'interval' resolution.
    """
    df = yf.download(symbol, period=f"{days}d", interval=interval, progress=False, auto_adjust=True)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    return df

# --------------------------- Data: News --------------------------- #
def news_from_newsapi(query: str, days: int = 3, page_size: int = 10) -> List[Dict[str, Any]]:
    if not NEWS_API_KEY:
        return []
    from_dt = (now_ist() - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "from": from_dt,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    articles = data.get("articles", [])
    results = []
    for a in articles:
        results.append({
            "title": a.get("title") or "",
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
        })
    return results

def news_from_google(query: str, days: int = 3, limit: int = 10) -> List[Dict[str, Any]]:
    # Fallback: Google News RSS
    feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    d = feedparser.parse(feed_url)
    results = []
    cutoff = now_ist() - dt.timedelta(days=days)
    for entry in d.entries[: 50]:
        title = entry.title
        link = entry.link
        published = entry.get("published_parsed")
        if published:
            pub_dt = dt.datetime(*published[:6])
        else:
            pub_dt = now_ist()
        if pub_dt < cutoff:
            continue
        results.append({
            "title": title,
            "url": link,
            "publishedAt": pub_dt.isoformat(),
            "source": "GoogleNewsRSS",
        })
        if len(results) >= limit:
            break
    return results

def get_latest_news(symbol: str, company_hint: Optional[str] = None, days: int = 3, limit: int = 12) -> List[Dict[str, Any]]:
    query = company_hint or symbol.replace(".NS", "")
    items = news_from_newsapi(query, days=days, page_size=limit)
    if not items:
        items = news_from_google(query, days=days, limit=limit)
    return items

# --------------------------- Sentiment: OpenAI --------------------------- #
def openai_classify_headlines(headlines: List[str], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """
    Classify headlines into positive/negative/neutral with impact score -2..+2.
    Uses basic caching to avoid repeated costs.
    """
    cache = load_cache(SENTIMENT_CACHE)
    results = []
    to_query = []
    for h in headlines:
        key = md5(h.strip().lower())
        if key in cache:
            results.append(cache[key])
        else:
            to_query.append(h)

    if to_query:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing. Put it in .env")
        url = "https://api.openai.com/v1/chat/completions"
        system = (
            "You are a markets analyst. For each headline, classify sentiment for the STOCK PRICE reaction "
            "as positive, negative, or neutral, and provide an impact score from -2 to +2 "
            "(-2 strongly bearish, +2 strongly bullish). Return only JSON with an array 'items' "
            "of objects {headline, sentiment, impact, reason}."
        )
        user = "Headlines:\n" + "\n".join([f"- {h}" for h in to_query])
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":user}
            ]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            for item in parsed.get("items", []):
                headline = item.get("headline","")
                sentiment = (item.get("sentiment","neutral") or "neutral").lower()
                impact = float(item.get("impact",0))
                reason = item.get("reason","")
                key = md5(headline.strip().lower())
                cache[key] = {"headline": headline, "sentiment": sentiment, "impact": impact, "reason": reason}
        except Exception as e:
            logging.exception("Failed to parse OpenAI JSON, falling back to neutral.")
            for h in to_query:
                key = md5(h.strip().lower())
                cache[key] = {"headline": h, "sentiment": "neutral", "impact": 0.0, "reason": "parse_error"}

        save_cache(SENTIMENT_CACHE, cache)

        # Add newly cached to results
        for h in to_query:
            key = md5(h.strip().lower())
            results.append(cache[key])

    # Reorder results to match input order (mix cached + new)
    ordered = []
    for h in headlines:
        key = md5(h.strip().lower())
        ordered.append(cache.get(key, {"headline": h, "sentiment": "neutral", "impact": 0.0, "reason": "cache_miss"}))
    return ordered

def aggregate_sentiment(items: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """
    Aggregate impact scores and return (score, details)
    """
    if not items:
        return 0.0, {"pos":0,"neg":0,"neu":0,"top": []}
    pos = sum(1 for x in items if x["sentiment"]=="positive")
    neg = sum(1 for x in items if x["sentiment"]=="negative")
    neu = sum(1 for x in items if x["sentiment"]=="neutral")
    score = float(np.clip(sum(safe_float(x["impact"],0.0) for x in items), -4.0, 4.0))
    # pick top 3 absolute impact
    top = sorted(items, key=lambda x: abs(safe_float(x["impact"],0.0)), reverse=True)[:3]
    return score, {"pos":pos, "neg":neg, "neu":neu, "top": top}

# --------------------------- Technicals --------------------------- #
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    # MAs / EMAs
    out["SMA50"]   = out["Close"].rolling(50, min_periods=1).mean()
    out["SMA200"]  = out["Close"].rolling(200, min_periods=1).mean()
    out["EMA20"]   = ta.ema(out["Close"], length=20)
    out["EMA50"]   = ta.ema(out["Close"], length=50)

    # MACD (robust to None or empty)
    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    if macd is not None and hasattr(macd, "empty") and not macd.empty:
        # Columns are typically: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # Use .iloc to be safe across ta versions.
        out["MACD"]  = macd.iloc[:, 0]
        out["MACDS"] = macd.iloc[:, 1]
    else:
        out["MACD"]  = np.nan
        out["MACDS"] = np.nan

    # RSI + Volume MA
    out["RSI"]     = ta.rsi(out["Close"], length=14)
    out["VolMA20"] = out["Volume"].rolling(20, min_periods=1).mean()

    return out


def detect_crossovers(row) -> Dict[str, bool]:
    golden = False
    death  = False
    if not (math.isnan(row["SMA50"]) or math.isnan(row["SMA200"])):
        golden = row["SMA50"] > row["SMA200"]
        death  = row["SMA50"] < row["SMA200"]
    return {"golden": golden, "death": death}

def trend_structure(df: pd.DataFrame) -> str:
    if df.empty:
        return "unknown"
    last = df.iloc[-1]
    if last["Close"] > last.get("SMA50", last["Close"]) > last.get("SMA200", last["Close"]):
        return "uptrend"
    if last["Close"] < last.get("SMA50", last["Close"]) < last.get("SMA200", last["Close"]):
        return "downtrend"
    return "sideways"

def last_swing_levels(df: pd.DataFrame, lookback: int = 30) -> Tuple[float, float]:
    if len(df) < lookback:
        segment = df
    else:
        segment = df.iloc[-lookback:]
    swing_high = segment["High"].rolling(5).max().iloc[-5:-1].max()
    swing_low  = segment["Low"].rolling(5).min().iloc[-5:-1].min()
    return float(swing_high), float(swing_low)

def derive_signal(df15: pd.DataFrame, sentiment_score: float) -> Tuple[str, int]:
    """
    Combine indicators + sentiment into a discrete signal.
    Returns (label, score)
    """
    if df15.empty or len(df15) < 50:
        return "NoData", 0
    last = df15.iloc[-1]
    score = 0

    # Trend
    tr = trend_structure(df15)
    if tr == "uptrend": score += 2
    elif tr == "downtrend": score -= 2

    # RSI
    if safe_float(last["RSI"], 50) < 40: score += 1
    elif safe_float(last["RSI"], 50) > 70: score -= 1

    # MACD (skip penalty if NaN)
    if pd.notna(last.get("MACD")) and pd.notna(last.get("MACDS")):
        if float(last["MACD"]) > float(last["MACDS"]):
            score += 1
        else:
            score -= 1

    # Sentiment
    score += round(sentiment_score)  # -4..4

    label = "Neutral"
    if score >= 4: label = "Strong Buy"
    elif score >= 2: label = "Buy"
    elif score <= -4: label = "Strong Sell"
    elif score <= -2: label = "Sell"
    return label, score

def entry_exit(df15: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Entry: close > last swing high AND volume > VolMA20
    Stop: last swing low
    Target: entry + 1.5 * (entry - stop)
    Returns (entry, target, stop, note)
    """
    if df15.empty or len(df15) < 40:
        return None, None, None, "insufficient data"

    swing_high, swing_low = last_swing_levels(df15, lookback=40)
    last = df15.iloc[-1]
    cond_breakout = last["Close"] > swing_high and last["Volume"] > safe_float(last.get("VolMA20"), 0)
    if cond_breakout and last["Close"] > last.get("EMA20", last["Close"]):
        entry = float(last["Close"])
        stop  = float(swing_low)
        risk  = max(0.05 * entry, entry - stop)  # guard: at least 5% of price if swing too tight
        target = entry + 1.5 * risk
        return entry, target, stop, "breakout with volume"
    else:
        # If not breakout: propose buy-the-dip near EMA20 (trend only)
        if last["Close"] > last.get("EMA50", last["Close"]) and last.get("EMA20", last["Close"]) > last.get("EMA50", last["Close"]):
            entry = float(last.get("EMA20"))
            stop  = float(max(df15["Low"].iloc[-10:]))
            risk  = 0.03 * entry
            target = entry + 1.2 * risk
            return entry, target, stop, "trend pullback near EMA20"
        return None, None, None, "no clean setup"

# --------------------------- Orchestration --------------------------- #
def analyze_symbol(symbol: str, model: str, days: int, make_charts: bool=False, charts_dir: str="charts") -> Dict[str, Any]:
    # 1) Data
    df15 = fetch_ohlcv(symbol, days=days, interval="15m")
    if df15.empty:
        return {"symbol": symbol, "error": "no_price_data"}
    df15 = compute_indicators(df15)

    # 2) News & Sentiment
    company_hint = symbol.replace(".NS","")
    news_items = get_latest_news(company_hint, company_hint=company_hint, days=days, limit=12)
    headlines = [x["title"] for x in news_items if x.get("title")]
    sentiments = openai_classify_headlines(headlines, model=model) if headlines else []
    sentiment_score, sentiment_detail = aggregate_sentiment(sentiments)

    # 3) Signal
    label, score = derive_signal(df15, sentiment_score)

    # 4) Entry/Exit
    entry, target, stop, note = entry_exit(df15)

    # 5) Optional chart
    if make_charts:
        try:
            import plotly.graph_objects as go
            os.makedirs(charts_dir, exist_ok=True)
            fig = go.Figure(data=[go.Candlestick(
                x=df15.index, open=df15['Open'], high=df15['High'], low=df15['Low'], close=df15['Close']
            )])
            fig.update_layout(title=f"{symbol} 15m", xaxis_title="Time", yaxis_title="Price")
            fp = os.path.join(charts_dir, f"{symbol.replace('.NS','')}_15m.html")
            fig.write_html(fp, include_plotlyjs="cdn")
        except Exception as e:
            logging.warning("Failed to generate chart for %s: %s", symbol, e)

    # 6) Assemble result
    top_news = []
    for it in sentiment_detail.get("top", []):
        # find url from original news_items
        url = ""
        for n in news_items:
            if n["title"] == it["headline"]:
                url = n.get("url","")
                break
        top_news.append({
            "headline": it["headline"],
            "sentiment": it["sentiment"],
            "impact": it["impact"],
            "url": url
        })

    last = df15.iloc[-1]
    cross = detect_crossovers(last)
    return {
        "symbol": symbol,
        "time": str(now_ist()),
        "close": float(last["Close"]),
        "rsi": float(safe_float(last["RSI"])),
        "macd_vs_signal": float(safe_float(last["MACD"]) - safe_float(last["MACDS"])),
        "trend": trend_structure(df15),
        "golden_cross": bool(cross["golden"]),
        "death_cross": bool(cross["death"]),
        "sentiment_score": float(sentiment_score),
        "sentiment_counts": sentiment_detail,
        "signal": label,
        "score": int(score),
        "entry": entry,
        "target": target,
        "stop": stop,
        "setup_note": note,
        "top_news": top_news[:3]
    }

def main():
    parser = argparse.ArgumentParser(description="NIFTY50 3-day scanner: news + technicals.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (e.g., gpt-4o-mini)")
    parser.add_argument("--days", type=int, default=3, help="Lookback days for prices/news")
    parser.add_argument("--limit", type=int, default=50, help="How many symbols from NIFTY50 to scan")
    parser.add_argument("--out", default="results.csv", help="CSV output path")
    parser.add_argument("--charts", action="store_true", help="Save Plotly HTML charts")
    args = parser.parse_args()

    symbols = NIFTY50[: args.limit]
    rows = []
    logging.info("Scanning %d symbols over %d days...", len(symbols), args.days)

    for sym in tqdm(symbols):
        try:
            res = analyze_symbol(sym, model=args.model, days=args.days, make_charts=args.charts)
            rows.append(res)
        except Exception as e:
            logging.exception("Error for %s: %s", sym, e)
            rows.append({"symbol": sym, "error": str(e)})

    # Normalize and save CSV
    df = pd.json_normalize(rows, sep="__")
    df.sort_values(["signal","score"], ascending=[True, False], inplace=True, na_position="last")
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
