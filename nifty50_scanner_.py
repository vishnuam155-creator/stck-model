#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIFTY50 3-Day Scanner: News sentiment + Technicals on 15-min data
- FinBERT (local) or OpenAI for news sentiment
- 15m technicals: RSI, MACD, EMAs/SMAs, volume
- True daily 50/200 DMA flags
- Entry/Target/Stop suggestions
- Markdown summary table and CSV output

Run:
  python nifty50_scanner.py --sentiment finbert --days 3 --limit 50 --out results.csv
"""
import os, json, math, argparse, logging, datetime as dt
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
import feedparser
from tqdm import tqdm
from dotenv import load_dotenv

# --------------------------- Setup --------------------------- #
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")

NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","LT.NS","SBIN.NS",
    "BHARTIARTL.NS","KOTAKBANK.NS","HINDUNILVR.NS","AXISBANK.NS","HCLTECH.NS","MARUTI.NS",
    "NTPC.NS","BAJFINANCE.NS","TITAN.NS","ASIANPAINT.NS","SUNPHARMA.NS","ULTRACEMCO.NS",
    "POWERGRID.NS","WIPRO.NS","ONGC.NS","ADANIENT.NS","ADANIPORTS.NS","NESTLEIND.NS",
    "BAJAJFINSV.NS","M&M.NS","TATASTEEL.NS","HEROMOTOCO.NS","COALINDIA.NS","LTIM.NS",
    "GRASIM.NS","INDUSINDBK.NS","DRREDDY.NS","JSWSTEEL.NS","HDFCLIFE.NS","CIPLA.NS",
    "BRITANNIA.NS","DIVISLAB.NS","EICHERMOT.NS","TECHM.NS","BPCL.NS","UPL.NS","HINDALCO.NS",
    "APOLLOHOSP.NS","BAJAJ-AUTO.NS","TATAMOTORS.NS","TATACONSUM.NS","SHRIRAMFIN.NS"
]

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
SENTIMENT_CACHE = os.path.join(CACHE_DIR, "sentiments.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------- Utils --------------------------- #
def now_ist() -> dt.datetime:
    """Timezone-aware IST timestamp (UTC+5:30)."""
    return dt.datetime.now(dt.timezone.utc).astimezone(
        dt.timezone(dt.timedelta(hours=5, minutes=30))
    )

def md5(s: str) -> str:
    import hashlib
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
    """Convert to float; unwrap 0/1-length Series if needed."""
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            x = x.iloc[0]
        return float(x)
    except Exception:
        return default

def row_scalar(row: pd.Series, col: str, default=np.nan):
    """Return a scalar value from a row (handles accidental 1-elem Series)."""
    if col not in row.index:
        return default
    v = row[col]
    if isinstance(v, pd.Series):
        return safe_float(v.iloc[0], default)
    return safe_float(v, default)

# --------------------------- Prices --------------------------- #
def fetch_ohlcv(symbol: str, days: int = 3, interval: str = "15m") -> pd.DataFrame:
    df = yf.download(symbol, period=f"{days}d", interval=interval, progress=False, auto_adjust=True)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all").rename(columns=str.title)
    return df

# --------------------------- News --------------------------- #
def news_from_newsapi(query: str, days: int = 3, page_size: int = 10) -> List[Dict[str, Any]]:
    if not NEWS_API_KEY:
        return []
    from_dt = (now_ist() - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query, "language": "en", "from": from_dt, "sortBy": "publishedAt",
        "pageSize": page_size, "apiKey": NEWS_API_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    res = []
    for a in data.get("articles", []):
        res.append({
            "title": a.get("title") or "",
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
        })
    return res

def news_from_google(query: str, days: int = 3, limit: int = 10) -> List[Dict[str, Any]]:
    feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    d = feedparser.parse(feed_url)
    results = []
    cutoff = now_ist() - dt.timedelta(days=days)
    for entry in d.entries[:50]:
        title = entry.title
        link = entry.link
        published = entry.get("published_parsed")
        pub_dt = dt.datetime(*published[:6]) if published else now_ist()
        # ensure timezone-aware (IST) if needed
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        if pub_dt < cutoff:
            continue
        results.append({
            "title": title,
            "url": link,
            "publishedAt": pub_dt.isoformat(),
            "source": "GoogleNewsRSS"
        })
        if len(results) >= limit:
            break
    return results

def get_latest_news(symbol: str, company_hint: Optional[str] = None, days: int = 3, limit: int = 12):
    query = company_hint or symbol.replace(".NS", "")
    items = news_from_newsapi(query, days=days, page_size=limit)
    if not items:
        items = news_from_google(query, days=days, limit=limit)
    return items

# ---------------- Sentiment Engines (FinBERT / OpenAI) ---------------- #
def openai_classify_headlines(headlines: List[str], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    cache = load_cache(SENTIMENT_CACHE)
    out, to_query = [], []
    for h in headlines:
        key = md5(h.strip().lower())
        (out if key in cache else to_query).append(cache.get(key, h))
    if to_query:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing. Use --sentiment finbert or set the key.")
        url = "https://api.openai.com/v1/chat/completions"
        system = (
            "You are a markets analyst. For each headline, classify sentiment for the STOCK PRICE reaction "
            "as positive, negative, or neutral, and provide an impact score from -2 to +2 "
            "(-2 strongly bearish, +2 strongly bullish). Return only JSON with an array 'items' "
            "of objects {headline, sentiment, impact, reason}."
        )
        user = "Headlines:\n" + "\n".join([f"- {h}" for h in to_query if isinstance(h, str)])
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            for item in parsed.get("items", []):
                headline  = item.get("headline","")
                sentiment = (item.get("sentiment","neutral") or "neutral").lower()
                impact    = float(item.get("impact",0))
                reason    = item.get("reason","")
                key = md5(headline.strip().lower())
                cache[key] = {"headline": headline, "sentiment": sentiment, "impact": impact, "reason": reason}
        except Exception:
            for h in to_query:
                key = md5(h.strip().lower())
                cache[key] = {"headline": h, "sentiment": "neutral", "impact": 0.0, "reason": "parse_error"}
        save_cache(SENTIMENT_CACHE, cache)
        for h in to_query:
            key = md5(h.strip().lower())
            out.append(cache[key])
    ordered = []
    for h in headlines:
        key = md5(h.strip().lower())
        ordered.append(cache.get(key, {"headline": h, "sentiment": "neutral", "impact": 0.0, "reason": "cache_miss"}))
    return ordered

_FINBERT = {"tokenizer": None, "model": None}
def finbert_load():
    global _FINBERT
    if _FINBERT["tokenizer"] is not None and _FINBERT["model"] is not None:
        return _FINBERT
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _FINBERT["tokenizer"] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _FINBERT["model"] = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return _FINBERT

def finbert_classify_headlines(headlines: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
    cache = load_cache(SENTIMENT_CACHE)
    out, to_run = [], []
    for h in headlines:
        key = md5(h.strip().lower())
        (out if (key in cache and "impact" in cache[key]) else to_run).append(cache.get(key, h))
    if to_run:
        fb = finbert_load()
        tok, mdl = fb["tokenizer"], fb["model"]
        import torch
        mdl.eval()
        with torch.no_grad():
            for i in range(0, len(to_run), batch_size):
                chunk = [x for x in to_run[i:i+batch_size] if isinstance(x, str)]
                if not chunk:
                    continue
                enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128)
                logits = mdl(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # FinBERT labels: [negative, neutral, positive]
                for idx, headline in enumerate(chunk):
                    p_neg, p_neu, p_pos = probs[idx].tolist()
                    if p_pos >= p_neg and p_pos >= p_neu:
                        sentiment = "positive"
                    elif p_neg >= p_pos and p_neg >= p_neu:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"
                    impact = float(np.clip((p_pos - p_neg) * 2.0, -2.0, 2.0))
                    key = md5(headline.strip().lower())
                    cache[key] = {
                        "headline": headline,
                        "sentiment": sentiment,
                        "impact": impact,
                        "reason": f"FinBERT p_pos={p_pos:.2f}, p_neg={p_neg:.2f}, p_neu={p_neu:.2f}"
                    }
        save_cache(SENTIMENT_CACHE, cache)
        for h in to_run:
            key = md5(h.strip().lower())
            out.append(cache[key])
    ordered = []
    for h in headlines:
        key = md5(h.strip().lower())
        ordered.append(cache.get(key, {"headline": h, "sentiment": "neutral", "impact": 0.0, "reason": "cache_miss"}))
    return ordered

def aggregate_sentiment(items: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    if not items:
        return 0.0, {"pos":0,"neg":0,"neu":0,"top": []}
    pos = sum(1 for x in items if x["sentiment"]=="positive")
    neg = sum(1 for x in items if x["sentiment"]=="negative")
    neu = sum(1 for x in items if x["sentiment"]=="neutral")
    score = float(np.clip(sum(safe_float(x["impact"],0.0) for x in items), -4.0, 4.0))
    top = sorted(items, key=lambda x: abs(safe_float(x["impact"],0.0)), reverse=True)[:3]
    return score, {"pos":pos, "neg":neg, "neu":neu, "top": top}

# --------------------------- Technicals --------------------------- #
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["SMA50"]   = out["Close"].rolling(50, min_periods=1).mean()
    out["SMA200"]  = out["Close"].rolling(200, min_periods=1).mean()
    out["EMA20"]   = ta.ema(out["Close"], length=20)
    out["EMA50"]   = ta.ema(out["Close"], length=50)

    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    if macd is not None and hasattr(macd, "empty") and not macd.empty:
        out["MACD"]  = macd.iloc[:, 0]
        out["MACDS"] = macd.iloc[:, 1]
    else:
        out["MACD"]  = np.nan
        out["MACDS"] = np.nan

    out["RSI"]     = ta.rsi(out["Close"], length=14)
    out["VolMA20"] = out["Volume"].rolling(20, min_periods=1).mean()
    return out

def trend_structure(df: pd.DataFrame) -> str:
    if df.empty:
        return "unknown"
    last = df.iloc[-1]
    close  = row_scalar(last, "Close")
    sma50  = row_scalar(last, "SMA50", close)
    sma200 = row_scalar(last, "SMA200", close)
    if not (np.isfinite(close) and np.isfinite(sma50) and np.isfinite(sma200)):
        return "sideways"
    if (close > sma50) and (sma50 > sma200):
        return "uptrend"
    if (close < sma50) and (sma50 < sma200):
        return "downtrend"
    return "sideways"

def last_swing_levels(df: pd.DataFrame, lookback: int = 30) -> Tuple[float, float]:
    seg = df.iloc[-lookback:] if len(df) >= lookback else df
    swing_high = seg["High"].rolling(5, min_periods=1).max().iloc[-5:-1].max()
    swing_low  = seg["Low"].rolling(5, min_periods=1).min().iloc[-5:-1].min()
    if isinstance(swing_high, pd.Series) and not swing_high.empty:
        swing_high = swing_high.iloc[0]
    if isinstance(swing_low, pd.Series) and not swing_low.empty:
        swing_low = swing_low.iloc[0]
    return float(swing_high), float(swing_low)

def derive_signal(df15: pd.DataFrame, sentiment_score: float) -> Tuple[str, int]:
    if df15.empty or len(df15) < 50:
        return "NoData", 0
    last = df15.iloc[-1]
    score = 0
    tr = trend_structure(df15)
    if tr == "uptrend": score += 2
    elif tr == "downtrend": score -= 2
    rsi = row_scalar(last, "RSI", 50)
    if rsi < 40: score += 1
    elif rsi > 70: score -= 1
    macd  = row_scalar(last, "MACD")
    macds = row_scalar(last, "MACDS")
    if np.isfinite(macd) and np.isfinite(macds):
        score += 1 if macd > macds else -1
    score += round(sentiment_score)  # -4..+4
    if score >= 4:   return "Strong Buy", score
    if score >= 2:   return "Buy", score
    if score <= -4:  return "Strong Sell", score
    if score <= -2:  return "Sell", score
    return "Neutral", score

def entry_exit(df15: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    if df15.empty or len(df15) < 40:
        return None, None, None, "insufficient data"
    swing_high, swing_low = last_swing_levels(df15, lookback=40)
    last = df15.iloc[-1]
    vol_ma = row_scalar(last, "VolMA20", 0.0)
    cond_breakout = (row_scalar(last, "Close") > swing_high) and (row_scalar(last, "Volume") > vol_ma)
    if cond_breakout and row_scalar(last, "Close") > row_scalar(last, "EMA20"):
        entry = float(row_scalar(last, "Close"))
        stop  = float(swing_low)
        risk  = max(0.05 * entry, entry - stop)
        target = entry + 1.5 * risk
        return entry, target, stop, "breakout with volume"
    else:
        close = row_scalar(last, "Close")
        ema20 = row_scalar(last, "EMA20", close)
        ema50 = row_scalar(last, "EMA50", close)
        if (close > ema50) and (ema20 > ema50):
            entry = float(ema20)
            stop  = float(max(df15["Low"].iloc[-10:]))
            risk  = 0.03 * entry
            target = entry + 1.2 * risk
            return entry, target, stop, "trend pullback near EMA20"
        return None, None, None, "no clean setup"

# --- Daily 50/200 DMA flags for context ---
def daily_ma_flags(symbol: str, lookback_days: int = 400) -> Dict[str, Optional[bool]]:
    try:
        df_d = yf.download(symbol, period=f"{lookback_days}d", interval="1d",
                           progress=False, auto_adjust=True)
        if not isinstance(df_d, pd.DataFrame) or df_d.empty:
            return {"above_50dma": None, "above_200dma": None}
        df_d = df_d.dropna(how="all").rename(columns=str.title)
        df_d["SMA50_D"]  = df_d["Close"].rolling(50,  min_periods=1).mean()
        df_d["SMA200_D"] = df_d["Close"].rolling(200, min_periods=1).mean()
        last = df_d.iloc[-1]
        close = safe_float(last.get("Close"))
        sma50 = safe_float(last.get("SMA50_D"))
        sma200 = safe_float(last.get("SMA200_D"))
        return {
            "above_50dma":  (np.isfinite(close) and np.isfinite(sma50)  and close > sma50)  or False,
            "above_200dma": (np.isfinite(close) and np.isfinite(sma200) and close > sma200) or False,
        }
    except Exception:
        return {"above_50dma": None, "above_200dma": None}

# --- Markdown summary helpers ---
def _sentiment_label(score: float) -> str:
    if score >= 1.0:  return "Positive"
    if score <= -1.0: return "Negative"
    return "Neutral"

def _macd_label(macd_delta: float) -> str:
    if not np.isfinite(macd_delta): return "—"
    return "Bullish" if macd_delta > 0 else "Bearish"

def _dma200_label(flag) -> str:
    if flag is True:  return "Above 200DMA"
    if flag is False: return "Below 200DMA"
    return "—"

def print_markdown_table(df: pd.DataFrame, out_path: Optional[str] = None):
    """Print and optionally save a Markdown table."""
    if df.empty:
        print("No rows to show.")
        return
    view = pd.DataFrame({
        "Stock": df["symbol"].str.replace(".NS","", regex=False),
        "News Sentiment": df["sentiment_score"].apply(_sentiment_label),
        "RSI": df["rsi"].apply(lambda x: f"{x:.0f}" if np.isfinite(x) else "—"),
        "MACD": df["macd_vs_signal"].apply(_macd_label),
        "Trend": df.apply(lambda r: _dma200_label(r.get("above_200dma")), axis=1),
        "Signal": df["signal"],
        "Entry Price": df["entry"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—"),
        "Target": df["target"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—"),
        "Stop Loss": df["stop"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—"),
    })
    headers = list(view.columns)
    align = ["--:", ":-:", ":-:", ":-:", ":-:", ":-:", "-:", "-:", "-:"]
    line = "| " + " | ".join(headers) + " |"
    sep  = "| " + " | ".join(align)   + " |"
    print(line); print(sep)
    for _, row in view.iterrows():
        print("| " + " | ".join(str(v) for v in row.values) + " |")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(line + "\n"); f.write(sep + "\n")
            for _, row in view.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")

# --------------------------- Orchestration --------------------------- #
def analyze_symbol(symbol: str, sentiment_engine: str, openai_model: str, days: int, make_charts: bool=False, charts_dir: str="charts") -> Dict[str, Any]:
    # 1) Price data + indicators
    df15 = fetch_ohlcv(symbol, days=days, interval="15m")
    if df15.empty:
        return {"symbol": symbol, "error": "no_price_data"}
    df15 = compute_indicators(df15)

    # 2) News + sentiment
    company_hint = symbol.replace(".NS","")
    news_items = get_latest_news(company_hint, company_hint=company_hint, days=days, limit=12)
    headlines = [x["title"] for x in news_items if x.get("title")]

    if headlines:
        if sentiment_engine == "finbert":
            sentiments = finbert_classify_headlines(headlines)
        else:
            sentiments = openai_classify_headlines(headlines, model=openai_model)
    else:
        sentiments = []
    sentiment_score, sentiment_detail = aggregate_sentiment(sentiments)

    # 3) Signal + entry/exit
    label, score = derive_signal(df15, sentiment_score)
    entry, target, stop, note = entry_exit(df15)

    # 4) Chart (optional)
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

    # 5) Build result row
    last = df15.iloc[-1]
    sma50  = row_scalar(last, "SMA50")
    sma200 = row_scalar(last, "SMA200")
    cross_golden = bool(np.isfinite(sma50) and np.isfinite(sma200) and (sma50 > sma200))
    cross_death  = bool(np.isfinite(sma50) and np.isfinite(sma200) and (sma50 < sma200))
    macd  = row_scalar(last, "MACD")
    macds = row_scalar(last, "MACDS")
    macd_delta = (macd - macds) if (np.isfinite(macd) and np.isfinite(macds)) else np.nan

    # Daily DMA context
    dma = daily_ma_flags(symbol)
    above_200dma = dma.get("above_200dma")
    above_50dma  = dma.get("above_50dma")

    # Map top 3 news with URLs
    top_news = []
    for it in sentiment_detail.get("top", []):
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

    return {
        "symbol": symbol,
        "time": str(now_ist()),
        "close": float(row_scalar(last, "Close")),
        "rsi": float(row_scalar(last, "RSI")),
        "macd_vs_signal": macd_delta,
        "trend": trend_structure(df15),
        "golden_cross": cross_golden,
        "death_cross": cross_death,
        "above_50dma":  bool(above_50dma) if above_50dma is not None else None,
        "above_200dma": bool(above_200dma) if above_200dma is not None else None,
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
    parser.add_argument("--sentiment", choices=["finbert","openai"], default="finbert", help="Sentiment engine")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (if --sentiment openai)")
    parser.add_argument("--days", type=int, default=3, help="Lookback days for prices/news")
    parser.add_argument("--limit", type=int, default=50, help="How many symbols from NIFTY50 to scan")
    parser.add_argument("--out", default="results.csv", help="CSV output path")
    parser.add_argument("--charts", action="store_true", help="Save Plotly HTML charts")
    args = parser.parse_args()

    symbols = NIFTY50[: args.limit]
    rows = []
    logging.info("Scanning %d symbols over %d days using %s sentiment...", len(symbols), args.days, args.sentiment)

    for sym in tqdm(symbols):
        try:
            res = analyze_symbol(sym, sentiment_engine=args.sentiment, openai_model=args.model, days=args.days, make_charts=args.charts)
            rows.append(res)
        except Exception as e:
            logging.exception("Error for %s: %s", sym, e)
            rows.append({"symbol": sym, "error": str(e)})

    df = pd.json_normalize(rows, sep="__")

    # Ensure columns exist even if many rows errored
    for col, default in [("signal", "Error"), ("score", np.nan)]:
        if col not in df.columns:
            df[col] = default

    # Sort only by columns that exist
    sort_cols = [c for c in ["signal", "score"] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, ascending=[True, False][:len(sort_cols)], inplace=True, na_position="last")

    df.to_csv(args.out, index=False)
    print(f"Saved {args.out} with {len(df)} rows.")

    # Markdown summary
    try:
        print_markdown_table(df, out_path="results.md")
    except Exception as e:
        logging.warning("Could not render markdown table: %s", e)

if __name__ == "__main__":
    main()
