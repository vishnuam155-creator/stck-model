#!/usr/bin/env python3
import os, io, sys, math, time, json, argparse, textwrap, datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import requests
from urllib.parse import quote_plus
from tqdm import tqdm
from dotenv import load_dotenv

# -------- Settings --------
NSE_NIFTY50_CSV = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"  # official list (NSE)
# For NIFTY PCR (optional quick check for context) – can be replaced with your paid datasource
NSE_OPTION_CHAIN_NIFTY = (
    "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
)


# -------------- Utils --------------
def now_ist():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))


def safe_round(x, n=2):
    try:
        return round(float(x), n)
    except:
        return x


def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int):
    return series.rolling(window=length).mean()


def pct(x1, x0):
    try:
        return (x1 / x0 - 1.0) * 100.0
    except:
        return np.nan


def swing_highs_lows(df: pd.DataFrame, lookback=5) -> Tuple[pd.Series, pd.Series]:
    # very simple swing high/low detector
    highs = (
        (
            df["High"]
            .rolling(lookback, center=True)
            .apply(lambda w: w[lookback // 2] == max(w), raw=True)
        )
        .fillna(0)
        .astype(bool)
    )
    lows = (
        (
            df["Low"]
            .rolling(lookback, center=True)
            .apply(lambda w: w[lookback // 2] == min(w), raw=True)
        )
        .fillna(0)
        .astype(bool)
    )
    return highs, lows


# -------------- News --------------
def fetch_news_newsapi(
    query: str, since: dt.datetime, api_key: Optional[str]
) -> List[Dict]:
    if not api_key:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={quote_plus(query)}&from={since.date()}&language=en&sortBy=publishedAt&pageSize=50"
    )
    r = requests.get(url, headers={"X-Api-Key": api_key}, timeout=20)
    if r.status_code != 200:
        return []
    data = r.json()
    arts = data.get("articles", [])
    res = []
    for a in arts:
        res.append(
            {
                "title": a.get("title"),
                "desc": a.get("description"),
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "publishedAt": a.get("publishedAt"),
            }
        )
    return res


def fetch_news_gdelt(query: str, since: dt.datetime) -> List[Dict]:
    # Simple GDELT 2.0 API query via JSON feed (English)
    # Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    q = f"query={quote_plus(query)}&mode=artlist&maxrecords=50&format=json"
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{q}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        arts = data.get("articles", [])
        res = []
        for a in arts:
            # Filter by time window (approximate)
            try:
                ts = dt.datetime.strptime(
                    a.get("seendate")[:14], "%Y%m%d%H%M%S"
                ).replace(tzinfo=dt.timezone.utc)
            except:
                ts = None
            if ts and ts < since.astimezone(dt.timezone.utc):
                continue
            res.append(
                {
                    "title": a.get("title"),
                    "desc": a.get("snippet"),
                    "url": a.get("url"),
                    "source": a.get("sourceCommonName"),
                    "publishedAt": ts.isoformat() if ts else None,
                }
            )
        return res
    except:
        return []


# -------------- Sentiment --------------
from nltk.sentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

_finbert = None


def finbert_pipeline():
    global _finbert
    if _finbert is not None:
        return _finbert
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TextClassificationPipeline,
        )

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert = TextClassificationPipeline(
            model=mdl, tokenizer=tok, return_all_scores=False
        )
        return _finbert
    except Exception as e:
        return None


def sentiment_label(text: str, prefer_finbert=True) -> Tuple[str, float]:
    if not text:
        return ("neutral", 0.0)
    if prefer_finbert:
        pipe = finbert_pipeline()
        if pipe:
            try:
                out = pipe(text[:512])[
                    0
                ]  # {'label': 'positive'|'negative'|'neutral', 'score': float}
                return (out["label"].lower(), float(out["score"]))
            except:
                pass
    s = _sia.polarity_scores(text)
    comp = s["compound"]
    if comp >= 0.2:
        return ("positive", comp)
    if comp <= -0.2:
        return ("negative", comp)
    return ("neutral", comp)


# -------------- Market breadth (optional) --------------
_session = requests.Session()


def get_nifty_pcr_quick() -> Optional[float]:
    # VERY basic PCR from option chain; PCR = total PUT OI / total CALL OI
    try:
        hdrs = {
            "user-agent": "Mozilla/5.0",
            "accept": "application/json,text/html",
            "referer": "https://www.nseindia.com/",
        }
        _session.get("https://www.nseindia.com/", headers=hdrs, timeout=10)
        r = _session.get(NSE_OPTION_CHAIN_NIFTY, headers=hdrs, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        recs = data.get("records", {}).get("data", [])
        put_oi = 0
        call_oi = 0
        for row in recs:
            if "CE" in row:
                call_oi += row["CE"].get("openInterest", 0) or 0
            if "PE" in row:
                put_oi += row["PE"].get("openInterest", 0) or 0
        if call_oi == 0:
            return None
        return float(put_oi) / float(call_oi)
    except:
        return None


# -------------- Signals --------------
@dataclass
class SignalResult:
    symbol: str
    name: str
    strong: str  # "Strong Buy"/"Strong Sell"/"Watch"
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    rrr: Optional[float]
    rsi: Optional[float]
    macd_hist: Optional[float]
    stoch_k: Optional[float]
    news_impact: str
    news_links: List[str]


def load_nifty50_list() -> pd.DataFrame:
    df = pd.read_csv(NSE_NIFTY50_CSV)
    # Columns: Company Name, Industry, Symbol, Series, ISIN Code
    # Use "Symbol" as ticker, append ".NS" for yfinance
    df["YF"] = df["Symbol"].astype(str).str.strip() + ".NS"
    return df


def fetch_prices(symbol_yf: str, days=3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 15-minute for intraday view (needs up to 5d window)
    intraday = yf.download(
        symbol_yf, period="5d", interval="15m", progress=False, auto_adjust=True
    )
    daily = yf.download(
        symbol_yf, period="2y", interval="1d", progress=False, auto_adjust=True
    )
    # trim intraday to last N days (IST sessions)
    if len(intraday) > 0:
        cutoff = now_ist() - dt.timedelta(days=days)
        intraday = intraday[
            intraday.index.tz_localize(None) >= (cutoff.replace(tzinfo=None))
        ]
    return intraday, daily


def compute_indicators(
    intra: pd.DataFrame, daily: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(intra) == 0 or len(daily) == 0:
        return intra, daily
    # intraday indicators
    intra = intra.copy()
    intra["EMA20"] = ta.ema(intra["Close"], length=20)
    intra["EMA50"] = ta.ema(intra["Close"], length=50)
    intra["RSI14"] = ta.rsi(intra["Close"], length=14)
    macd = ta.macd(intra["Close"])
    if macd is not None and not macd.empty:
        intra["MACD"] = macd["MACD_12_26_9"]
        intra["MACDsig"] = macd["MACDs_12_26_9"]
        intra["MACDhist"] = macd["MACDh_12_26_9"]
    stoch = ta.stoch(intra["High"], intra["Low"], intra["Close"])
    if stoch is not None and not stoch.empty:
        intra["STOCHK"] = stoch["STOCHk_14_3_3"]
        intra["STOCHD"] = stoch["STOCHd_14_3_3"]
    intra["VolMA20"] = sma(intra["Volume"], 20)

    # daily trend structure
    daily = daily.copy()
    daily["SMA50"] = sma(daily["Close"], 50)
    daily["SMA200"] = sma(daily["Close"], 200)
    daily["HH"] = (daily["High"] > daily["High"].shift(1)) & (
        daily["Low"] > daily["Low"].shift(1)
    )  # crude
    return intra, daily


def breakout_retest_levels(
    daily: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    # Simple: use last 20d high as resistance; if close > that, breakout. Entry at retest ~ that level.
    if len(daily) < 50:
        return None, None
    last_close = daily["Close"].iloc[-1]
    res = daily["High"].rolling(20).max().iloc[-2]  # prior 20d high
    sup = daily["Low"].rolling(20).min().iloc[-2]  # prior 20d low
    # If breakout above res, entry ~res; if breakdown below sup, entry ~sup (for shorts)
    return res, sup


def classify(
    intra: pd.DataFrame, daily: pd.DataFrame
) -> Tuple[
    str, str, Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """
    Returns: (label, reason, entry, stop, target, rrr)
    Implements your rule-book conservatively.
    """
    if len(intra) == 0 or len(daily) == 0:
        return ("Watch", "No data", None, None, None, None)
    close = daily["Close"].iloc[-1]
    s50 = daily["SMA50"].iloc[-1]
    s200 = daily["SMA200"].iloc[-1]

    # Trend structure
    higher_trend = (
        close > s50
        and close > s200
        and (daily["Close"].iloc[-5:] > daily["Close"].shift(1).iloc[-5:]).sum() >= 3
    )
    lower_trend = (
        close < s50
        and close < s200
        and (daily["Close"].iloc[-5:] < daily["Close"].shift(1).iloc[-5:]).sum() >= 3
    )

    # Crossovers
    golden = (
        daily["SMA50"].iloc[-1] > daily["SMA200"].iloc[-1]
        and daily["SMA50"].iloc[-2] <= daily["SMA200"].iloc[-2]
    )
    death = (
        daily["SMA50"].iloc[-1] < daily["SMA200"].iloc[-1]
        and daily["SMA50"].iloc[-2] >= daily["SMA200"].iloc[-2]
    )

    # Intraday confirmations
    last = intra.iloc[-1]
    rsi = last.get("RSI14", np.nan)
    macd_hist = last.get("MACDhist", np.nan)
    stochk = last.get("STOCHK", np.nan)

    # Breakout / breakdown
    res, sup = breakout_retest_levels(daily)
    strong_buy = False
    strong_sell = False
    reason = []

    if higher_trend:
        reason.append("HH/HL + above 50/200-DMA")
    if lower_trend:
        reason.append("LH/LL + below 50/200-DMA")
    if golden:
        reason.append("Golden cross")
    if death:
        reason.append("Death cross")

    # Volume confirmation
    vol_ok = (intra["Close"].pct_change().iloc[-5:] > 0).sum() >= 3 and (
        intra["Volume"].iloc[-5:].mean() > (intra["VolMA20"].iloc[-5:].mean() or 1)
    )

    # Buy case
    if (
        higher_trend
        and vol_ok
        and rsi >= 50
        and (not np.isnan(macd_hist) and macd_hist > 0)
    ):
        # Breakout check
        if res and daily["Close"].iloc[-1] > res:
            # retest entry = res, SL = res - 1.5 ATR(14), target = res + 2 * ATR
            atr = ta.atr(daily["High"], daily["Low"], daily["Close"], length=14).iloc[
                -1
            ]
            entry = float(res)
            stop = (
                float(entry - 1.5 * atr) if not math.isnan(atr) else float(entry * 0.98)
            )
            target = (
                float(entry + 2.0 * atr) if not math.isnan(atr) else float(entry * 1.03)
            )
            rrr = (target - entry) / max(1e-6, (entry - stop))
            strong_buy = True
            reason.append("Breakout + bullish retest setup")
            return (
                "Strong Buy",
                ", ".join(reason),
                entry,
                stop,
                target,
                safe_round(rrr, 2),
            )

    # Sell case
    if lower_trend and rsi <= 50 and (not np.isnan(macd_hist) and macd_hist < 0):
        if sup and daily["Close"].iloc[-1] < sup:
            atr = ta.atr(daily["High"], daily["Low"], daily["Close"], length=14).iloc[
                -1
            ]
            entry = float(sup)
            stop = (
                float(entry + 1.5 * atr) if not math.isnan(atr) else float(entry * 1.02)
            )
            target = (
                float(entry - 2.0 * atr) if not math.isnan(atr) else float(entry * 0.97)
            )
            rrr = (entry - target) / max(1e-6, (stop - entry))
            strong_sell = True
            reason.append("Breakdown + failed retest setup")
            return (
                "Strong Sell",
                ", ".join(reason),
                entry,
                stop,
                target,
                safe_round(rrr, 2),
            )

    return (
        "Watch",
        ", ".join(reason) if reason else "Mixed/No confirmation",
        None,
        None,
        None,
        None,
    )


def trend_last_n_days(daily: pd.DataFrame, n=3) -> str:
    if len(daily) < n + 1:
        return "flat"
    ret = pct(daily["Close"].iloc[-1], daily["Close"].iloc[-n - 1])
    if ret is None or math.isnan(ret):
        return "flat"
    if ret > 1.5:
        return "continuously_increasing"
    if ret < -1.5:
        return "continuously_decreasing"
    return "flat"


def analyze_symbol(row, days=3, prefer_finbert=True) -> Optional[SignalResult]:
    name = row["Company Name"]
    sym = row["Symbol"]
    yf_sym = row["YF"]
    intra, daily = fetch_prices(yf_sym, days=days)
    if len(intra) == 0 or len(daily) == 0:
        return None
    intra, daily = compute_indicators(intra, daily)
    label, reason, entry, stop, target, rrr = classify(intra, daily)

    # News (3 days)
    since = now_ist() - dt.timedelta(days=days)
    api_key = os.getenv("NEWSAPI_KEY", "").strip() or None
    q = f'"{name}" OR {sym} stock OR {sym} NSE'
    news = fetch_news_newsapi(q, since, api_key) or fetch_news_gdelt(q, since)

    # score sentiment
    pos, neg = 0, 0
    links = []
    for a in news[:10]:
        txt = f"{a.get('title','')} {a.get('desc','')}"
        lab, conf = sentiment_label(txt, prefer_finbert=prefer_finbert)
        if lab == "positive":
            pos += 1
        elif lab == "negative":
            neg += 1
        links.append(a.get("url"))
    if pos > neg:
        news_tag = f"positive ({pos} vs {neg})"
    elif neg > pos:
        news_tag = f"negative ({neg} vs {pos})"
    else:
        news_tag = "mixed/neutral"

    # continuous trend note
    trend_tag = trend_last_n_days(daily, n=days)
    if trend_tag != "flat":
        news_tag += f"; {trend_tag}"

    last_rsi = safe_round(intra["RSI14"].iloc[-1], 1)
    last_macdh = (
        safe_round(intra["MACDhist"].iloc[-1], 3) if "MACDhist" in intra else None
    )
    last_stochk = safe_round(intra["STOCHK"].iloc[-1], 1) if "STOCHK" in intra else None

    return SignalResult(
        symbol=sym,
        name=name,
        strong=label,
        reason=reason,
        entry=entry,
        stop=stop,
        target=target,
        rrr=rrr,
        rsi=last_rsi,
        macd_hist=last_macdh,
        stoch_k=last_stochk,
        news_impact=news_tag,
        news_links=links,
    )


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--days", type=int, default=3, help="Lookback window for news & trend notes"
    )
    ap.add_argument(
        "--max-stocks", type=int, default=50, help="Limit scan for speed while testing"
    )
    ap.add_argument(
        "--make-report", action="store_true", help="Save markdown report & CSV"
    )
    ap.add_argument(
        "--finbert",
        action="store_true",
        help="Prefer FinBERT sentiment (needs transformers)",
    )
    args = ap.parse_args()

    print(f"[{now_ist().strftime('%Y-%m-%d %H:%M')}] Loading NIFTY 50 list from NSE…")
    df_list = load_nifty50_list()
    if args.max_stocks < len(df_list):
        df_list = df_list.head(args.max_stocks)

    results: List[SignalResult] = []
    for _, row in tqdm(df_list.iterrows(), total=len(df_list), desc="Scanning"):
        try:
            r = analyze_symbol(row, days=args.days, prefer_finbert=args.finbert)
            if r:
                results.append(r)
        except Exception as e:
            # keep going
            continue

    # Rank: Strong Buy (RRR desc) → Strong Sell (RRR desc)
    buys = [r for r in results if r.strong == "Strong Buy"]
    sells = [r for r in results if r.strong == "Strong Sell"]
    buys.sort(key=lambda x: (x.rrr or 0), reverse=True)
    sells.sort(key=lambda x: (x.rrr or 0), reverse=True)

    # Console summary
    def rowfmt(r: SignalResult):
        return [
            r.symbol,
            r.strong,
            safe_round(r.entry, 2),
            safe_round(r.stop, 2),
            safe_round(r.target, 2),
            safe_round(r.rrr, 2),
            r.rsi,
            r.macd_hist,
            r.stoch_k,
            r.news_impact,
        ]

    cols = [
        "Symbol",
        "Signal",
        "Entry",
        "SL",
        "Target",
        "RRR",
        "RSI",
        "MACD_hist",
        "StochK",
        "News",
    ]
    table = [rowfmt(r) for r in buys + sells]
    df_out = pd.DataFrame(table, columns=cols)
    print("\n=== Picks (last 3 days) ===")
    print(df_out.to_string(index=False))

    # Save artifacts
    ts = now_ist().strftime("%Y%m%d_%H%M")
    df_all = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "name": r.name,
                "signal": r.strong,
                "reason": r.reason,
                "entry": r.entry,
                "stop": r.stop,
                "target": r.target,
                "rrr": r.rrr,
                "rsi": r.rsi,
                "macd_hist": r.macd_hist,
                "stoch_k": r.stoch_k,
                "news": r.news_impact,
                "news_links": " | ".join(r.news_links),
            }
            for r in results
        ]
    )
    df_all.to_csv(f"signals_last{args.days}d.csv", index=False)

    if args.make_report:
        lines = []
        lines.append(
            f"# NIFTY 50 Signals — last {args.days} days (IST: {now_ist().strftime('%Y-%m-%d %H:%M')})"
        )
        # Market context (optional PCR)
        pcr = get_nifty_pcr_quick()
        if pcr:
            ctx = (
                "Overbought/Caution"
                if pcr > 1.3
                else ("Oversold/Bounce-risk" if pcr < 0.7 else "Neutral")
            )
            lines.append(f"\n**NIFTY PCR** ≈ {pcr:.2f} → {ctx}")
        lines.append("\n## Strong Buy")
        for r in buys:
            lines.append(
                f"- **{r.symbol}** — {r.reason} | Entry ~ **{safe_round(r.entry,2)}**, SL **{safe_round(r.stop,2)}**, Target **{safe_round(r.target,2)}** (RRR {r.rrr}); RSI {r.rsi}, MACDh {r.macd_hist}. News: {r.news_impact}"
            )
        lines.append("\n## Strong Sell")
        for r in sells:
            lines.append(
                f"- **{r.symbol}** — {r.reason} | Short near **{safe_round(r.entry,2)}**, SL **{safe_round(r.stop,2)}**, Target **{safe_round(r.target,2)}** (RRR {r.rrr}); RSI {r.rsi}, MACDh {r.macd_hist}. News: {r.news_impact}"
            )
        lines.append("\n## Notes")
        lines.append(
            "- Entry levels use breakout/breakdown retest logic from prior 20-day swing & ATR-based SL/targets."
        )
        lines.append(
            "- Signals require **trend (50/200-DMA), volume, RSI>50 or <50, MACD histogram sign** to align."
        )
        lines.append(
            "- News is tagged **positive/negative/mixed** and whether price is **continuously increasing/decreasing** over the last 3 days."
        )
        with open(f"report_last{args.days}d.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\nSaved report_last{args.days}d.md and signals_last{args.days}d.csv")


if __name__ == "__main__":
    main()
