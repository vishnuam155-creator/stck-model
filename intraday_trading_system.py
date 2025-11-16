#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Intraday Trading System for Indian Stock Market

A systematic approach implementing:
1. Pre-Market Preparation (Stock Filtering & Screening)
2. Intraday Execution Toolkit (Technical Indicators)
3. High-Probability Trade Setups (ORB, Confluence)
4. Risk Management (1% Rule, RRR, Position Sizing)

Based on professional intraday trading framework for NSE/BSE stocks.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================== Configuration ============================== #

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# IST Timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Stock Universe - NIFTY 100 for high liquidity
NIFTY_100_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "AXISBANK.NS",
    "HCLTECH.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "BAJFINANCE.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
    "POWERGRID.NS",
    "WIPRO.NS",
    "ONGC.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "NESTLEIND.NS",
    "BAJAJFINSV.NS",
    "M&M.NS",
    "TATASTEEL.NS",
    "HEROMOTOCO.NS",
    "COALINDIA.NS",
    "LTIM.NS",
    "GRASIM.NS",
    "INDUSINDBK.NS",
    "DRREDDY.NS",
    "JSWSTEEL.NS",
    "HDFCLIFE.NS",
    "CIPLA.NS",
    "BRITANNIA.NS",
    "DIVISLAB.NS",
    "EICHERMOT.NS",
    "TECHM.NS",
    "BPCL.NS",
    "UPL.NS",
    "HINDALCO.NS",
    "APOLLOHOSP.NS",
    "BAJAJ-AUTO.NS",
    "TATAMOTORS.NS",
    "TATACONSUM.NS",
    "SHRIRAMFIN.NS",
    "TRENT.NS",
    "SIEMENS.NS",
    "VEDL.NS",
    "AMBUJACEM.NS",
    "DLF.NS",
    "DABUR.NS",
    "GODREJCP.NS",
    "HAVELLS.NS",
    "PIDILITIND.NS",
    "BERGEPAINT.NS",
    "ADANIGREEN.NS",
    "ADANITRANS.NS",
    "TORNTPHARM.NS",
    "INDIGO.NS",
    "ICICIPRULI.NS",
    "SBILIFE.NS",
    "CHOLAFIN.NS",
    "LUPIN.NS",
    "ATGL.NS",
    "MOTHERSON.NS",
    "MCDOWELL-N.NS",
    "BANDHANBNK.NS",
    "ICICIGI.NS",
    "INDUSTOWER.NS",
    "TATAPOWER.NS",
    "BOSCHLTD.NS",
    "LICI.NS",
    "MARICO.NS",
    "ABB.NS",
    "COLPAL.NS",
    "NAUKRI.NS",
    "ZOMATO.NS",
    "POLICYBZR.NS",
    "PAYTM.NS",
    "DMART.NS",
    "BAJAJHLDNG.NS",
    "GLAND.NS",
    "SBICARD.NS",
    "PGHH.NS",
    "HAL.NS",
    "BEL.NS",
    "CANBK.NS",
    "IOC.NS",
    "SAIL.NS",
    "PNB.NS",
    "BANKBARODA.NS",
    "RECLTD.NS",
    "PFC.NS",
    "NMDC.NS",
    "IDEA.NS",
]

# Risk Management Parameters
DEFAULT_CAPITAL = 100000  # ₹1 Lakh
MAX_RISK_PERCENT = 1.0  # 1% per trade
MIN_RRR = 2.0  # Minimum 1:2 Risk-Reward Ratio
MAX_POSITION_SIZE_PERCENT = 10.0  # Max 10% of capital per position

# Technical Indicator Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_TREND_SUPPORT = 40  # For trending markets

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

EMA_SHORT = 9
EMA_LONG = 21

BB_PERIOD = 20
BB_STD = 2

ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.0

VOLUME_BREAKOUT_MULTIPLIER = 3.0  # 3x average volume
MIN_DAILY_VOLUME = 1_000_000  # 1M shares minimum

# Pre-Market Gap Threshold
GAP_UP_THRESHOLD = 2.0  # 2% gap up
GAP_DOWN_THRESHOLD = -2.0  # 2% gap down

# Opening Range Breakout (ORB)
ORB_RANGE_MINUTES = 15  # First 15 minutes (9:15-9:30)


# ============================== Data Classes ============================== #


@dataclass
class TechnicalIndicators:
    """Container for all technical indicators"""

    vwap: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr: Optional[float] = None
    volume: Optional[float] = None
    avg_volume_20: Optional[float] = None

    # Price levels
    current_price: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None

    # ORB levels
    orb_high: Optional[float] = None
    orb_low: Optional[float] = None
    orb_midpoint: Optional[float] = None


@dataclass
class TradeSignal:
    """Complete trade signal with entry, exit, and risk parameters"""

    symbol: str
    signal_type: str  # "BUY" or "SELL"
    strategy: str  # "ORB", "CONFLUENCE", "VWAP_PULLBACK", etc.

    # Price levels
    entry_price: float
    stop_loss: float
    target_price: float
    current_price: float

    # Risk metrics
    risk_amount: float  # in ₹
    reward_amount: float  # in ₹
    rrr: float  # Risk-Reward Ratio

    # Position sizing
    position_size: int  # Number of shares
    capital_required: float  # Total capital needed

    # Technical indicators
    indicators: TechnicalIndicators

    # Signal strength
    confidence_score: float  # 0-100
    confluences: List[str] = field(default_factory=list)  # List of confirming factors

    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now(IST).strftime("%H:%M:%S")
    )

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.rrr is None or self.rrr == 0:
            if self.signal_type == "BUY":
                risk = abs(self.entry_price - self.stop_loss)
                reward = abs(self.target_price - self.entry_price)
            else:  # SELL
                risk = abs(self.stop_loss - self.entry_price)
                reward = abs(self.entry_price - self.target_price)

            self.rrr = round(reward / risk, 2) if risk > 0 else 0

        self.risk_amount = abs(self.entry_price - self.stop_loss) * self.position_size
        self.reward_amount = (
            abs(self.target_price - self.entry_price) * self.position_size
        )
        self.capital_required = self.entry_price * self.position_size


# ============================== Utility Functions ============================== #


def now_ist() -> datetime:
    """Get current time in IST"""
    return datetime.now(IST)


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:15 - 15:30 IST)"""
    now = now_ist()
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def safe_float(val: Any, default: float = np.nan) -> float:
    """Safely convert to float"""
    try:
        if isinstance(val, pd.Series):
            val = val.iloc[-1] if not val.empty else default
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError, IndexError):
        return default


# ============================== Phase 1: Pre-Market Preparation ============================== #


def fetch_intraday_data(
    symbol: str, interval: str = "5m", days: int = 5
) -> pd.DataFrame:
    """
    Fetch intraday data for a symbol

    Args:
        symbol: Stock symbol with .NS suffix
        interval: Data interval (5m, 15m, 1h)
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    try:
        df = yf.download(
            symbol,
            period=f"{days}d",
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            return pd.DataFrame()

        # Ensure proper column names
        df.columns = [
            col.title() if isinstance(col, str) else col for col in df.columns
        ]
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Calculate Average True Range"""
    if len(df) < period:
        return np.nan

    atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=period)
    return safe_float(atr_series.iloc[-1]) if atr_series is not None else np.nan


def filter_by_liquidity(symbols: List[str]) -> List[str]:
    """
    Filter stocks by liquidity (daily volume > threshold)

    Args:
        symbols: List of stock symbols

    Returns:
        Filtered list of liquid stocks
    """
    liquid_stocks = []

    logger.info(f"Filtering {len(symbols)} stocks by liquidity...")

    for symbol in tqdm(symbols, desc="Liquidity Filter"):
        try:
            # Get daily data for volume check
            daily = yf.download(symbol, period="5d", interval="1d", progress=False)

            if daily.empty:
                continue

            avg_volume = daily["Volume"].mean()

            if avg_volume >= MIN_DAILY_VOLUME:
                liquid_stocks.append(symbol)

        except Exception as e:
            logger.debug(f"Liquidity check failed for {symbol}: {e}")
            continue

    logger.info(f"✓ {len(liquid_stocks)} liquid stocks found")
    return liquid_stocks


def filter_by_volatility(
    symbols: List[str], min_atr_pct: float = 0.5, max_atr_pct: float = 5.0
) -> List[str]:
    """
    Filter stocks by optimal volatility (medium ATR)

    Args:
        symbols: List of stock symbols
        min_atr_pct: Minimum ATR as % of price
        max_atr_pct: Maximum ATR as % of price

    Returns:
        Filtered list with optimal volatility
    """
    optimal_stocks = []

    logger.info(f"Filtering {len(symbols)} stocks by volatility...")

    for symbol in tqdm(symbols, desc="Volatility Filter"):
        try:
            daily = yf.download(
                symbol, period="30d", interval="1d", progress=False, auto_adjust=True
            )

            if len(daily) < ATR_PERIOD:
                continue

            daily.columns = [col.title() for col in daily.columns]
            atr = calculate_atr(daily)
            current_price = safe_float(daily["Close"].iloc[-1])

            if np.isnan(atr) or current_price == 0:
                continue

            atr_pct = (atr / current_price) * 100

            if min_atr_pct <= atr_pct <= max_atr_pct:
                optimal_stocks.append(symbol)

        except Exception as e:
            logger.debug(f"Volatility check failed for {symbol}: {e}")
            continue

    logger.info(f"✓ {len(optimal_stocks)} stocks with optimal volatility found")
    return optimal_stocks


def detect_pre_market_gaps(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Detect stocks with significant pre-market gaps

    Args:
        symbols: List of stock symbols

    Returns:
        Dictionary of gapping stocks with gap percentage
    """
    gappers = {}

    logger.info("Scanning for pre-market gaps...")

    for symbol in tqdm(symbols, desc="Gap Scanner"):
        try:
            # Get last 2 days to compare today vs yesterday
            df = yf.download(symbol, period="2d", interval="1d", progress=False)

            if len(df) < 2:
                continue

            yesterday_close = df["Close"].iloc[-2]
            today_open = df["Open"].iloc[-1] if len(df) > 1 else df["Close"].iloc[-1]

            gap_pct = ((today_open - yesterday_close) / yesterday_close) * 100

            if abs(gap_pct) >= abs(GAP_UP_THRESHOLD):
                gappers[symbol] = {
                    "gap_pct": round(gap_pct, 2),
                    "yesterday_close": round(yesterday_close, 2),
                    "today_open": round(today_open, 2),
                    "direction": "UP" if gap_pct > 0 else "DOWN",
                }

        except Exception as e:
            logger.debug(f"Gap detection failed for {symbol}: {e}")
            continue

    logger.info(f"✓ {len(gappers)} gapping stocks found")
    return gappers


# ============================== Phase 2: Technical Analysis ============================== #


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)

    VWAP = Σ(Price × Volume) / Σ(Volume)
    """
    if df.empty or "Volume" not in df.columns:
        return pd.Series(dtype=float)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return vwap


def calculate_support_resistance(
    df: pd.DataFrame, lookback: int = 50
) -> Tuple[float, float]:
    """
    Calculate support and resistance levels using swing highs/lows

    Args:
        df: OHLCV DataFrame
        lookback: Number of periods to look back

    Returns:
        Tuple of (support, resistance)
    """
    if len(df) < lookback:
        lookback = len(df)

    recent_df = df.tail(lookback)

    # Simple method: use min/max of recent range
    support = recent_df["Low"].min()
    resistance = recent_df["High"].max()

    return support, resistance


def calculate_all_indicators(df: pd.DataFrame) -> TechnicalIndicators:
    """
    Calculate all technical indicators for a DataFrame

    Args:
        df: OHLCV DataFrame

    Returns:
        TechnicalIndicators object with all calculated values
    """
    if df.empty or len(df) < 30:
        return TechnicalIndicators()

    indicators = TechnicalIndicators()

    try:
        # Current price
        indicators.current_price = safe_float(df["Close"].iloc[-1])

        # VWAP
        vwap_series = calculate_vwap(df)
        indicators.vwap = (
            safe_float(vwap_series.iloc[-1]) if not vwap_series.empty else None
        )

        # RSI
        rsi_series = ta.rsi(df["Close"], length=RSI_PERIOD)
        indicators.rsi = (
            safe_float(rsi_series.iloc[-1]) if rsi_series is not None else None
        )

        # MACD
        macd = ta.macd(df["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is not None and not macd.empty:
            indicators.macd = safe_float(
                macd[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
            )
            indicators.macd_signal = safe_float(
                macd[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
            )
            indicators.macd_histogram = safe_float(
                macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"].iloc[-1]
            )

        # EMAs
        ema_short_series = ta.ema(df["Close"], length=EMA_SHORT)
        ema_long_series = ta.ema(df["Close"], length=EMA_LONG)
        indicators.ema_short = (
            safe_float(ema_short_series.iloc[-1])
            if ema_short_series is not None
            else None
        )
        indicators.ema_long = (
            safe_float(ema_long_series.iloc[-1])
            if ema_long_series is not None
            else None
        )

        # Bollinger Bands
        bb = ta.bbands(df["Close"], length=BB_PERIOD, std=BB_STD)
        if bb is not None and not bb.empty:
            indicators.bb_upper = safe_float(bb[f"BBU_{BB_PERIOD}_{BB_STD}.0"].iloc[-1])
            indicators.bb_middle = safe_float(
                bb[f"BBM_{BB_PERIOD}_{BB_STD}.0"].iloc[-1]
            )
            indicators.bb_lower = safe_float(bb[f"BBL_{BB_PERIOD}_{BB_STD}.0"].iloc[-1])

        # ATR
        indicators.atr = calculate_atr(df)

        # Volume
        indicators.volume = safe_float(df["Volume"].iloc[-1])
        indicators.avg_volume_20 = safe_float(df["Volume"].tail(20).mean())

        # Support & Resistance
        support, resistance = calculate_support_resistance(df)
        indicators.support = support
        indicators.resistance = resistance

    except Exception as e:
        logger.warning(f"Error calculating indicators: {e}")

    return indicators


def calculate_orb_levels(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Opening Range Breakout levels (first 15 minutes)

    Returns:
        Tuple of (orb_high, orb_low, orb_midpoint)
    """
    if df.empty:
        return None, None, None

    # Get today's data
    today = df[df.index.date == df.index[-1].date()]

    if today.empty or len(today) < 3:
        return None, None, None

    # First 15 minutes (approximately 3 candles of 5-min data)
    orb_range = today.head(3)

    orb_high = orb_range["High"].max()
    orb_low = orb_range["Low"].min()
    orb_midpoint = (orb_high + orb_low) / 2

    return orb_high, orb_low, orb_midpoint


# ============================== Phase 3: Signal Generation ============================== #


def generate_orb_signal(
    symbol: str, df: pd.DataFrame, indicators: TechnicalIndicators, capital: float
) -> Optional[TradeSignal]:
    """
    Generate Opening Range Breakout (ORB) signal

    Strategy:
    - BUY: Price breaks above ORB high with high volume
    - SELL: Price breaks below ORB low with high volume
    """
    orb_high, orb_low, orb_midpoint = calculate_orb_levels(df)

    if orb_high is None or indicators.current_price is None:
        return None

    indicators.orb_high = orb_high
    indicators.orb_low = orb_low
    indicators.orb_midpoint = orb_midpoint

    current_price = indicators.current_price
    volume = indicators.volume or 0
    avg_volume = indicators.avg_volume_20 or 1

    # Volume confirmation: 3-5x average
    volume_spike = volume >= (avg_volume * VOLUME_BREAKOUT_MULTIPLIER)

    signal = None

    # BUY Signal: Breakout above ORB high
    if current_price > orb_high and volume_spike:
        entry = current_price
        stop_loss = orb_midpoint  # Stop at ORB midpoint
        target = entry + (entry - stop_loss) * MIN_RRR

        # Position sizing
        risk_per_share = abs(entry - stop_loss)
        max_risk_amount = capital * (MAX_RISK_PERCENT / 100)
        position_size = (
            int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        )

        if position_size > 0:
            confluences = ["ORB_BREAKOUT_UP", "HIGH_VOLUME"]
            if indicators.rsi and indicators.rsi > 50:
                confluences.append("RSI_BULLISH")
            if indicators.vwap and current_price > indicators.vwap:
                confluences.append("ABOVE_VWAP")

            signal = TradeSignal(
                symbol=symbol.replace(".NS", ""),
                signal_type="BUY",
                strategy="ORB",
                entry_price=entry,
                stop_loss=stop_loss,
                target_price=target,
                current_price=current_price,
                risk_amount=0,
                reward_amount=0,
                rrr=0,
                position_size=position_size,
                capital_required=0,
                indicators=indicators,
                confidence_score=min(100, len(confluences) * 20),
                confluences=confluences,
            )

    # SELL Signal: Breakdown below ORB low
    elif current_price < orb_low and volume_spike:
        entry = current_price
        stop_loss = orb_midpoint  # Stop at ORB midpoint
        target = entry - (stop_loss - entry) * MIN_RRR

        # Position sizing
        risk_per_share = abs(stop_loss - entry)
        max_risk_amount = capital * (MAX_RISK_PERCENT / 100)
        position_size = (
            int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        )

        if position_size > 0:
            confluences = ["ORB_BREAKDOWN", "HIGH_VOLUME"]
            if indicators.rsi and indicators.rsi < 50:
                confluences.append("RSI_BEARISH")
            if indicators.vwap and current_price < indicators.vwap:
                confluences.append("BELOW_VWAP")

            signal = TradeSignal(
                symbol=symbol.replace(".NS", ""),
                signal_type="SELL",
                strategy="ORB",
                entry_price=entry,
                stop_loss=stop_loss,
                target_price=target,
                current_price=current_price,
                risk_amount=0,
                reward_amount=0,
                rrr=0,
                position_size=position_size,
                capital_required=0,
                indicators=indicators,
                confidence_score=min(100, len(confluences) * 20),
                confluences=confluences,
            )

    return signal


def generate_confluence_signal(
    symbol: str, df: pd.DataFrame, indicators: TechnicalIndicators, capital: float
) -> Optional[TradeSignal]:
    """
    Generate Confluence Reversal signal at Support/Resistance

    Strategy:
    - BUY: Price at support + RSI oversold + bullish MACD + above VWAP trend
    - SELL: Price at resistance + RSI overbought + bearish MACD + below VWAP trend
    """
    if not all(
        [
            indicators.current_price,
            indicators.support,
            indicators.resistance,
            indicators.rsi,
            indicators.vwap,
        ]
    ):
        return None

    current_price = indicators.current_price
    support = indicators.support
    resistance = indicators.resistance

    # Allow 1% tolerance for support/resistance touch
    tolerance = current_price * 0.01

    signal = None
    confluences = []

    # BUY Signal: Confluence at Support
    at_support = abs(current_price - support) <= tolerance

    if at_support:
        # Check for bullish confluences
        if indicators.rsi and indicators.rsi < RSI_OVERSOLD:
            confluences.append("RSI_OVERSOLD")

        if indicators.macd_histogram and indicators.macd_histogram > 0:
            confluences.append("MACD_BULLISH")

        if indicators.vwap and current_price > indicators.vwap:
            confluences.append("ABOVE_VWAP")

        if (
            indicators.ema_short
            and indicators.ema_long
            and indicators.ema_short > indicators.ema_long
        ):
            confluences.append("EMA_CROSSOVER_BULL")

        # Need at least 3 confluences for high probability
        if len(confluences) >= 3:
            entry = current_price

            # Use ATR for stop-loss if available
            if indicators.atr and not np.isnan(indicators.atr):
                stop_loss = entry - (indicators.atr * ATR_STOP_MULTIPLIER)
                target = entry + (indicators.atr * ATR_TARGET_MULTIPLIER)
            else:
                stop_loss = support * 0.98  # 2% below support
                risk = entry - stop_loss
                target = entry + (risk * MIN_RRR)

            # Position sizing
            risk_per_share = abs(entry - stop_loss)
            max_risk_amount = capital * (MAX_RISK_PERCENT / 100)
            position_size = (
                int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
            )

            if position_size > 0:
                signal = TradeSignal(
                    symbol=symbol.replace(".NS", ""),
                    signal_type="BUY",
                    strategy="CONFLUENCE_REVERSAL",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    target_price=target,
                    current_price=current_price,
                    risk_amount=0,
                    reward_amount=0,
                    rrr=0,
                    position_size=position_size,
                    capital_required=0,
                    indicators=indicators,
                    confidence_score=min(100, len(confluences) * 20),
                    confluences=["AT_SUPPORT"] + confluences,
                )

    # SELL Signal: Confluence at Resistance
    at_resistance = abs(current_price - resistance) <= tolerance

    if at_resistance and not signal:  # Only if no buy signal
        confluences = []

        # Check for bearish confluences
        if indicators.rsi and indicators.rsi > RSI_OVERBOUGHT:
            confluences.append("RSI_OVERBOUGHT")

        if indicators.macd_histogram and indicators.macd_histogram < 0:
            confluences.append("MACD_BEARISH")

        if indicators.vwap and current_price < indicators.vwap:
            confluences.append("BELOW_VWAP")

        if (
            indicators.ema_short
            and indicators.ema_long
            and indicators.ema_short < indicators.ema_long
        ):
            confluences.append("EMA_CROSSOVER_BEAR")

        # Need at least 3 confluences
        if len(confluences) >= 3:
            entry = current_price

            # Use ATR for stop-loss if available
            if indicators.atr and not np.isnan(indicators.atr):
                stop_loss = entry + (indicators.atr * ATR_STOP_MULTIPLIER)
                target = entry - (indicators.atr * ATR_TARGET_MULTIPLIER)
            else:
                stop_loss = resistance * 1.02  # 2% above resistance
                risk = stop_loss - entry
                target = entry - (risk * MIN_RRR)

            # Position sizing
            risk_per_share = abs(stop_loss - entry)
            max_risk_amount = capital * (MAX_RISK_PERCENT / 100)
            position_size = (
                int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
            )

            if position_size > 0:
                signal = TradeSignal(
                    symbol=symbol.replace(".NS", ""),
                    signal_type="SELL",
                    strategy="CONFLUENCE_REVERSAL",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    target_price=target,
                    current_price=current_price,
                    risk_amount=0,
                    reward_amount=0,
                    rrr=0,
                    position_size=position_size,
                    capital_required=0,
                    indicators=indicators,
                    confidence_score=min(100, len(confluences) * 20),
                    confluences=["AT_RESISTANCE"] + confluences,
                )

    return signal


def generate_vwap_pullback_signal(
    symbol: str, df: pd.DataFrame, indicators: TechnicalIndicators, capital: float
) -> Optional[TradeSignal]:
    """
    Generate VWAP Pullback signal

    Strategy:
    - BUY: Strong uptrend (above VWAP), price pulls back to VWAP (dynamic support)
    - SELL: Strong downtrend (below VWAP), price bounces to VWAP (dynamic resistance)
    """
    if not all([indicators.current_price, indicators.vwap]):
        return None

    current_price = indicators.current_price
    vwap = indicators.vwap

    # Calculate if price is near VWAP (within 0.5%)
    tolerance = current_price * 0.005
    near_vwap = abs(current_price - vwap) <= tolerance

    if not near_vwap:
        return None

    signal = None

    # Determine trend direction using recent price action
    if len(df) >= 10:
        recent_closes = df["Close"].tail(10)
        trend_up = recent_closes.iloc[-1] > recent_closes.iloc[0]

        if trend_up and current_price >= vwap:
            # Bullish VWAP pullback
            confluences = ["VWAP_PULLBACK", "UPTREND"]

            if indicators.rsi and indicators.rsi > RSI_TREND_SUPPORT:
                confluences.append("RSI_BULLISH")

            if indicators.macd_histogram and indicators.macd_histogram > 0:
                confluences.append("MACD_BULLISH")

            entry = current_price
            stop_loss = vwap * 0.99  # 1% below VWAP
            risk = entry - stop_loss
            target = entry + (risk * MIN_RRR)

            # Position sizing
            risk_per_share = abs(entry - stop_loss)
            max_risk_amount = capital * (MAX_RISK_PERCENT / 100)
            position_size = (
                int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
            )

            if position_size > 0:
                signal = TradeSignal(
                    symbol=symbol.replace(".NS", ""),
                    signal_type="BUY",
                    strategy="VWAP_PULLBACK",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    target_price=target,
                    current_price=current_price,
                    risk_amount=0,
                    reward_amount=0,
                    rrr=0,
                    position_size=position_size,
                    capital_required=0,
                    indicators=indicators,
                    confidence_score=min(100, len(confluences) * 20),
                    confluences=confluences,
                )

        elif not trend_up and current_price <= vwap:
            # Bearish VWAP bounce
            confluences = ["VWAP_BOUNCE", "DOWNTREND"]

            if indicators.rsi and indicators.rsi < (100 - RSI_TREND_SUPPORT):
                confluences.append("RSI_BEARISH")

            if indicators.macd_histogram and indicators.macd_histogram < 0:
                confluences.append("MACD_BEARISH")

            entry = current_price
            stop_loss = vwap * 1.01  # 1% above VWAP
            risk = stop_loss - entry
            target = entry - (risk * MIN_RRR)

            # Position sizing
            risk_per_share = abs(stop_loss - entry)
            max_risk_amount = capital * (MAX_RISK_PERCENT / 100)
            position_size = (
                int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
            )

            if position_size > 0:
                signal = TradeSignal(
                    symbol=symbol.replace(".NS", ""),
                    signal_type="SELL",
                    strategy="VWAP_PULLBACK",
                    entry_price=entry,
                    stop_loss=stop_loss,
                    target_price=target,
                    current_price=current_price,
                    risk_amount=0,
                    reward_amount=0,
                    rrr=0,
                    position_size=position_size,
                    capital_required=0,
                    indicators=indicators,
                    confidence_score=min(100, len(confluences) * 20),
                    confluences=confluences,
                )

    return signal


def scan_for_signals(symbols: List[str], capital: float) -> List[TradeSignal]:
    """
    Scan all symbols and generate trade signals

    Args:
        symbols: List of stock symbols to scan
        capital: Trading capital for position sizing

    Returns:
        List of TradeSignal objects
    """
    all_signals = []

    logger.info(f"Scanning {len(symbols)} stocks for trade signals...")

    for symbol in tqdm(symbols, desc="Signal Scanner"):
        try:
            # Fetch 5-minute intraday data
            df = fetch_intraday_data(symbol, interval="5m", days=5)

            if df.empty or len(df) < 30:
                continue

            # Ensure column names are correct
            df.columns = [
                col.title() if isinstance(col, str) else col for col in df.columns
            ]

            # Calculate all indicators
            indicators = calculate_all_indicators(df)

            if indicators.current_price is None:
                continue

            # Try all strategies
            strategies = [
                generate_orb_signal,
                generate_confluence_signal,
                generate_vwap_pullback_signal,
            ]

            for strategy_func in strategies:
                signal = strategy_func(symbol, df, indicators, capital)

                if signal and signal.rrr >= MIN_RRR:
                    all_signals.append(signal)
                    break  # Only one signal per stock

        except Exception as e:
            logger.debug(f"Signal generation failed for {symbol}: {e}")
            continue

    logger.info(f"✓ {len(all_signals)} trade signals generated")
    return all_signals


# ============================== Output & Reporting ============================== #


def signals_to_dataframe(signals: List[TradeSignal]) -> pd.DataFrame:
    """Convert list of signals to pandas DataFrame"""
    if not signals:
        return pd.DataFrame()

    data = []
    for sig in signals:
        row = {
            "Symbol": sig.symbol,
            "Signal": sig.signal_type,
            "Strategy": sig.strategy,
            "Entry": round(sig.entry_price, 2),
            "Stop Loss": round(sig.stop_loss, 2),
            "Target": round(sig.target_price, 2),
            "Current": round(sig.current_price, 2),
            "RRR": sig.rrr,
            "Position Size": sig.position_size,
            "Capital Required": f"₹{sig.capital_required:,.0f}",
            "Risk": f"₹{sig.risk_amount:,.0f}",
            "Reward": f"₹{sig.reward_amount:,.0f}",
            "Confidence": f"{sig.confidence_score}%",
            "RSI": round(sig.indicators.rsi, 1) if sig.indicators.rsi else "-",
            "VWAP": round(sig.indicators.vwap, 2) if sig.indicators.vwap else "-",
            "ATR": round(sig.indicators.atr, 2) if sig.indicators.atr else "-",
            "Volume vs Avg": (
                f"{(sig.indicators.volume / sig.indicators.avg_volume_20):.1f}x"
                if sig.indicators.avg_volume_20
                else "-"
            ),
            "Confluences": ", ".join(sig.confluences),
            "Time": sig.timestamp,
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by confidence score and RRR
    df = df.sort_values(["Confidence", "RRR"], ascending=[False, False])

    return df


def print_summary(
    buy_signals: List[TradeSignal], sell_signals: List[TradeSignal], capital: float
):
    """Print executive summary"""
    print("\n" + "=" * 100)
    print("INTRADAY TRADING SYSTEM - EXECUTIVE SUMMARY".center(100))
    print("=" * 100)

    print(f"\n📊 Trading Capital: ₹{capital:,.0f}")
    print(
        f"⚠️  Max Risk per Trade: ₹{capital * (MAX_RISK_PERCENT/100):,.0f} ({MAX_RISK_PERCENT}%)"
    )
    print(f"🎯 Minimum RRR Required: 1:{MIN_RRR}")

    print(f"\n\n{'BUY SIGNALS':-^100}")
    if buy_signals:
        buy_df = signals_to_dataframe(buy_signals)
        print(buy_df.to_string(index=False))

        total_capital_needed = sum(s.capital_required for s in buy_signals)
        total_potential_profit = sum(s.reward_amount for s in buy_signals)
        print(f"\n💰 Total Capital Needed (All Buys): ₹{total_capital_needed:,.0f}")
        print(f"💵 Total Potential Profit: ₹{total_potential_profit:,.0f}")
    else:
        print("No BUY signals found")

    print(f"\n\n{'SELL SIGNALS':-^100}")
    if sell_signals:
        sell_df = signals_to_dataframe(sell_signals)
        print(sell_df.to_string(index=False))

        total_capital_needed = sum(s.capital_required for s in sell_signals)
        total_potential_profit = sum(s.reward_amount for s in sell_signals)
        print(f"\n💰 Total Capital Needed (All Sells): ₹{total_capital_needed:,.0f}")
        print(f"💵 Total Potential Profit: ₹{total_potential_profit:,.0f}")
    else:
        print("No SELL signals found")

    print("\n" + "=" * 100)
    print(f"⏰ Report Generated: {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 100 + "\n")


def save_results(signals: List[TradeSignal], output_file: str = "intraday_signals.csv"):
    """Save signals to CSV file"""
    if not signals:
        logger.warning("No signals to save")
        return

    df = signals_to_dataframe(signals)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Results saved to {output_file}")


# ============================== Main Execution ============================== #


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Intraday Trading System for Indian Stock Market"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help=f"Trading capital in ₹ (default: {DEFAULT_CAPITAL})",
    )
    parser.add_argument(
        "--stocks", type=int, default=30, help="Number of stocks to scan (default: 30)"
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Skip liquidity and volatility filters (faster but less optimal)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="intraday_signals.csv",
        help="Output CSV file name",
    )

    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("INTRADAY TRADING SYSTEM - INDIAN STOCK MARKET".center(100))
    print("=" * 100)
    print(f"\n🕒 Current Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"💼 Trading Capital: ₹{args.capital:,.0f}")
    print(f"📈 Scanning: {args.stocks} stocks from NIFTY 100")

    # Phase 1: Pre-Market Preparation
    print(f"\n{'='*100}")
    print("PHASE 1: PRE-MARKET PREPARATION".center(100))
    print("=" * 100)

    # Select stocks
    universe = NIFTY_100_SYMBOLS[: args.stocks]

    if not args.skip_filters:
        # Filter by liquidity
        liquid_stocks = filter_by_liquidity(universe)

        # Filter by volatility
        optimal_stocks = filter_by_volatility(liquid_stocks)

        # Detect gaps
        gappers = detect_pre_market_gaps(optimal_stocks)

        if gappers:
            print(f"\n📊 Pre-Market Gappers:")
            for symbol, data in list(gappers.items())[:10]:
                print(
                    f"  {symbol.replace('.NS', '')}: {data['direction']} {data['gap_pct']:.2f}%"
                )

        scan_symbols = optimal_stocks
    else:
        scan_symbols = universe
        logger.info("⚡ Skipping filters for faster execution")

    # Phase 2 & 3: Technical Analysis and Signal Generation
    print(f"\n{'='*100}")
    print("PHASE 2 & 3: TECHNICAL ANALYSIS & SIGNAL GENERATION".center(100))
    print("=" * 100)

    all_signals = scan_for_signals(scan_symbols, args.capital)

    # Separate buy and sell signals
    buy_signals = [s for s in all_signals if s.signal_type == "BUY"]
    sell_signals = [s for s in all_signals if s.signal_type == "SELL"]

    # Phase 4: Results and Risk Management
    print(f"\n{'='*100}")
    print("PHASE 4: RESULTS & RISK MANAGEMENT".center(100))
    print("=" * 100)

    print_summary(buy_signals, sell_signals, args.capital)

    # Save results
    if all_signals:
        save_results(all_signals, args.output)

    # Trading recommendations
    print("\n📋 TRADING RECOMMENDATIONS:")
    print("1. Never risk more than 1% of capital per trade")
    print("2. Always wait for volume confirmation on breakouts")
    print("3. Set stop-loss BEFORE entering the trade")
    print("4. Only take trades with RRR >= 1:2")
    print("5. Monitor live price action on 5-minute charts")
    print("6. Exit immediately if stop-loss is hit - NO EXCEPTIONS")
    print("7. Consider partial profit booking at 1:1 RRR")
    print("8. Trail stop-loss to breakeven after 1:1 RRR achieved")

    print("\n⚠️  DISCLAIMER:")
    print("This is an educational tool. Always do your own analysis before trading.")
    print("Past performance does not guarantee future results.")
    print("Trading involves substantial risk of loss.\n")


if __name__ == "__main__":
    main()
