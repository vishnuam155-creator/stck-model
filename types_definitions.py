#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type definitions for the stock analysis system.

This module provides comprehensive type hints using TypedDict, Protocol,
and standard typing constructs to ensure type safety across the codebase.
"""
from typing import TypedDict, Protocol, Optional, List, Dict, Any, Literal, Union, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from numpy.typing import NDArray


# ============================== Enums and Literals ============================== #

SignalType = Literal["Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell", "Watch", "NoData", "Error"]
SentimentType = Literal["positive", "negative", "neutral"]
TrendType = Literal["uptrend", "downtrend", "sideways", "unknown", "flat",
                    "continuously_increasing", "continuously_decreasing"]
SentimentEngine = Literal["finbert", "openai", "nltk"]


# ============================== News Types ============================== #

class NewsArticle(TypedDict, total=False):
    """Represents a news article with metadata."""
    title: str
    desc: Optional[str]
    url: Optional[str]
    source: Optional[str]
    publishedAt: Optional[str]


class SentimentResult(TypedDict):
    """Sentiment analysis result for a piece of text."""
    headline: str
    sentiment: SentimentType
    impact: float  # -2.0 to +2.0
    reason: str


class SentimentDetail(TypedDict):
    """Aggregated sentiment details."""
    pos: int
    neg: int
    neu: int
    top: List[SentimentResult]


# ============================== Technical Indicators ============================== #

class TechnicalIndicators(TypedDict, total=False):
    """Collection of technical indicator values."""
    # Price-based
    Close: float
    Open: float
    High: float
    Low: float
    Volume: float

    # Moving Averages
    SMA50: float
    SMA200: float
    EMA20: float
    EMA50: float

    # Oscillators
    RSI: float
    RSI14: float

    # MACD
    MACD: float
    MACDS: float  # MACD Signal
    MACDhist: float  # MACD Histogram

    # Stochastic
    STOCHK: float
    STOCHD: float

    # Volume
    VolMA20: float

    # Trend Detection
    HH: bool  # Higher High
    golden_cross: bool
    death_cross: bool


class SwingLevels(TypedDict):
    """Support and resistance levels from swing analysis."""
    resistance: float
    support: float
    swing_high: float
    swing_low: float


# ============================== Signal Types ============================== #

class EntryExitLevels(TypedDict, total=False):
    """Entry, stop loss, and target levels for a trade setup."""
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    rrr: Optional[float]  # Risk-Reward Ratio
    setup_note: str


class SignalData(TypedDict, total=False):
    """Complete signal data for a stock."""
    symbol: str
    name: str
    strong: SignalType
    signal: SignalType
    reason: str
    score: int

    # Entry/Exit
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    rrr: Optional[float]

    # Technical Indicators
    rsi: Optional[float]
    macd_hist: Optional[float]
    macd_vs_signal: Optional[float]
    stoch_k: Optional[float]

    # Trend Analysis
    trend: TrendType
    golden_cross: bool
    death_cross: bool
    above_50dma: Optional[bool]
    above_200dma: Optional[bool]

    # Sentiment
    sentiment_score: float
    sentiment_counts: SentimentDetail
    news_impact: str
    news_links: List[str]
    top_news: List[SentimentResult]

    # Metadata
    time: str
    close: float
    setup_note: str
    error: Optional[str]


# ============================== Market Data Types ============================== #

class OHLCVData(TypedDict):
    """OHLCV (Open, High, Low, Close, Volume) data point."""
    timestamp: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class MarketBreadth(TypedDict):
    """Market breadth indicators."""
    pcr: Optional[float]  # Put-Call Ratio
    context: str  # Interpretation


class DailyMAFlags(TypedDict):
    """Daily moving average positioning flags."""
    above_50dma: Optional[bool]
    above_200dma: Optional[bool]


# ============================== Configuration Types ============================== #

class APIConfig(TypedDict, total=False):
    """API configuration settings."""
    newsapi_key: Optional[str]
    openai_api_key: Optional[str]
    api_timeout: int
    api_retries: int


class ScannerConfig(TypedDict, total=False):
    """Scanner configuration parameters."""
    days_lookback: int
    max_stocks: int
    interval: str  # "15m", "1h", "1d", etc.
    sentiment_engine: SentimentEngine
    openai_model: str
    prefer_finbert: bool
    make_report: bool
    make_charts: bool
    cache_dir: str
    output_csv: str
    output_md: str


class IndicatorConfig(TypedDict):
    """Technical indicator calculation parameters."""
    rsi_length: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    ema_short: int
    ema_long: int
    sma_medium: int
    sma_long: int
    volume_ma: int
    atr_length: int
    stoch_k: int
    stoch_d: int
    stoch_smooth: int
    swing_lookback: int
    breakout_lookback: int


# ============================== Protocols ============================== #

class SentimentAnalyzer(Protocol):
    """Protocol for sentiment analysis engines."""

    def classify_headlines(
        self,
        headlines: List[str],
        **kwargs: Any
    ) -> List[SentimentResult]:
        """
        Classify sentiment of headlines.

        Args:
            headlines: List of headline strings
            **kwargs: Additional engine-specific parameters

        Returns:
            List of sentiment results
        """
        ...

    def aggregate_sentiment(
        self,
        results: List[SentimentResult]
    ) -> tuple[float, SentimentDetail]:
        """
        Aggregate individual sentiment results.

        Args:
            results: List of sentiment results

        Returns:
            Tuple of (overall_score, detailed_counts)
        """
        ...


class NewsProvider(Protocol):
    """Protocol for news data providers."""

    def fetch_news(
        self,
        query: str,
        since: datetime,
        limit: int,
        **kwargs: Any
    ) -> List[NewsArticle]:
        """
        Fetch news articles matching query.

        Args:
            query: Search query string
            since: Fetch articles since this datetime
            limit: Maximum number of articles
            **kwargs: Provider-specific parameters

        Returns:
            List of news articles
        """
        ...


class TechnicalAnalyzer(Protocol):
    """Protocol for technical analysis engines."""

    def compute_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute technical indicators on OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        ...

    def detect_signals(
        self,
        df: pd.DataFrame,
        sentiment_score: float
    ) -> tuple[SignalType, int]:
        """
        Detect trading signals from indicators.

        Args:
            df: DataFrame with indicators
            sentiment_score: Sentiment score to incorporate

        Returns:
            Tuple of (signal_label, signal_score)
        """
        ...


# ============================== Result Types ============================== #

class AnalysisResult(TypedDict, total=False):
    """Complete analysis result for a symbol."""
    symbol: str
    success: bool
    error: Optional[str]
    signal_data: Optional[SignalData]
    raw_data: Optional[pd.DataFrame]
    execution_time_ms: float


class ScanResults(TypedDict):
    """Results from scanning multiple symbols."""
    scan_time: datetime
    total_symbols: int
    successful: int
    failed: int
    signals: List[SignalData]
    errors: List[Dict[str, str]]
    market_breadth: Optional[MarketBreadth]


# ============================== Cache Types ============================== #

class CacheEntry(TypedDict):
    """Cache entry structure."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float]


# ============================== Utility Types ============================== #

NumericValue = Union[int, float, np.number, np.floating, np.integer]
NumericArray = Union[List[NumericValue], NDArray[np.floating], pd.Series]
TimeSeriesData = pd.DataFrame
OptionalFloat = Optional[float]
OptionalInt = Optional[int]
OptionalStr = Optional[str]


# ============================== Function Signatures ============================== #

# Type aliases for common function signatures
NewsFilter = Callable[[NewsArticle], bool]
DataTransform = Callable[[pd.DataFrame], pd.DataFrame]
SignalFilter = Callable[[SignalData], bool]
PriceFormatter = Callable[[OptionalFloat], str]


# ============================== Validation Types ============================== #

class ValidationResult(TypedDict):
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataQuality(TypedDict):
    """Data quality metrics."""
    completeness: float  # 0.0 to 1.0
    missing_fields: List[str]
    invalid_values: Dict[str, int]
    data_age_hours: float


# ============================== Export Types ============================== #

__all__ = [
    # Literals
    "SignalType",
    "SentimentType",
    "TrendType",
    "SentimentEngine",

    # News Types
    "NewsArticle",
    "SentimentResult",
    "SentimentDetail",

    # Technical Types
    "TechnicalIndicators",
    "SwingLevels",

    # Signal Types
    "EntryExitLevels",
    "SignalData",

    # Market Data
    "OHLCVData",
    "MarketBreadth",
    "DailyMAFlags",

    # Configuration
    "APIConfig",
    "ScannerConfig",
    "IndicatorConfig",

    # Protocols
    "SentimentAnalyzer",
    "NewsProvider",
    "TechnicalAnalyzer",

    # Results
    "AnalysisResult",
    "ScanResults",

    # Cache
    "CacheEntry",

    # Utilities
    "NumericValue",
    "NumericArray",
    "TimeSeriesData",
    "OptionalFloat",
    "OptionalInt",
    "OptionalStr",
    "NewsFilter",
    "DataTransform",
    "SignalFilter",
    "PriceFormatter",

    # Validation
    "ValidationResult",
    "DataQuality",
]
