#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for the stock analysis system.

This module centralizes all configuration constants, default values,
and settings to avoid magic numbers and improve maintainability.
"""
import os
from typing import Final, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================== API Configuration ============================== #


@dataclass(frozen=True)
class APISettings:
    """API keys and endpoints configuration."""

    # API Keys (from environment)
    NEWSAPI_KEY: Optional[str] = field(default_factory=lambda: os.getenv("NEWSAPI_KEY"))
    OPENAI_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    NEWS_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY")
    )

    # API Endpoints
    NSE_NIFTY50_CSV: str = (
        "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
    )
    NSE_OPTION_CHAIN_NIFTY: str = (
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    )
    NEWSAPI_ENDPOINT: str = "https://newsapi.org/v2/everything"
    OPENAI_ENDPOINT: str = "https://api.openai.com/v1/chat/completions"
    GDELT_API_ENDPOINT: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    GOOGLE_NEWS_RSS: str = "https://news.google.com/rss/search"

    # API Timeouts (seconds)
    DEFAULT_TIMEOUT: int = 20
    NEWS_TIMEOUT: int = 20
    OPENAI_TIMEOUT: int = 60
    NSE_TIMEOUT: int = 15

    # API Retry Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0


# ============================== Data Configuration ============================== #


@dataclass(frozen=True)
class DataSettings:
    """Data fetching and processing configuration."""

    # Time periods
    DEFAULT_DAYS_LOOKBACK: int = 3
    PRICE_HISTORY_DAYS: int = 400  # For daily MAs
    MAX_HISTORICAL_YEARS: int = 2

    # Data intervals
    INTRADAY_INTERVAL: str = "15m"
    DAILY_INTERVAL: str = "1d"
    INTRADAY_PERIOD: str = "5d"  # yfinance requires larger period for 15m

    # Data quality thresholds
    MIN_BARS_FOR_ANALYSIS: int = 50
    MIN_BARS_FOR_DAILY: int = 200
    MIN_NEWS_ARTICLES: int = 3

    # News settings
    DEFAULT_NEWS_LIMIT: int = 12
    MAX_NEWS_ARTICLES: int = 50
    NEWS_SNIPPET_LENGTH: int = 200


# ============================== Technical Indicators Configuration ============================== #


@dataclass(frozen=True)
class IndicatorSettings:
    """Technical indicator calculation parameters."""

    # Moving Averages
    SMA_SHORT: int = 50
    SMA_LONG: int = 200
    EMA_SHORT: int = 20
    EMA_LONG: int = 50

    # RSI
    RSI_LENGTH: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_BULLISH_THRESHOLD: float = 50.0
    RSI_BEARISH_THRESHOLD: float = 50.0

    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # Stochastic
    STOCH_K: int = 14
    STOCH_D: int = 3
    STOCH_SMOOTH: int = 3

    # ATR (Average True Range)
    ATR_LENGTH: int = 14
    ATR_STOP_MULTIPLIER: float = 1.5
    ATR_TARGET_MULTIPLIER: float = 2.0

    # Volume
    VOLUME_MA_LENGTH: int = 20

    # Swing Detection
    SWING_LOOKBACK: int = 5
    SWING_WINDOW: int = 30
    BREAKOUT_LOOKBACK: int = 20

    # Volume confirmation threshold (as % of moving average)
    VOLUME_THRESHOLD: float = 1.0


# ============================== Signal Generation Configuration ============================== #


@dataclass(frozen=True)
class SignalSettings:
    """Signal generation and scoring parameters."""

    # Signal scoring thresholds
    STRONG_BUY_SCORE: int = 4
    BUY_SCORE: int = 2
    NEUTRAL_SCORE: int = 0
    SELL_SCORE: int = -2
    STRONG_SELL_SCORE: int = -4

    # Trend confirmation
    TREND_CONFIRMATION_BARS: int = 3
    MIN_TREND_MOVE_PCT: float = 1.5

    # Risk management
    MIN_RRR: float = 1.5  # Minimum Risk-Reward Ratio
    DEFAULT_RISK_PCT: float = 0.02  # 2% of entry
    MAX_RISK_PCT: float = 0.05  # 5% of entry

    # Entry/Exit buffer
    ENTRY_BUFFER_PCT: float = 0.005  # 0.5%
    BREAKOUT_CONFIRMATION_PCT: float = 0.01  # 1%

    # Position sizing
    DEFAULT_POSITION_SIZE_PCT: float = 0.10  # 10% of portfolio


# ============================== Sentiment Configuration ============================== #


@dataclass(frozen=True)
class SentimentSettings:
    """Sentiment analysis configuration."""

    # Sentiment engines
    DEFAULT_ENGINE: str = "finbert"
    FALLBACK_ENGINE: str = "nltk"

    # FinBERT settings
    FINBERT_MODEL: str = "ProsusAI/finbert"
    FINBERT_BATCH_SIZE: int = 16
    FINBERT_MAX_LENGTH: int = 128

    # OpenAI settings
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    OPENAI_MAX_TOKENS: Optional[int] = None

    # Sentiment scoring
    SENTIMENT_IMPACT_MIN: float = -2.0
    SENTIMENT_IMPACT_MAX: float = 2.0
    SENTIMENT_AGGREGATE_MIN: float = -4.0
    SENTIMENT_AGGREGATE_MAX: float = 4.0

    # Sentiment thresholds
    POSITIVE_THRESHOLD: float = 0.2
    NEGATIVE_THRESHOLD: float = -0.2
    STRONG_POSITIVE_THRESHOLD: float = 1.0
    STRONG_NEGATIVE_THRESHOLD: float = -1.0

    # Text processing
    MAX_HEADLINE_LENGTH: int = 512
    CACHE_TTL_DAYS: int = 7


# ============================== NIFTY 50 Symbols ============================== #


@dataclass(frozen=True)
class SymbolSettings:
    """Stock symbol lists and mappings."""

    # Full NIFTY 50 list with Yahoo Finance suffix
    NIFTY50_SYMBOLS: tuple[str, ...] = (
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
    )

    # Symbol suffix for Yahoo Finance
    YF_SUFFIX: str = ".NS"

    # Default test symbols (for faster development/testing)
    TEST_SYMBOLS: tuple[str, ...] = (
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "ITC.NS",
        "LT.NS",
        "SBIN.NS",
    )


# ============================== File and Path Configuration ============================== #


@dataclass
class PathSettings:
    """File paths and directory configuration."""

    # Cache directory
    CACHE_DIR: str = ".cache"
    SENTIMENT_CACHE_FILE: str = "sentiments.json"
    PRICE_CACHE_FILE: str = "prices.json"

    # Output directories
    OUTPUT_DIR: str = "output"
    CHARTS_DIR: str = "charts"
    REPORTS_DIR: str = "reports"

    # Output file names
    DEFAULT_CSV_NAME: str = "results.csv"
    DEFAULT_MD_NAME: str = "results.md"
    SIGNAL_CSV_TEMPLATE: str = "signals_last{days}d.csv"
    REPORT_MD_TEMPLATE: str = "report_last{days}d.md"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for directory in [
            self.CACHE_DIR,
            self.OUTPUT_DIR,
            self.CHARTS_DIR,
            self.REPORTS_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    def get_sentiment_cache_path(self) -> str:
        """Get full path to sentiment cache file."""
        return os.path.join(self.CACHE_DIR, self.SENTIMENT_CACHE_FILE)

    def get_price_cache_path(self) -> str:
        """Get full path to price cache file."""
        return os.path.join(self.CACHE_DIR, self.PRICE_CACHE_FILE)

    def get_output_csv_path(self, filename: Optional[str] = None) -> str:
        """Get full path to output CSV file."""
        return os.path.join(self.OUTPUT_DIR, filename or self.DEFAULT_CSV_NAME)

    def get_output_md_path(self, filename: Optional[str] = None) -> str:
        """Get full path to output Markdown file."""
        return os.path.join(self.REPORTS_DIR, filename or self.DEFAULT_MD_NAME)

    def get_chart_path(self, symbol: str) -> str:
        """Get full path to chart file for a symbol."""
        clean_symbol = symbol.replace(".NS", "")
        return os.path.join(self.CHARTS_DIR, f"{clean_symbol}_15m.html")


# ============================== Scanner Configuration ============================== #


@dataclass
class ScannerSettings:
    """Scanner runtime configuration."""

    # Scan limits
    max_stocks: int = 50
    days_lookback: int = 3
    parallel_requests: int = 5  # For concurrent API calls

    # Feature flags
    make_report: bool = True
    make_charts: bool = False
    use_cache: bool = True
    verbose: bool = False

    # Sentiment engine
    sentiment_engine: str = "finbert"
    prefer_finbert: bool = True

    # Output format
    output_csv: Optional[str] = None
    output_md: Optional[str] = None


# ============================== Market Context Configuration ============================== #


@dataclass(frozen=True)
class MarketSettings:
    """Market context and breadth indicators configuration."""

    # PCR (Put-Call Ratio) thresholds
    PCR_OVERBOUGHT: float = 1.3
    PCR_OVERSOLD: float = 0.7

    # Market breadth
    ADVANCE_DECLINE_THRESHOLD: float = 0.6  # 60% advance/decline ratio

    # Trading hours (IST)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30

    # Timezone
    IST_OFFSET_HOURS: float = 5.5


# ============================== Logging Configuration ============================== #


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_FILE: Optional[str] = None  # Set to enable file logging


# ============================== Global Configuration Instance ============================== #


class Config:
    """
    Global configuration singleton.

    Access configuration via: Config.api, Config.data, etc.
    """

    api: APISettings = APISettings()
    data: DataSettings = DataSettings()
    indicators: IndicatorSettings = IndicatorSettings()
    signals: SignalSettings = SignalSettings()
    sentiment: SentimentSettings = SentimentSettings()
    symbols: SymbolSettings = SymbolSettings()
    paths: PathSettings = PathSettings()
    scanner: ScannerSettings = ScannerSettings()
    market: MarketSettings = MarketSettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def reload(cls):
        """Reload configuration (useful after environment changes)."""
        load_dotenv(override=True)
        cls.api = APISettings()
        cls.paths = PathSettings()

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages
        """
        messages = []

        # Check API keys
        if not cls.api.NEWSAPI_KEY and not cls.api.NEWS_API_KEY:
            messages.append(
                "WARNING: No NewsAPI key configured. News fetching may be limited."
            )

        if not cls.api.OPENAI_API_KEY:
            messages.append(
                "WARNING: No OpenAI API key configured. OpenAI sentiment unavailable."
            )

        # Check indicator sanity
        if cls.indicators.SMA_SHORT >= cls.indicators.SMA_LONG:
            messages.append("ERROR: SMA_SHORT must be less than SMA_LONG")

        if cls.indicators.EMA_SHORT >= cls.indicators.EMA_LONG:
            messages.append("ERROR: EMA_SHORT must be less than EMA_LONG")

        # Check signal thresholds
        if cls.signals.MIN_RRR < 1.0:
            messages.append(
                "WARNING: MIN_RRR less than 1.0 indicates unfavorable risk/reward"
            )

        return messages


# ============================== Constants ============================== #

# HTTP Headers for NSE
NSE_HEADERS: Final[dict[str, str]] = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "accept": "application/json,text/html,application/xhtml+xml",
    "referer": "https://www.nseindia.com/",
    "accept-language": "en-US,en;q=0.9",
}

# OpenAI System Prompts
OPENAI_SENTIMENT_SYSTEM_PROMPT: Final[str] = (
    "You are a markets analyst. For each headline, classify sentiment for the STOCK PRICE reaction "
    "as positive, negative, or neutral, and provide an impact score from -2 to +2 "
    "(-2 strongly bearish, +2 strongly bullish). Return only JSON with an array 'items' "
    "of objects {headline, sentiment, impact, reason}."
)


# ============================== Export ============================== #

__all__ = [
    "Config",
    "APISettings",
    "DataSettings",
    "IndicatorSettings",
    "SignalSettings",
    "SentimentSettings",
    "SymbolSettings",
    "PathSettings",
    "ScannerSettings",
    "MarketSettings",
    "LoggingSettings",
    "NSE_HEADERS",
    "OPENAI_SENTIMENT_SYSTEM_PROMPT",
]
