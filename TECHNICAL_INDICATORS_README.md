# Technical Indicators System for Real Stocks

A comprehensive Python library for applying technical indicators to real stock data with classification and screening capabilities.

## 🚀 Features

### 1. **Comprehensive Technical Indicators (20+ Indicators)**
- **Trend Indicators**: SMA, EMA, WMA, DEMA, TEMA, VWAP, SuperTrend, ADX, Ichimoku
- **Momentum Indicators**: RSI, MACD, Stochastic, CCI, Williams %R, ROC, MFI, Ultimate Oscillator
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels, Donchian Channels, Historical Volatility
- **Volume Indicators**: OBV, A/D, CMF, VWMA, Volume SMA, PVT
- **Support/Resistance**: Pivot Points, Swing Highs/Lows
- **Pattern Recognition**: Doji, Hammer, Shooting Star, Engulfing Patterns

### 2. **Real Stock Data Integration**
- Primary source: Yahoo Finance
- Supports NSE, BSE, and US exchanges
- Intraday and historical data
- Live price feeds
- Automatic fallback mechanisms

### 3. **Stock Classification System**
- **Market Cap**: Large-cap, Mid-cap, Small-cap
- **Sectors**: IT, Banking, Pharma, Auto, FMCG, Metals, Energy, etc.
- **Indices**: NIFTY 50, NIFTY 100
- **Liquidity**: High, Medium, Low
- **Volatility**: High, Medium, Low

### 4. **Indicator Application Engine**
- Apply any indicator to any stock
- Batch analysis for multiple stocks
- Stock screening with custom criteria
- Comparison and ranking
- Export to CSV, JSON, Excel

## 📦 Installation

```bash
# Install required packages
pip install yfinance pandas pandas_ta numpy requests openpyxl
```

## 🎯 Quick Start

### Example 1: Basic Indicator Calculation

```python
from technical_indicators import TechnicalIndicators
from stock_data_fetcher import fetch_stock

# Fetch stock data
df = fetch_stock('RELIANCE', period='3mo', interval='1d', exchange='NSE')

# Calculate all indicators
ti = TechnicalIndicators(df)
result_df = ti.calculate_all()

# Get latest signals
signals = ti.get_latest_signals()
print(f"Trend: {signals['signals']['trend']}")
print(f"Momentum: {signals['signals']['momentum']}")
```

### Example 2: Stock Classification

```python
from stock_classifier import classify_stock

# Classify a stock
classification = classify_stock('TCS', exchange='NSE')

print(f"Market Cap: {classification.market_cap_category}")
print(f"Sector: {classification.sector}")
print(f"Liquidity: {classification.liquidity}")
print(f"Volatility: {classification.volatility}")
```

### Example 3: Complete Stock Analysis

```python
from indicator_engine import analyze_stock

# Comprehensive analysis
analysis = analyze_stock('INFY', period='3mo', interval='1d', exchange='NSE')

print(f"Price: ₹{analysis['latest_price']:.2f}")
print(f"RSI: {analysis['indicators']['RSI_14']:.2f}")
print(f"Trend: {analysis['signals']['trend']}")
print(f"Sector: {analysis['classification']['sector']}")
```

### Example 4: Stock Screening

```python
from indicator_engine import screen_stocks

# Define screening criteria
criteria = {
    'rsi_min': 30,
    'rsi_max': 70,
    'trend': 'BULLISH',
    'volume_spike': True,
    'market_cap': 'LARGE',
    'liquidity': 'HIGH'
}

# Screen stocks
symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
screened = screen_stocks(symbols, criteria, period='1mo', exchange='NSE')

for symbol, analysis in screened.items():
    print(f"{symbol}: RSI={analysis['indicators']['RSI_14']:.2f}")
```

### Example 5: Compare Multiple Stocks

```python
from indicator_engine import IndicatorEngine

engine = IndicatorEngine()

# Compare stocks
symbols = ['RELIANCE', 'TCS', 'INFY']
comparison = engine.compare_stocks(
    symbols,
    metrics=['latest_price', 'RSI_14', 'trend', 'momentum', 'sector'],
    period='1mo'
)

print(comparison)
```

### Example 6: Sector Analysis

```python
from indicator_engine import IndicatorEngine

engine = IndicatorEngine()

# Analyze IT sector
it_analysis = engine.analyze_by_stock_type(
    stock_type='IT',
    period='1mo',
    limit=10
)

for symbol, analysis in it_analysis.items():
    print(f"{symbol}: {analysis['signals']['trend']}")
```

### Example 7: Intraday Analysis

```python
from stock_data_fetcher import fetch_stock
from technical_indicators import TechnicalIndicators

# Fetch 15-minute data
df = fetch_stock('RELIANCE', period='5d', interval='15m', exchange='NSE')

# Calculate indicators
ti = TechnicalIndicators(df)
ti.vwap()
ti.rsi(14)
ti.supertrend()

# Get latest signals
signals = ti.get_latest_signals()
```

## 📊 Available Modules

### 1. `technical_indicators.py`
Core indicators library with 20+ technical indicators.

**Key Classes:**
- `TechnicalIndicators`: Main class for calculating indicators

**Methods:**
- `sma(period)`: Simple Moving Average
- `ema(period)`: Exponential Moving Average
- `rsi(period)`: Relative Strength Index
- `macd()`: MACD indicator
- `bollinger_bands()`: Bollinger Bands
- `atr(period)`: Average True Range
- `vwap()`: Volume Weighted Average Price
- `obv()`: On Balance Volume
- `calculate_all()`: Calculate all indicators at once
- `get_latest_signals()`: Get latest indicator values and signals

### 2. `stock_data_fetcher.py`
Fetches real stock data from multiple sources.

**Key Classes:**
- `StockDataFetcher`: Main data fetching class

**Methods:**
- `fetch_stock_data(symbol, period, interval)`: Fetch stock data
- `fetch_multiple_stocks(symbols)`: Fetch multiple stocks
- `get_live_price(symbol)`: Get current live price
- `get_stock_info(symbol)`: Get comprehensive stock info
- `get_nifty50_stocks()`: Get NIFTY 50 list
- `get_sector_stocks(sector)`: Get stocks by sector

### 3. `stock_classifier.py`
Classifies stocks by various criteria.

**Key Classes:**
- `StockClassifier`: Main classification class
- `StockClassification`: Data structure for classification

**Methods:**
- `classify_stock(symbol)`: Classify a single stock
- `classify_multiple(symbols)`: Classify multiple stocks
- `filter_by_market_cap(stocks, category)`: Filter by market cap
- `filter_by_sector(stocks, sector)`: Filter by sector
- `get_stocks_by_type(stock_type)`: Get stocks by type
- `get_summary_stats(stocks)`: Get summary statistics

### 4. `indicator_engine.py`
Complete engine to apply indicators and generate analysis.

**Key Classes:**
- `IndicatorEngine`: Main application engine

**Methods:**
- `analyze_stock(symbol)`: Complete stock analysis
- `analyze_multiple_stocks(symbols)`: Analyze multiple stocks
- `analyze_by_stock_type(stock_type)`: Analyze by stock type
- `screen_stocks(symbols, criteria)`: Screen stocks by criteria
- `compare_stocks(symbols, metrics)`: Compare multiple stocks
- `generate_report(analysis, format)`: Generate analysis report
- `export_analysis(analysis, filename, format)`: Export to file

## 🔧 Supported Stock Types

### By Index
- `NIFTY50`: Top 50 stocks
- `NIFTY100`: Top 100 stocks

### By Market Cap
- `LARGE_CAP`: Market cap > ₹20,000 Cr
- `MID_CAP`: Market cap ₹5,000 - ₹20,000 Cr
- `SMALL_CAP`: Market cap < ₹5,000 Cr

### By Sector
- `IT`: Information Technology
- `BANKING`: Banking & Financial Services
- `PHARMA`: Pharmaceuticals
- `AUTO`: Automobile
- `FMCG`: Fast Moving Consumer Goods
- `METALS`: Metals & Mining
- `ENERGY`: Energy & Power
- `REALTY`: Real Estate
- `TELECOM`: Telecommunications
- `CEMENT`: Cement

### By Liquidity
- `HIGH_LIQUIDITY`: Avg volume > 1M
- `MEDIUM_LIQUIDITY`: Avg volume 100K - 1M
- `LOW_LIQUIDITY`: Avg volume < 100K

## 📈 Supported Data Intervals

### Historical Data
- `1d`: Daily
- `1wk`: Weekly
- `1mo`: Monthly

### Intraday Data
- `1m`: 1 minute
- `5m`: 5 minutes
- `15m`: 15 minutes
- `30m`: 30 minutes
- `1h`: 1 hour

## 🎨 Export Formats

- **CSV**: Plain CSV with all data
- **JSON**: JSON format with metadata
- **Excel**: Excel workbook with multiple sheets

## 🧪 Run Examples

```bash
python example_usage.py
```

This will show you interactive examples of all features.

## 📝 Screening Criteria

You can screen stocks using various criteria:

```python
criteria = {
    # Momentum
    'rsi_min': 30,          # Minimum RSI
    'rsi_max': 70,          # Maximum RSI

    # Trend
    'trend': 'BULLISH',     # BULLISH, BEARISH, NEUTRAL
    'momentum': 'OVERSOLD', # OVERBOUGHT, OVERSOLD, BULLISH, BEARISH

    # Volume
    'volume_spike': True,   # High volume required

    # Classification
    'market_cap': 'LARGE',  # LARGE, MID, SMALL
    'sector': 'Technology', # Specific sector
    'liquidity': 'HIGH'     # HIGH, MEDIUM, LOW
}
```

## 📊 Analysis Output Structure

```python
{
    'symbol': 'RELIANCE',
    'latest_price': 2450.50,
    'price_change_pct': 2.5,
    'indicators': {
        'RSI_14': 55.2,
        'MACD': 12.5,
        'ATR_14': 45.3,
        'VWAP': 2445.0,
        # ... all calculated indicators
    },
    'signals': {
        'trend': 'BULLISH',
        'momentum': 'NEUTRAL',
        'volatility': 'NORMAL',
        'volume': 'HIGH'
    },
    'classification': {
        'market_cap_category': 'LARGE',
        'sector': 'Energy',
        'liquidity': 'HIGH',
        'volatility': 'MEDIUM',
        # ... more classification data
    }
}
```

## 🔍 Advanced Usage

### Custom Indicator Configuration

```python
from technical_indicators import TechnicalIndicators

ti = TechnicalIndicators(df)

# Custom configuration
custom_config = {
    'sma_periods': [10, 20, 50, 100, 200],
    'ema_periods': [5, 9, 12, 21, 26, 50],
    'rsi_period': 14,
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bb_period': 20,
    'atr_period': 14
}

result_df = ti.calculate_all(custom_config)
```

### Batch Processing

```python
from indicator_engine import IndicatorEngine
from stock_data_fetcher import StockDataFetcher

engine = IndicatorEngine()
fetcher = StockDataFetcher()

# Get all NIFTY 50 stocks
symbols = fetcher.get_nifty50_stocks()

# Analyze all at once
results = engine.analyze_multiple_stocks(symbols, period='3mo')

# Filter bullish stocks
bullish_stocks = {
    symbol: analysis
    for symbol, analysis in results.items()
    if analysis['signals']['trend'] == 'BULLISH'
}
```

## 🤝 Integration with Existing System

This library integrates seamlessly with your existing trading system:

```python
# Use with your existing intraday_trading_system.py
from intraday_trading_system import IntradayTradingSystem
from indicator_engine import IndicatorEngine

# Your existing system
trading_system = IntradayTradingSystem()

# New indicator engine
indicator_engine = IndicatorEngine()

# Combine both
symbols = trading_system.get_liquid_stocks()
analysis = indicator_engine.analyze_multiple_stocks(symbols)
```

## 📚 API Reference

See individual module docstrings for detailed API documentation:

```python
# View help for any module
from technical_indicators import TechnicalIndicators
help(TechnicalIndicators)

from stock_classifier import StockClassifier
help(StockClassifier)

from indicator_engine import IndicatorEngine
help(IndicatorEngine)
```

## ⚡ Performance Tips

1. **Use specific indicators** instead of `calculate_all()` if you only need a few
2. **Cache stock data** for repeated analysis
3. **Use batch operations** for multiple stocks
4. **Limit data period** for faster processing
5. **Use daily interval** unless you need intraday data

## 🐛 Troubleshooting

### No data returned
- Check internet connection
- Verify symbol is correct (e.g., 'RELIANCE' not 'RELIANCE.NS')
- Try different data source
- Check if market is open (for live data)

### Rate limiting
- Add delays between requests
- Use batch fetching
- Cache results

### Missing indicators
- Ensure sufficient data points (need 200+ for some indicators)
- Check for NaN values in data

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- `yfinance` for stock data
- `pandas_ta` for technical indicators
- NSE India for index data

## 📞 Support

For issues or questions, check the example scripts or module docstrings.

---

**Happy Trading! 📈**
