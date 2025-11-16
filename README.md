# Stock Analysis System - NIFTY 50 Scanner

A comprehensive stock analysis system for scanning NIFTY 50 stocks with technical indicators, news sentiment analysis, and automated trading signal generation.

## Features

- **Technical Analysis**: RSI, MACD, Stochastic Oscillator, Moving Averages (SMA/EMA), ATR
- **News Sentiment**: Integration with NewsAPI, GDELT, Google News RSS, and FinBERT/OpenAI sentiment analysis
- **Signal Generation**: Automated buy/sell signals with entry, stop-loss, and target levels
- **Risk Management**: Risk-reward ratio (RRR) calculation and position sizing
- **Market Breadth**: NIFTY Put-Call Ratio (PCR) for market context
- **Multiple Scanners**: Three scanner implementations with different features
- **Type Safety**: Comprehensive type definitions using TypedDict and Protocol
- **Error Handling**: Custom exception hierarchy for precise error management
- **Configuration Management**: Centralized configuration with validation

## Project Structure

```
stck-model/
├── StockFinder.py           # Main scanner with FinBERT/NLTK sentiment
├── nifty50_scanner.py        # Scanner with OpenAI sentiment
├── nifty50_scanner_.py       # Enhanced scanner with both engines
├── types_definitions.py      # Comprehensive type definitions
├── config.py                 # Configuration management
├── exceptions.py             # Custom exception classes
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── .cache/                   # Sentiment cache (auto-created)
├── output/                   # CSV results (auto-created)
├── reports/                  # Markdown reports (auto-created)
└── charts/                   # HTML charts (auto-created)
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stck-model
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (for StockFinder.py)

```python
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Optional: For news fetching (get from https://newsapi.org/)
NEWSAPI_KEY=your_newsapi_key_here
NEWS_API_KEY=your_newsapi_key_here

# Optional: For OpenAI sentiment (get from https://platform.openai.com/)
OPENAI_API_KEY=your_openai_key_here
```

## Usage

### StockFinder.py - Main Scanner

**Features**: FinBERT or NLTK sentiment, comprehensive technical analysis, market breadth

```bash
# Basic scan (3 days, max 50 stocks)
python StockFinder.py

# Custom parameters
python StockFinder.py --days 5 --max-stocks 20

# Generate reports and use FinBERT
python StockFinder.py --make-report --finbert --days 3

# Full options
python StockFinder.py --days 3 --max-stocks 50 --make-report --finbert
```

**Arguments**:
- `--days`: Lookback window for news and trend (default: 3)
- `--max-stocks`: Limit scan for speed (default: 50)
- `--make-report`: Save markdown report and CSV
- `--finbert`: Use FinBERT sentiment instead of NLTK

**Output**:
- Console: Summary table with signals
- `signals_last{N}d.csv`: Detailed results
- `report_last{N}d.md`: Markdown report (if --make-report)

### nifty50_scanner.py - OpenAI Sentiment Scanner

**Features**: OpenAI GPT-based sentiment, caching, 15-minute interval analysis

```bash
# Basic scan with OpenAI
python nifty50_scanner.py --model gpt-4o-mini

# Custom parameters
python nifty50_scanner.py --days 3 --limit 50 --out results.csv

# Generate charts
python nifty50_scanner.py --charts

# Full options
python nifty50_scanner.py --model gpt-4o-mini --days 3 --limit 50 --out results.csv --charts
```

**Arguments**:
- `--model`: OpenAI model (default: gpt-4o-mini)
- `--days`: Lookback days (default: 3)
- `--limit`: Number of symbols to scan (default: 50)
- `--out`: CSV output path (default: results.csv)
- `--charts`: Save Plotly HTML charts

**Output**:
- `results.csv`: Analysis results
- `charts/*.html`: Interactive price charts (if --charts)

### nifty50_scanner_.py - Enhanced Scanner

**Features**: FinBERT or OpenAI sentiment, daily 50/200 DMA flags, markdown table output

```bash
# Use FinBERT (faster, local)
python nifty50_scanner_.py --sentiment finbert

# Use OpenAI (requires API key)
python nifty50_scanner_.py --sentiment openai --model gpt-4o-mini

# Custom scan
python nifty50_scanner_.py --sentiment finbert --days 5 --limit 30 --charts

# Full options
python nifty50_scanner_.py --sentiment finbert --days 3 --limit 50 --out results.csv --charts
```

**Arguments**:
- `--sentiment`: Engine (finbert or openai, default: finbert)
- `--model`: OpenAI model if using openai sentiment
- `--days`: Lookback days (default: 3)
- `--limit`: Number of symbols (default: 50)
- `--out`: CSV output path (default: results.csv)
- `--charts`: Save Plotly HTML charts

**Output**:
- `results.csv`: Detailed analysis
- `results.md`: Markdown summary table
- `charts/*.html`: Interactive charts (if --charts)

## Configuration

### Centralized Configuration (`config.py`)

All configuration is centralized in `config.py`:

```python
from config import Config

# Access configuration
api_key = Config.api.NEWSAPI_KEY
rsi_length = Config.indicators.RSI_LENGTH
min_rrr = Config.signals.MIN_RRR
```

### Key Configuration Sections

1. **API Settings**: Keys, endpoints, timeouts
2. **Data Settings**: Lookback periods, intervals, quality thresholds
3. **Indicator Settings**: RSI, MACD, SMA, EMA, ATR parameters
4. **Signal Settings**: Scoring thresholds, risk management
5. **Sentiment Settings**: Engine selection, model parameters
6. **Symbol Settings**: NIFTY 50 list, Yahoo Finance suffixes
7. **Path Settings**: Cache, output, chart directories
8. **Market Settings**: PCR thresholds, trading hours

### Validate Configuration

```python
from config import Config

warnings = Config.validate()
for warning in warnings:
    print(warning)
```

## Type Definitions

Comprehensive type safety using `types_definitions.py`:

```python
from types_definitions import SignalData, SentimentResult, TechnicalIndicators

# Type-checked function
def process_signal(signal: SignalData) -> None:
    print(f"{signal['symbol']}: {signal['strong']}")
```

### Available Types

- **Signal Types**: `SignalType`, `SignalData`, `EntryExitLevels`
- **Sentiment Types**: `SentimentType`, `SentimentResult`, `SentimentDetail`
- **Technical Types**: `TechnicalIndicators`, `SwingLevels`
- **News Types**: `NewsArticle`
- **Configuration Types**: `APIConfig`, `ScannerConfig`, `IndicatorConfig`
- **Protocols**: `SentimentAnalyzer`, `NewsProvider`, `TechnicalAnalyzer`

## Exception Handling

Custom exception hierarchy in `exceptions.py`:

```python
from exceptions import DataFetchError, APIKeyMissingError

try:
    # Fetch data
    pass
except DataFetchError as e:
    print(f"Data fetch failed: {e.message}")
    print(f"Details: {e.details}")
except APIKeyMissingError as e:
    print(f"Missing API key: {e.details['env_var']}")
```

### Exception Categories

- **Data Errors**: `DataFetchError`, `DataValidationError`, `InsufficientDataError`
- **API Errors**: `APIKeyMissingError`, `APIRateLimitError`, `APIResponseError`
- **Analysis Errors**: `IndicatorCalculationError`, `SignalGenerationError`
- **Sentiment Errors**: `SentimentModelError`, `NewsNotFoundError`
- **Configuration Errors**: `InvalidConfigurationError`, `MissingDependencyError`

## Utility Functions

Common utilities in `utils.py`:

```python
from utils import safe_float, now_ist, clean_symbol, md5_hash

# Safe numeric conversion
price = safe_float(data['close'], default=0.0)

# Time handling
current_time = now_ist()  # IST timezone-aware

# Symbol cleaning
symbol = clean_symbol("RELIANCE.NS")  # Returns "RELIANCE"

# Cache key generation
cache_key = md5_hash("some text")
```

## Output Format

### CSV Output

Columns include:
- `symbol`, `name`: Stock identification
- `signal`, `reason`: Trading signal and rationale
- `entry`, `stop`, `target`, `rrr`: Trade levels
- `rsi`, `macd_hist`, `stoch_k`: Technical indicators
- `trend`, `golden_cross`, `death_cross`: Trend analysis
- `above_50dma`, `above_200dma`: Daily MA positioning
- `sentiment_score`, `news`: Sentiment analysis
- `top_news`: Top 3 impactful headlines

### Markdown Report

Generated with `--make-report` flag:

```markdown
# NIFTY 50 Signals — last 3 days

**NIFTY PCR** ≈ 1.15 → Neutral

## Strong Buy
- **RELIANCE** — HH/HL + above 50/200-DMA, Breakout + bullish retest setup | Entry ~ 2450, SL 2420, Target 2490 (RRR 1.33); RSI 62.5, MACDh 3.2. News: positive (5 vs 2); continuously_increasing

## Strong Sell
- **TATASTEEL** — LH/LL + below 50/200-DMA, Breakdown + failed retest setup | Short near 115, SL 118, Target 109 (RRR 2.0); RSI 38, MACDh -1.5. News: negative (4 vs 1)
```

## Technical Indicators Explained

### RSI (Relative Strength Index)
- **Length**: 14
- **Oversold**: < 30
- **Overbought**: > 70
- **Signal**: RSI > 50 for bullish, < 50 for bearish

### MACD (Moving Average Convergence Divergence)
- **Fast**: 12, **Slow**: 26, **Signal**: 9
- **Signal**: MACD above signal line = bullish

### Moving Averages
- **SMA 50/200**: Daily trend structure
- **EMA 20/50**: Intraday support/resistance
- **Golden Cross**: SMA 50 crosses above SMA 200
- **Death Cross**: SMA 50 crosses below SMA 200

### ATR (Average True Range)
- **Length**: 14
- **Usage**: Stop-loss at entry ± 1.5 × ATR
- **Target**: Entry + 2.0 × ATR

## Signal Generation Logic

### Strong Buy Criteria
1. **Trend**: Close > SMA50 > SMA200
2. **Higher Highs/Higher Lows**: Last 3+ bars
3. **Volume**: Above 20-period moving average
4. **RSI**: > 50
5. **MACD**: Histogram > 0
6. **Breakout**: Close > 20-day high
7. **Sentiment**: Positive news impact

### Strong Sell Criteria
1. **Trend**: Close < SMA50 < SMA200
2. **Lower Highs/Lower Lows**: Last 3+ bars
3. **RSI**: < 50
4. **MACD**: Histogram < 0
5. **Breakdown**: Close < 20-day low
6. **Sentiment**: Negative news impact

## Risk Management

### Entry/Exit Levels

**Breakout Entry**:
```
Entry = Previous 20-day high
Stop Loss = Entry - (1.5 × ATR)
Target = Entry + (2.0 × ATR)
RRR = (Target - Entry) / (Entry - Stop Loss)
```

**Pullback Entry**:
```
Entry = EMA20 level
Stop Loss = Recent swing low
Target = Entry + (1.2 × risk)
```

### Position Sizing

```python
portfolio_value = 1000000  # 10 lakh
risk_per_trade = 0.02      # 2%
entry = 2450
stop = 2420
risk_per_share = entry - stop  # 30

position_size = (portfolio_value * risk_per_trade) / risk_per_share
# = (1000000 × 0.02) / 30 = 666 shares
```

## News Sources

1. **NewsAPI**: Requires API key, 50 articles/day free tier
2. **GDELT**: Free, global news database
3. **Google News RSS**: Free, no API key required
4. **NSE**: Option chain data for PCR calculation

## Sentiment Analysis

### FinBERT (Recommended)
- **Model**: ProsusAI/finbert
- **Pros**: Free, local, fast, finance-specific
- **Cons**: Requires ~1GB model download

### OpenAI GPT
- **Model**: gpt-4o-mini (configurable)
- **Pros**: State-of-art accuracy
- **Cons**: Requires API key, costs per request

### NLTK VADER
- **Pros**: Instant, no setup
- **Cons**: Lower accuracy for financial text

## Performance Tips

1. **Limit stocks**: Use `--max-stocks` or `--limit` to scan fewer symbols
2. **Use cache**: Sentiment results are cached automatically
3. **FinBERT over OpenAI**: Faster and free for local analysis
4. **Skip charts**: Omit `--charts` flag for faster execution
5. **Parallel scanning**: Scanners use concurrent requests internally

## Troubleshooting

### Import Errors

```bash
# Missing dependencies
pip install -r requirements.txt

# FinBERT model download
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert')"
```

### API Errors

```bash
# Check .env file
cat .env

# Verify API key
echo $NEWSAPI_KEY
```

### Data Fetch Errors

- **yfinance rate limits**: Add delays between requests
- **NSE blocks**: Use VPN or retry with delays
- **Empty data**: Symbol may be delisted or suspended

### Type Checking

```bash
# Run mypy for type validation
pip install mypy
mypy StockFinder.py --ignore-missing-imports
```

## Development

### Code Quality

```bash
# Format code
pip install black
black *.py

# Lint code
pip install flake8
flake8 *.py --max-line-length=100

# Sort imports
pip install isort
isort *.py
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests (when test files are added)
pytest tests/ -v --cov=.
```

## License

This project is for educational and research purposes. Always comply with exchange regulations and terms of service for data providers.

## Disclaimer

**This software is for educational purposes only. Do not use it for actual trading without proper testing and risk management. The authors are not responsible for any financial losses.**

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add type hints and docstrings
4. Update documentation
5. Submit a pull request

## Acknowledgments

- **yfinance**: Yahoo Finance data API
- **pandas-ta**: Technical analysis library
- **FinBERT**: ProsusAI's financial sentiment model
- **NewsAPI**: News aggregation service
- **GDELT**: Global news database

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Changelog

### Version 2.0 (Current)
- Added comprehensive type definitions
- Implemented custom exception hierarchy
- Centralized configuration management
- Created utility module for code reuse
- Enhanced documentation
- Improved error handling

### Version 1.0
- Initial release with three scanner implementations
- Basic technical analysis and sentiment analysis
