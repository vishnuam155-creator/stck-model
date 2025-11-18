"""
Real Stock Data Fetcher
Supports multiple data sources: Yahoo Finance, NSE, and extensible for others
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Literal
import time
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetch real stock data from multiple sources
    Primary: Yahoo Finance
    Secondary: NSE India
    """

    def __init__(self, source: Literal['yahoo', 'nse', 'auto'] = 'auto'):
        """
        Initialize stock data fetcher

        Args:
            source: Data source to use ('yahoo', 'nse', 'auto')
        """
        self.source = source
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_stock_data(
        self,
        symbol: str,
        period: str = '1mo',
        interval: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: str = 'NSE'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with automatic fallback

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            exchange: Exchange suffix ('NSE', 'BSE', 'US', None for no suffix)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Try primary source
        df = None

        if self.source in ['yahoo', 'auto']:
            df = self._fetch_yahoo(symbol, period, interval, start_date, end_date, exchange)

        # Fallback to NSE if yahoo fails
        if df is None and self.source in ['nse', 'auto'] and interval == '1d':
            df = self._fetch_nse(symbol)

        if df is not None:
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_cols):
                return df

        logger.warning(f"Failed to fetch data for {symbol}")
        return None

    def _fetch_yahoo(
        self,
        symbol: str,
        period: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
        exchange: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            # Format symbol with exchange suffix
            if exchange == 'NSE':
                yahoo_symbol = f"{symbol}.NS"
            elif exchange == 'BSE':
                yahoo_symbol = f"{symbol}.BO"
            elif exchange == 'US':
                yahoo_symbol = symbol
            else:
                yahoo_symbol = symbol

            logger.info(f"Fetching {yahoo_symbol} from Yahoo Finance...")

            ticker = yf.Ticker(yahoo_symbol)

            # Fetch data
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {yahoo_symbol}")
                return None

            # Clean data
            df = df.reset_index()
            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)

            # Ensure timezone-naive datetime
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            df.set_index('Date', inplace=True)

            logger.info(f"Successfully fetched {len(df)} rows for {yahoo_symbol}")
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
            return None

    def _fetch_nse(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from NSE India (daily data only)"""
        try:
            logger.info(f"Fetching {symbol} from NSE...")

            # NSE API endpoint (example - may need updates based on NSE API changes)
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse NSE data (structure may vary)
            # This is a simplified example
            if 'priceInfo' in data:
                price_info = data['priceInfo']
                df = pd.DataFrame([{
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Open': price_info.get('open', 0),
                    'High': price_info.get('intraDayHighLow', {}).get('max', 0),
                    'Low': price_info.get('intraDayHighLow', {}).get('min', 0),
                    'Close': price_info.get('lastPrice', 0),
                    'Volume': price_info.get('totalTradedVolume', 0)
                }])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df

            return None

        except Exception as e:
            logger.error(f"NSE error for {symbol}: {str(e)}")
            return None

    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            exchange: Exchange suffix

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")
            df = self.fetch_stock_data(symbol, period, interval, exchange=exchange)

            if df is not None:
                results[symbol] = df
            else:
                logger.warning(f"Skipping {symbol} - no data available")

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} stocks")
        return results

    def get_live_price(self, symbol: str, exchange: str = 'NSE') -> Optional[Dict[str, Any]]:
        """
        Get current live price and basic info

        Args:
            symbol: Stock symbol
            exchange: Exchange suffix

        Returns:
            Dictionary with live price info
        """
        try:
            # Format symbol
            if exchange == 'NSE':
                yahoo_symbol = f"{symbol}.NS"
            elif exchange == 'BSE':
                yahoo_symbol = f"{symbol}.BO"
            else:
                yahoo_symbol = symbol

            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            if not info:
                return None

            live_data = {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'open': info.get('regularMarketOpen', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'previous_close': info.get('previousClose', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now().isoformat()
            }

            # Calculate change
            if live_data['previous_close'] > 0:
                change = live_data['price'] - live_data['previous_close']
                change_pct = (change / live_data['previous_close']) * 100
                live_data['change'] = change
                live_data['change_percent'] = change_pct

            return live_data

        except Exception as e:
            logger.error(f"Error getting live price for {symbol}: {str(e)}")
            return None

    def get_stock_info(self, symbol: str, exchange: str = 'NSE') -> Optional[Dict[str, Any]]:
        """
        Get comprehensive stock information

        Args:
            symbol: Stock symbol
            exchange: Exchange suffix

        Returns:
            Dictionary with stock info (sector, industry, market cap, etc.)
        """
        try:
            # Format symbol
            if exchange == 'NSE':
                yahoo_symbol = f"{symbol}.NS"
            elif exchange == 'BSE':
                yahoo_symbol = f"{symbol}.BO"
            else:
                yahoo_symbol = symbol

            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            if not info:
                return None

            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52week_high': info.get('fiftyTwoWeekHigh', 0),
                '52week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'description': info.get('longBusinessSummary', ''),
                'exchange': info.get('exchange', exchange),
                'currency': info.get('currency', 'INR')
            }

            return stock_info

        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return None

    @lru_cache(maxsize=1)
    def get_nifty50_stocks(self) -> List[str]:
        """Fetch current NIFTY 50 stock list from NSE"""
        try:
            url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
            df = pd.read_csv(url)

            if 'Symbol' in df.columns:
                symbols = df['Symbol'].tolist()
                logger.info(f"Fetched {len(symbols)} NIFTY 50 stocks")
                return symbols

            return []

        except Exception as e:
            logger.error(f"Error fetching NIFTY 50 list: {str(e)}")
            # Fallback to hardcoded list
            return self._get_fallback_nifty50()

    def _get_fallback_nifty50(self) -> List[str]:
        """Fallback NIFTY 50 list"""
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'BAJFINANCE',
            'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
            'HCLTECH', 'ULTRACEMCO', 'SUNPHARMA', 'TITAN', 'NESTLEIND',
            'BAJAJFINSV', 'WIPRO', 'ADANIPORTS', 'ONGC', 'NTPC',
            'TECHM', 'POWERGRID', 'M&M', 'TATAMOTORS', 'TATASTEEL',
            'INDUSINDBK', 'COALINDIA', 'BRITANNIA', 'GRASIM', 'JSWSTEEL',
            'HINDALCO', 'DRREDDY', 'CIPLA', 'EICHERMOT', 'UPL',
            'DIVISLAB', 'APOLLOHOSP', 'HEROMOTOCO', 'BAJAJ-AUTO', 'SHREECEM',
            'TATACONSUM', 'ADANIENT', 'SBILIFE', 'BPCL', 'HDFCLIFE'
        ]

    def get_nifty100_stocks(self) -> List[str]:
        """Get NIFTY 100 stock list"""
        # This is a simplified version - can be expanded
        nifty50 = self.get_nifty50_stocks()

        # Additional stocks to make NIFTY 100
        additional = [
            'VEDL', 'PIDILITIND', 'SIEMENS', 'DLF', 'GODREJCP',
            'DABUR', 'HAVELLS', 'TORNTPHARM', 'BOSCHLTD', 'BERGEPAINT',
            'ALKEM', 'BANDHANBNK', 'MCDOWELL-N', 'AMBUJACEM', 'HINDPETRO',
            'INDIGO', 'AUROPHARMA', 'LUPIN', 'MARICO', 'MUTHOOTFIN',
            'ICICIPRULI', 'PEL', 'COLPAL', 'ACC', 'GAIL',
            'CONCOR', 'BIOCON', 'PETRONET', 'NMDC', 'SAIL',
            'MOTHERSON', 'TORNTPOWER', 'HDFCAMC', 'IDEA', 'GMRINFRA',
            'PAGEIND', 'PGHH', 'MPHASIS', 'OFSS', 'PERSISTENT',
            'LALPATHLAB', 'BATAINDIA', 'TATAPOWER', 'CANBK', 'PNB',
            'BANKBARODA', 'RECLTD', 'PFC', 'CHOLAFIN', 'ICICIGI'
        ]

        return nifty50 + additional

    def get_sector_stocks(self, sector: str) -> List[str]:
        """
        Get stocks by sector

        Args:
            sector: Sector name (e.g., 'IT', 'BANKING', 'PHARMA', 'AUTO')

        Returns:
            List of stock symbols in that sector
        """
        sector_mapping = {
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'COFORGE', 'MPHASIS', 'PERSISTENT'],
            'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB', 'CANBK'],
            'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'AUROPHARMA', 'LUPIN', 'TORNTPHARM', 'ALKEM', 'BIOCON'],
            'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'EICHERMOT', 'BAJAJ-AUTO', 'HEROMOTOCO', 'MOTHERSON', 'TVSMOTOR'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP', 'TATACONSUM', 'COLPAL'],
            'METALS': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC', 'SAIL', 'JINDALSTEL'],
            'ENERGY': ['RELIANCE', 'ONGC', 'BPCL', 'HINDPETRO', 'IOC', 'GAIL', 'ADANIGREEN', 'NTPC', 'POWERGRID'],
            'REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE', 'SOBHA'],
            'TELECOM': ['BHARTIARTL', 'IDEA', 'TATACOMM'],
            'CEMENT': ['ULTRACEMCO', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALMIACEM']
        }

        sector_upper = sector.upper()
        return sector_mapping.get(sector_upper, [])


# Convenience functions
def fetch_stock(symbol: str, period: str = '1mo', interval: str = '1d', exchange: str = 'NSE') -> Optional[pd.DataFrame]:
    """Quick function to fetch single stock data"""
    fetcher = StockDataFetcher()
    return fetcher.fetch_stock_data(symbol, period, interval, exchange=exchange)


def fetch_stocks(symbols: List[str], period: str = '1mo', interval: str = '1d', exchange: str = 'NSE') -> Dict[str, pd.DataFrame]:
    """Quick function to fetch multiple stocks"""
    fetcher = StockDataFetcher()
    return fetcher.fetch_multiple_stocks(symbols, period, interval, exchange)


def get_live_prices(symbols: List[str], exchange: str = 'NSE') -> Dict[str, Dict[str, Any]]:
    """Quick function to get live prices for multiple stocks"""
    fetcher = StockDataFetcher()
    results = {}

    for symbol in symbols:
        price_data = fetcher.get_live_price(symbol, exchange)
        if price_data:
            results[symbol] = price_data
        time.sleep(0.3)  # Rate limiting

    return results
