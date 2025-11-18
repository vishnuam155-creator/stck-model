"""
Stock Classification System
Classify stocks by market cap, sector, industry, indices, and custom criteria
"""

import pandas as pd
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockClassification:
    """Stock classification data structure"""
    symbol: str
    name: str
    market_cap: float
    market_cap_category: str  # Large, Mid, Small
    sector: str
    industry: str
    index_membership: List[str]  # NIFTY50, NIFTY100, etc.
    liquidity: str  # High, Medium, Low
    volatility: str  # High, Medium, Low
    beta: float
    pe_ratio: float
    pb_ratio: float


class StockClassifier:
    """
    Classify and categorize stocks based on various criteria
    """

    # Market cap thresholds in Crores (INR)
    LARGE_CAP_THRESHOLD = 20000  # > 20,000 Cr
    MID_CAP_THRESHOLD = 5000     # 5,000 - 20,000 Cr
    # < 5,000 Cr is Small Cap

    # Liquidity thresholds (average daily volume)
    HIGH_LIQUIDITY_THRESHOLD = 1000000    # > 1M
    MEDIUM_LIQUIDITY_THRESHOLD = 100000   # 100K - 1M

    # Volatility thresholds (ATR %)
    HIGH_VOLATILITY_THRESHOLD = 3.0  # > 3%
    MEDIUM_VOLATILITY_THRESHOLD = 1.5  # 1.5% - 3%

    def __init__(self):
        """Initialize stock classifier"""
        self.nifty50_list = self._get_nifty50()
        self.nifty100_list = self._get_nifty100()
        self.sector_map = self._get_sector_mapping()

    def classify_stock(self, symbol: str, exchange: str = 'NSE') -> Optional[StockClassification]:
        """
        Classify a single stock

        Args:
            symbol: Stock symbol
            exchange: Exchange suffix

        Returns:
            StockClassification object or None
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
                logger.warning(f"No info available for {symbol}")
                return None

            # Extract basic info
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            name = info.get('longName', symbol)
            avg_volume = info.get('averageVolume', 0)
            beta = info.get('beta', 0)
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)

            # Convert market cap to INR Crores (assuming it's in the stock's currency)
            # For NSE stocks, Yahoo returns in INR
            market_cap_cr = market_cap / 10000000  # Convert to Crores

            # Classify market cap
            market_cap_category = self._classify_market_cap(market_cap_cr)

            # Determine index membership
            index_membership = []
            if symbol in self.nifty50_list:
                index_membership.append('NIFTY50')
            if symbol in self.nifty100_list:
                index_membership.append('NIFTY100')

            # Classify liquidity
            liquidity = self._classify_liquidity(avg_volume)

            # Get historical data for volatility
            hist = ticker.history(period='1mo', interval='1d')
            volatility = 'UNKNOWN'
            if not hist.empty and len(hist) > 1:
                # Calculate ATR as % of price
                high_low = hist['High'] - hist['Low']
                high_close = abs(hist['High'] - hist['Close'].shift(1))
                low_close = abs(hist['Low'] - hist['Close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.mean()
                current_price = hist['Close'].iloc[-1]
                atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
                volatility = self._classify_volatility(atr_pct)

            classification = StockClassification(
                symbol=symbol,
                name=name,
                market_cap=market_cap_cr,
                market_cap_category=market_cap_category,
                sector=sector,
                industry=industry,
                index_membership=index_membership,
                liquidity=liquidity,
                volatility=volatility,
                beta=beta,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio
            )

            return classification

        except Exception as e:
            logger.error(f"Error classifying {symbol}: {str(e)}")
            return None

    def classify_multiple(self, symbols: List[str], exchange: str = 'NSE') -> Dict[str, StockClassification]:
        """
        Classify multiple stocks

        Args:
            symbols: List of stock symbols
            exchange: Exchange suffix

        Returns:
            Dictionary mapping symbol to StockClassification
        """
        results = {}

        for symbol in symbols:
            logger.info(f"Classifying {symbol}...")
            classification = self.classify_stock(symbol, exchange)

            if classification:
                results[symbol] = classification

        logger.info(f"Successfully classified {len(results)}/{len(symbols)} stocks")
        return results

    def filter_by_market_cap(
        self,
        stocks: Dict[str, StockClassification],
        category: Literal['LARGE', 'MID', 'SMALL']
    ) -> Dict[str, StockClassification]:
        """Filter stocks by market cap category"""
        return {
            symbol: classification
            for symbol, classification in stocks.items()
            if classification.market_cap_category == category
        }

    def filter_by_sector(
        self,
        stocks: Dict[str, StockClassification],
        sector: str
    ) -> Dict[str, StockClassification]:
        """Filter stocks by sector"""
        sector_upper = sector.upper()
        return {
            symbol: classification
            for symbol, classification in stocks.items()
            if classification.sector.upper() == sector_upper
        }

    def filter_by_index(
        self,
        stocks: Dict[str, StockClassification],
        index: str
    ) -> Dict[str, StockClassification]:
        """Filter stocks by index membership"""
        return {
            symbol: classification
            for symbol, classification in stocks.items()
            if index in classification.index_membership
        }

    def filter_by_liquidity(
        self,
        stocks: Dict[str, StockClassification],
        liquidity: Literal['HIGH', 'MEDIUM', 'LOW']
    ) -> Dict[str, StockClassification]:
        """Filter stocks by liquidity"""
        return {
            symbol: classification
            for symbol, classification in stocks.items()
            if classification.liquidity == liquidity
        }

    def filter_by_volatility(
        self,
        stocks: Dict[str, StockClassification],
        volatility: Literal['HIGH', 'MEDIUM', 'LOW']
    ) -> Dict[str, StockClassification]:
        """Filter stocks by volatility"""
        return {
            symbol: classification
            for symbol, classification in stocks.items()
            if classification.volatility == volatility
        }

    def get_stocks_by_type(
        self,
        stock_type: str,
        exchange: str = 'NSE'
    ) -> Dict[str, StockClassification]:
        """
        Get stocks classified by a specific type

        Args:
            stock_type: Type of stocks to get
                - 'large_cap', 'mid_cap', 'small_cap'
                - 'nifty50', 'nifty100'
                - 'high_liquidity', 'medium_liquidity', 'low_liquidity'
                - Sector names: 'IT', 'BANKING', 'PHARMA', etc.

        Returns:
            Dictionary of classified stocks
        """
        stock_type_upper = stock_type.upper()

        # Get appropriate stock list
        if stock_type_upper == 'NIFTY50':
            symbols = self.nifty50_list
        elif stock_type_upper == 'NIFTY100':
            symbols = self.nifty100_list
        elif stock_type_upper in self.sector_map:
            symbols = self.sector_map[stock_type_upper]
        else:
            # For market cap or other criteria, use NIFTY100 as base
            symbols = self.nifty100_list

        # Classify all stocks
        classified = self.classify_multiple(symbols, exchange)

        # Apply filters based on type
        if stock_type_upper == 'LARGE_CAP':
            return self.filter_by_market_cap(classified, 'LARGE')
        elif stock_type_upper == 'MID_CAP':
            return self.filter_by_market_cap(classified, 'MID')
        elif stock_type_upper == 'SMALL_CAP':
            return self.filter_by_market_cap(classified, 'SMALL')
        elif stock_type_upper == 'HIGH_LIQUIDITY':
            return self.filter_by_liquidity(classified, 'HIGH')
        elif stock_type_upper == 'MEDIUM_LIQUIDITY':
            return self.filter_by_liquidity(classified, 'MEDIUM')
        elif stock_type_upper == 'LOW_LIQUIDITY':
            return self.filter_by_liquidity(classified, 'LOW')
        else:
            return classified

    def get_summary_stats(self, stocks: Dict[str, StockClassification]) -> Dict[str, Any]:
        """Get summary statistics for a set of classified stocks"""
        if not stocks:
            return {}

        market_caps = [s.market_cap for s in stocks.values() if s.market_cap > 0]
        pe_ratios = [s.pe_ratio for s in stocks.values() if s.pe_ratio > 0]
        pb_ratios = [s.pb_ratio for s in stocks.values() if s.pb_ratio > 0]
        betas = [s.beta for s in stocks.values() if s.beta > 0]

        # Count by category
        market_cap_counts = {}
        sector_counts = {}
        liquidity_counts = {}
        volatility_counts = {}

        for classification in stocks.values():
            # Market cap
            market_cap_counts[classification.market_cap_category] = \
                market_cap_counts.get(classification.market_cap_category, 0) + 1

            # Sector
            sector_counts[classification.sector] = \
                sector_counts.get(classification.sector, 0) + 1

            # Liquidity
            liquidity_counts[classification.liquidity] = \
                liquidity_counts.get(classification.liquidity, 0) + 1

            # Volatility
            volatility_counts[classification.volatility] = \
                volatility_counts.get(classification.volatility, 0) + 1

        summary = {
            'total_stocks': len(stocks),
            'market_cap': {
                'average': sum(market_caps) / len(market_caps) if market_caps else 0,
                'median': pd.Series(market_caps).median() if market_caps else 0,
                'min': min(market_caps) if market_caps else 0,
                'max': max(market_caps) if market_caps else 0,
                'by_category': market_cap_counts
            },
            'valuation': {
                'avg_pe': sum(pe_ratios) / len(pe_ratios) if pe_ratios else 0,
                'avg_pb': sum(pb_ratios) / len(pb_ratios) if pb_ratios else 0,
            },
            'risk': {
                'avg_beta': sum(betas) / len(betas) if betas else 0,
            },
            'sector_distribution': sector_counts,
            'liquidity_distribution': liquidity_counts,
            'volatility_distribution': volatility_counts
        }

        return summary

    # ==================== HELPER METHODS ====================

    def _classify_market_cap(self, market_cap_cr: float) -> str:
        """Classify market cap"""
        if market_cap_cr >= self.LARGE_CAP_THRESHOLD:
            return 'LARGE'
        elif market_cap_cr >= self.MID_CAP_THRESHOLD:
            return 'MID'
        else:
            return 'SMALL'

    def _classify_liquidity(self, avg_volume: float) -> str:
        """Classify liquidity based on average volume"""
        if avg_volume >= self.HIGH_LIQUIDITY_THRESHOLD:
            return 'HIGH'
        elif avg_volume >= self.MEDIUM_LIQUIDITY_THRESHOLD:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _classify_volatility(self, atr_pct: float) -> str:
        """Classify volatility based on ATR %"""
        if atr_pct >= self.HIGH_VOLATILITY_THRESHOLD:
            return 'HIGH'
        elif atr_pct >= self.MEDIUM_VOLATILITY_THRESHOLD:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_nifty50(self) -> List[str]:
        """Get NIFTY 50 stock list"""
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

    def _get_nifty100(self) -> List[str]:
        """Get NIFTY 100 stock list"""
        nifty50 = self._get_nifty50()
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

    def _get_sector_mapping(self) -> Dict[str, List[str]]:
        """Get sector to stocks mapping"""
        return {
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'COFORGE', 'MPHASIS', 'PERSISTENT', 'OFSS'],
            'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB', 'CANBK', 'BANDHANBNK'],
            'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'AUROPHARMA', 'LUPIN', 'TORNTPHARM', 'ALKEM', 'BIOCON'],
            'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'EICHERMOT', 'BAJAJ-AUTO', 'HEROMOTOCO', 'MOTHERSON'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'GODREJCP', 'TATACONSUM', 'COLPAL'],
            'METALS': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC', 'SAIL'],
            'ENERGY': ['RELIANCE', 'ONGC', 'BPCL', 'HINDPETRO', 'GAIL', 'NTPC', 'POWERGRID'],
            'REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE'],
            'TELECOM': ['BHARTIARTL', 'IDEA'],
            'CEMENT': ['ULTRACEMCO', 'SHREECEM', 'AMBUJACEM', 'ACC']
        }


# Convenience functions
def classify_stock(symbol: str, exchange: str = 'NSE') -> Optional[StockClassification]:
    """Quick function to classify a single stock"""
    classifier = StockClassifier()
    return classifier.classify_stock(symbol, exchange)


def get_stocks_by_type(stock_type: str, exchange: str = 'NSE') -> Dict[str, StockClassification]:
    """Quick function to get stocks by type"""
    classifier = StockClassifier()
    return classifier.get_stocks_by_type(stock_type, exchange)
