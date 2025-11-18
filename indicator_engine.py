"""
Indicator Application Engine
Apply technical indicators to stocks and generate comprehensive analysis
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import logging
import json

from technical_indicators import TechnicalIndicators, calculate_indicators
from stock_data_fetcher import StockDataFetcher
from stock_classifier import StockClassifier, StockClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Complete engine to apply technical indicators to stocks with classification
    """

    def __init__(self, data_source: str = 'auto'):
        """
        Initialize indicator engine

        Args:
            data_source: Data source for stock data ('yahoo', 'nse', 'auto')
        """
        self.data_fetcher = StockDataFetcher(source=data_source)
        self.classifier = StockClassifier()

    def analyze_stock(
        self,
        symbol: str,
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE',
        indicators: Optional[List[str]] = None,
        include_classification: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Complete analysis of a single stock

        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            exchange: Exchange suffix
            indicators: List of specific indicators (None = all)
            include_classification: Include stock classification

        Returns:
            Dictionary with complete analysis
        """
        logger.info(f"Analyzing {symbol}...")

        # Fetch stock data
        df = self.data_fetcher.fetch_stock_data(
            symbol, period, interval, exchange=exchange
        )

        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return None

        # Calculate indicators
        ti = TechnicalIndicators(df)

        if indicators:
            # Calculate specific indicators
            for indicator in indicators:
                self._calculate_indicator(ti, indicator)
            result_df = df.copy()
            for key, value in ti.results.items():
                if isinstance(value, pd.Series):
                    result_df[key] = value
        else:
            # Calculate all indicators
            result_df = ti.calculate_all()

        # Get latest signals
        signals = ti.get_latest_signals()

        # Get classification
        classification = None
        if include_classification:
            classification = self.classifier.classify_stock(symbol, exchange)

        # Compile analysis
        analysis = {
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': datetime.now().isoformat(),
            'data_period': period,
            'data_interval': interval,
            'data_points': len(result_df),
            'latest_price': float(result_df['Close'].iloc[-1]),
            'price_change_pct': float(
                ((result_df['Close'].iloc[-1] - result_df['Close'].iloc[0]) /
                 result_df['Close'].iloc[0] * 100)
                if len(result_df) > 1 else 0
            ),
            'indicators': signals['indicators'],
            'signals': signals['signals'],
            'classification': classification.__dict__ if classification else None,
            'data': result_df.tail(10).to_dict('records'),  # Last 10 rows
            'full_dataframe': result_df  # For further processing
        }

        return analysis

    def analyze_multiple_stocks(
        self,
        symbols: List[str],
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE',
        indicators: Optional[List[str]] = None,
        include_classification: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple stocks

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            exchange: Exchange suffix
            indicators: List of specific indicators (None = all)
            include_classification: Include stock classification

        Returns:
            Dictionary mapping symbol to analysis
        """
        results = {}

        for symbol in symbols:
            analysis = self.analyze_stock(
                symbol, period, interval, exchange, indicators, include_classification
            )

            if analysis:
                results[symbol] = analysis

        logger.info(f"Successfully analyzed {len(results)}/{len(symbols)} stocks")
        return results

    def analyze_by_stock_type(
        self,
        stock_type: str,
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE',
        indicators: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze stocks by type (e.g., NIFTY50, IT sector, Large Cap)

        Args:
            stock_type: Type of stocks ('NIFTY50', 'IT', 'LARGE_CAP', etc.)
            period: Data period
            interval: Data interval
            exchange: Exchange suffix
            indicators: List of specific indicators
            limit: Maximum number of stocks to analyze

        Returns:
            Dictionary mapping symbol to analysis
        """
        logger.info(f"Analyzing {stock_type} stocks...")

        # Get stocks by type
        classified_stocks = self.classifier.get_stocks_by_type(stock_type, exchange)

        if not classified_stocks:
            logger.warning(f"No stocks found for type: {stock_type}")
            return {}

        symbols = list(classified_stocks.keys())

        if limit:
            symbols = symbols[:limit]

        return self.analyze_multiple_stocks(
            symbols, period, interval, exchange, indicators, include_classification=True
        )

    def screen_stocks(
        self,
        symbols: List[str],
        criteria: Dict[str, Any],
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Screen stocks based on technical criteria

        Args:
            symbols: List of stock symbols
            criteria: Screening criteria, e.g.:
                {
                    'rsi_min': 30,
                    'rsi_max': 70,
                    'trend': 'BULLISH',
                    'volume_spike': True,
                    'market_cap': 'LARGE',
                    'sector': 'IT'
                }
            period: Data period
            interval: Data interval
            exchange: Exchange suffix

        Returns:
            Dictionary of stocks that meet criteria
        """
        logger.info(f"Screening {len(symbols)} stocks...")

        # Analyze all stocks
        analyses = self.analyze_multiple_stocks(
            symbols, period, interval, exchange, include_classification=True
        )

        # Apply criteria
        filtered = {}

        for symbol, analysis in analyses.items():
            if self._meets_criteria(analysis, criteria):
                filtered[symbol] = analysis

        logger.info(f"{len(filtered)} stocks passed screening")
        return filtered

    def compare_stocks(
        self,
        symbols: List[str],
        metrics: Optional[List[str]] = None,
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE'
    ) -> pd.DataFrame:
        """
        Compare multiple stocks side-by-side

        Args:
            symbols: List of stock symbols
            metrics: List of metrics to compare (None = common metrics)
            period: Data period
            interval: Data interval
            exchange: Exchange suffix

        Returns:
            DataFrame with comparison
        """
        analyses = self.analyze_multiple_stocks(
            symbols, period, interval, exchange, include_classification=True
        )

        if not analyses:
            return pd.DataFrame()

        # Default metrics
        if not metrics:
            metrics = [
                'latest_price', 'price_change_pct', 'RSI_14', 'MACD',
                'trend', 'momentum', 'volatility', 'market_cap_category',
                'sector', 'liquidity'
            ]

        # Build comparison dataframe
        comparison_data = []

        for symbol, analysis in analyses.items():
            row = {'symbol': symbol}

            for metric in metrics:
                # Check in different sections
                if metric in analysis:
                    row[metric] = analysis[metric]
                elif metric in analysis.get('indicators', {}):
                    row[metric] = analysis['indicators'][metric]
                elif metric in analysis.get('signals', {}):
                    row[metric] = analysis['signals'][metric]
                elif analysis.get('classification') and metric in analysis['classification']:
                    row[metric] = analysis['classification'][metric]
                else:
                    row[metric] = None

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def generate_report(
        self,
        analysis: Dict[str, Any],
        format: Literal['text', 'json', 'dict'] = 'text'
    ) -> Any:
        """
        Generate analysis report

        Args:
            analysis: Stock analysis dictionary
            format: Output format ('text', 'json', 'dict')

        Returns:
            Report in specified format
        """
        if format == 'dict':
            # Return clean dict (without full dataframe)
            clean_analysis = {k: v for k, v in analysis.items() if k != 'full_dataframe'}
            return clean_analysis

        elif format == 'json':
            clean_analysis = {k: v for k, v in analysis.items() if k != 'full_dataframe'}
            return json.dumps(clean_analysis, indent=2, default=str)

        else:  # text
            report_lines = []
            report_lines.append("=" * 70)
            report_lines.append(f"STOCK ANALYSIS REPORT: {analysis['symbol']}")
            report_lines.append("=" * 70)
            report_lines.append(f"Timestamp: {analysis['timestamp']}")
            report_lines.append(f"Exchange: {analysis['exchange']}")
            report_lines.append(f"Period: {analysis['data_period']} | Interval: {analysis['data_interval']}")
            report_lines.append("")

            # Price info
            report_lines.append("PRICE INFORMATION")
            report_lines.append("-" * 70)
            report_lines.append(f"Latest Price: ₹{analysis['latest_price']:.2f}")
            report_lines.append(f"Change: {analysis['price_change_pct']:.2f}%")
            report_lines.append("")

            # Classification
            if analysis.get('classification'):
                cls = analysis['classification']
                report_lines.append("CLASSIFICATION")
                report_lines.append("-" * 70)
                report_lines.append(f"Name: {cls['name']}")
                report_lines.append(f"Sector: {cls['sector']} | Industry: {cls['industry']}")
                report_lines.append(f"Market Cap: ₹{cls['market_cap']:.2f} Cr ({cls['market_cap_category']})")
                report_lines.append(f"Liquidity: {cls['liquidity']} | Volatility: {cls['volatility']}")
                report_lines.append(f"PE Ratio: {cls['pe_ratio']:.2f} | PB Ratio: {cls['pb_ratio']:.2f}")
                report_lines.append(f"Beta: {cls['beta']:.2f}")
                if cls['index_membership']:
                    report_lines.append(f"Indices: {', '.join(cls['index_membership'])}")
                report_lines.append("")

            # Signals
            signals = analysis['signals']
            report_lines.append("TECHNICAL SIGNALS")
            report_lines.append("-" * 70)
            report_lines.append(f"Trend: {signals['trend']}")
            report_lines.append(f"Momentum: {signals['momentum']}")
            report_lines.append(f"Volatility: {signals['volatility']}")
            report_lines.append(f"Volume: {signals['volume']}")
            report_lines.append("")

            # Key indicators
            indicators = analysis['indicators']
            report_lines.append("KEY INDICATORS")
            report_lines.append("-" * 70)

            # Show important indicators
            important = ['RSI_14', 'MACD', 'MACD_SIGNAL', 'ATR_14', 'VWAP',
                        'BB_UPPER', 'BB_LOWER', 'ADX', 'OBV']

            for ind in important:
                if ind in indicators and indicators[ind] is not None:
                    try:
                        value = float(indicators[ind])
                        report_lines.append(f"{ind}: {value:.2f}")
                    except (ValueError, TypeError):
                        pass

            report_lines.append("")
            report_lines.append("=" * 70)

            return "\n".join(report_lines)

    def export_analysis(
        self,
        analysis: Dict[str, Any],
        filename: str,
        format: Literal['csv', 'json', 'excel'] = 'csv'
    ):
        """
        Export analysis to file

        Args:
            analysis: Stock analysis dictionary
            filename: Output filename
            format: Export format
        """
        df = analysis['full_dataframe']

        if format == 'csv':
            df.to_csv(filename)
            logger.info(f"Exported to {filename}")

        elif format == 'json':
            df.to_json(filename, orient='records', indent=2)
            logger.info(f"Exported to {filename}")

        elif format == 'excel':
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Price Data')

                # Add summary sheet
                summary_data = {
                    'Metric': ['Symbol', 'Latest Price', 'Change %', 'Trend', 'Momentum'],
                    'Value': [
                        analysis['symbol'],
                        analysis['latest_price'],
                        analysis['price_change_pct'],
                        analysis['signals']['trend'],
                        analysis['signals']['momentum']
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            logger.info(f"Exported to {filename}")

    # ==================== HELPER METHODS ====================

    def _calculate_indicator(self, ti: TechnicalIndicators, indicator: str):
        """Calculate a specific indicator"""
        indicator_upper = indicator.upper()

        if 'RSI' in indicator_upper:
            period = int(indicator_upper.split('_')[1]) if '_' in indicator_upper else 14
            ti.rsi(period)
        elif 'MACD' in indicator_upper:
            ti.macd()
        elif 'BB' in indicator_upper or 'BOLLINGER' in indicator_upper:
            ti.bollinger_bands()
        elif 'ATR' in indicator_upper:
            period = int(indicator_upper.split('_')[1]) if '_' in indicator_upper else 14
            ti.atr(period)
        elif 'VWAP' in indicator_upper:
            ti.vwap()
        elif 'STOCH' in indicator_upper:
            ti.stochastic()
        elif 'ADX' in indicator_upper:
            ti.adx()
        elif 'OBV' in indicator_upper:
            ti.obv()
        # Add more as needed

    def _meets_criteria(self, analysis: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if analysis meets screening criteria"""
        indicators = analysis.get('indicators', {})
        signals = analysis.get('signals', {})
        classification = analysis.get('classification', {})

        for key, value in criteria.items():
            # RSI criteria
            if key == 'rsi_min' and indicators.get('RSI_14', 0) < value:
                return False
            if key == 'rsi_max' and indicators.get('RSI_14', 100) > value:
                return False

            # Trend criteria
            if key == 'trend' and signals.get('trend') != value:
                return False

            # Momentum criteria
            if key == 'momentum' and signals.get('momentum') != value:
                return False

            # Volume spike
            if key == 'volume_spike' and value and signals.get('volume') != 'HIGH':
                return False

            # Classification criteria
            if classification:
                if key == 'market_cap' and classification.get('market_cap_category') != value:
                    return False
                if key == 'sector' and classification.get('sector') != value:
                    return False
                if key == 'liquidity' and classification.get('liquidity') != value:
                    return False

        return True


# Convenience functions
def analyze_stock(
    symbol: str,
    period: str = '1mo',
    interval: str = '1d',
    exchange: str = 'NSE'
) -> Optional[Dict[str, Any]]:
    """Quick function to analyze a single stock"""
    engine = IndicatorEngine()
    return engine.analyze_stock(symbol, period, interval, exchange)


def analyze_stocks(
    symbols: List[str],
    period: str = '1mo',
    interval: str = '1d',
    exchange: str = 'NSE'
) -> Dict[str, Dict[str, Any]]:
    """Quick function to analyze multiple stocks"""
    engine = IndicatorEngine()
    return engine.analyze_multiple_stocks(symbols, period, interval, exchange)


def screen_stocks(
    symbols: List[str],
    criteria: Dict[str, Any],
    period: str = '1mo',
    exchange: str = 'NSE'
) -> Dict[str, Dict[str, Any]]:
    """Quick function to screen stocks"""
    engine = IndicatorEngine()
    return engine.screen_stocks(symbols, criteria, period, '1d', exchange)
