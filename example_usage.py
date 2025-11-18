"""
Example Usage Script
Demonstrates how to use the complete technical indicators system
"""

import pandas as pd
from datetime import datetime

# Import all modules
from technical_indicators import TechnicalIndicators, calculate_indicators
from stock_data_fetcher import StockDataFetcher, fetch_stock, fetch_stocks
from stock_classifier import StockClassifier, classify_stock, get_stocks_by_type
from indicator_engine import IndicatorEngine, analyze_stock, analyze_stocks, screen_stocks


def example_1_basic_indicators():
    """Example 1: Calculate indicators for a single stock"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Technical Indicators")
    print("="*70)

    # Fetch stock data
    df = fetch_stock('RELIANCE', period='3mo', interval='1d', exchange='NSE')

    if df is not None:
        # Calculate all indicators
        ti = TechnicalIndicators(df)
        result_df = ti.calculate_all()

        # Display last 5 rows
        print("\nLast 5 rows with indicators:")
        print(result_df[['Close', 'RSI_14', 'MACD', 'VWAP', 'ATR_14']].tail())

        # Get latest signals
        signals = ti.get_latest_signals()
        print("\nLatest Signals:")
        print(f"  Trend: {signals['signals']['trend']}")
        print(f"  Momentum: {signals['signals']['momentum']}")
        print(f"  Volatility: {signals['signals']['volatility']}")
        print(f"  Volume: {signals['signals']['volume']}")


def example_2_stock_classification():
    """Example 2: Classify stocks"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Stock Classification")
    print("="*70)

    # Classify a single stock
    classification = classify_stock('TCS', exchange='NSE')

    if classification:
        print(f"\nStock: {classification.symbol}")
        print(f"Name: {classification.name}")
        print(f"Sector: {classification.sector}")
        print(f"Industry: {classification.industry}")
        print(f"Market Cap: ₹{classification.market_cap:.2f} Cr ({classification.market_cap_category})")
        print(f"Liquidity: {classification.liquidity}")
        print(f"Volatility: {classification.volatility}")
        print(f"Beta: {classification.beta:.2f}")
        print(f"PE Ratio: {classification.pe_ratio:.2f}")
        print(f"Indices: {', '.join(classification.index_membership)}")


def example_3_fetch_multiple_stocks():
    """Example 3: Fetch and analyze multiple stocks"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Stocks Analysis")
    print("="*70)

    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

    # Fetch data for multiple stocks
    stock_data = fetch_stocks(symbols, period='1mo', interval='1d', exchange='NSE')

    print(f"\nFetched data for {len(stock_data)} stocks")

    # Calculate indicators for each
    for symbol, df in stock_data.items():
        ti = TechnicalIndicators(df)
        signals = ti.get_latest_signals()

        latest_price = df['Close'].iloc[-1]
        rsi = signals['indicators'].get('RSI_14', 0)

        print(f"\n{symbol}:")
        print(f"  Price: ₹{latest_price:.2f}")
        print(f"  RSI: {rsi:.2f}")
        print(f"  Trend: {signals['signals']['trend']}")
        print(f"  Momentum: {signals['signals']['momentum']}")


def example_4_stocks_by_type():
    """Example 4: Get stocks by specific type"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Stocks by Type")
    print("="*70)

    # Get IT sector stocks
    it_stocks = get_stocks_by_type('IT', exchange='NSE')

    print(f"\nIT Sector Stocks ({len(it_stocks)}):")
    for symbol, classification in list(it_stocks.items())[:5]:  # Show first 5
        print(f"  {symbol}: {classification.name} - ₹{classification.market_cap:.2f} Cr")


def example_5_complete_analysis():
    """Example 5: Complete stock analysis using IndicatorEngine"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Stock Analysis")
    print("="*70)

    # Analyze a single stock
    analysis = analyze_stock('RELIANCE', period='3mo', interval='1d', exchange='NSE')

    if analysis:
        # Generate text report
        engine = IndicatorEngine()
        report = engine.generate_report(analysis, format='text')
        print(report)


def example_6_compare_stocks():
    """Example 6: Compare multiple stocks"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Stock Comparison")
    print("="*70)

    engine = IndicatorEngine()

    # Compare stocks
    symbols = ['RELIANCE', 'TCS', 'INFY']
    comparison = engine.compare_stocks(
        symbols,
        metrics=['latest_price', 'price_change_pct', 'RSI_14', 'trend', 'momentum', 'sector'],
        period='1mo',
        interval='1d',
        exchange='NSE'
    )

    print("\nStock Comparison:")
    print(comparison.to_string(index=False))


def example_7_stock_screening():
    """Example 7: Screen stocks based on criteria"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Stock Screening")
    print("="*70)

    # Screen for oversold stocks in IT sector
    criteria = {
        'rsi_max': 40,  # RSI below 40 (oversold)
        'trend': 'BULLISH',  # Overall trend is bullish
        'sector': 'Technology',  # IT sector
        'liquidity': 'HIGH'  # High liquidity
    }

    symbols = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']

    screened = screen_stocks(symbols, criteria, period='1mo', exchange='NSE')

    print(f"\nScreened {len(screened)} stocks matching criteria:")
    for symbol, analysis in screened.items():
        print(f"\n{symbol}:")
        print(f"  Price: ₹{analysis['latest_price']:.2f}")
        print(f"  RSI: {analysis['indicators'].get('RSI_14', 0):.2f}")
        print(f"  Trend: {analysis['signals']['trend']}")
        print(f"  Momentum: {analysis['signals']['momentum']}")


def example_8_sector_analysis():
    """Example 8: Analyze entire sector"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Sector Analysis")
    print("="*70)

    engine = IndicatorEngine()

    # Analyze IT sector (limit to 5 stocks for demo)
    analysis_results = engine.analyze_by_stock_type(
        stock_type='IT',
        period='1mo',
        interval='1d',
        exchange='NSE',
        limit=5
    )

    print(f"\nAnalyzed {len(analysis_results)} IT stocks:")

    # Summary statistics
    bullish_count = 0
    total_count = len(analysis_results)

    for symbol, analysis in analysis_results.items():
        if analysis['signals']['trend'] == 'BULLISH':
            bullish_count += 1

        print(f"\n{symbol}:")
        print(f"  Price: ₹{analysis['latest_price']:.2f} ({analysis['price_change_pct']:.2f}%)")
        print(f"  Signals: {analysis['signals']}")

    print(f"\n\nSector Summary:")
    print(f"  Bullish stocks: {bullish_count}/{total_count} ({bullish_count/total_count*100:.1f}%)")


def example_9_intraday_analysis():
    """Example 9: Intraday analysis with 15-minute data"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Intraday Analysis (15-min)")
    print("="*70)

    # Fetch intraday data
    df = fetch_stock('RELIANCE', period='5d', interval='15m', exchange='NSE')

    if df is not None:
        # Calculate indicators
        ti = TechnicalIndicators(df)
        ti.rsi(14)
        ti.macd()
        ti.vwap()
        ti.bollinger_bands(20)

        # Get latest values
        latest = df.tail(1).iloc[0]
        rsi = ti.results['RSI_14'].iloc[-1] if 'RSI_14' in ti.results else 0
        vwap = ti.results['VWAP'].iloc[-1] if 'VWAP' in ti.results else 0

        print(f"\nRELIANCE - Latest 15-min candle:")
        print(f"  Time: {df.index[-1]}")
        print(f"  Close: ₹{latest['Close']:.2f}")
        print(f"  RSI: {rsi:.2f}")
        print(f"  VWAP: ₹{vwap:.2f}")
        print(f"  Price vs VWAP: {'Above' if latest['Close'] > vwap else 'Below'}")


def example_10_custom_indicators():
    """Example 10: Calculate specific custom indicators"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Custom Indicator Selection")
    print("="*70)

    # Fetch data
    df = fetch_stock('TCS', period='3mo', interval='1d', exchange='NSE')

    if df is not None:
        # Calculate only specific indicators
        ti = TechnicalIndicators(df)

        # Custom set of indicators
        ti.rsi(14)
        ti.rsi(21)  # Different period
        ti.macd()
        ti.adx(14)
        ti.bollinger_bands(20)
        ti.supertrend()
        ti.obv()

        print("\nCustom Indicators Calculated:")
        print(f"  RSI(14): {ti.results.get('RSI_14', pd.Series()).iloc[-1]:.2f}")
        print(f"  RSI(21): {ti.results.get('RSI_21', pd.Series()).iloc[-1]:.2f}")
        print(f"  MACD: {ti.results.get('MACD', pd.Series()).iloc[-1]:.2f}")
        print(f"  ADX: {ti.results.get('ADX', pd.Series()).iloc[-1]:.2f}")
        print(f"  OBV: {ti.results.get('OBV', pd.Series()).iloc[-1]:.0f}")


def example_11_export_analysis():
    """Example 11: Export analysis to file"""
    print("\n" + "="*70)
    print("EXAMPLE 11: Export Analysis")
    print("="*70)

    engine = IndicatorEngine()

    # Analyze stock
    analysis = engine.analyze_stock('INFY', period='3mo', interval='1d', exchange='NSE')

    if analysis:
        # Export to different formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # CSV export
        csv_file = f'analysis_INFY_{timestamp}.csv'
        engine.export_analysis(analysis, csv_file, format='csv')
        print(f"✓ Exported to CSV: {csv_file}")

        # JSON export
        json_file = f'analysis_INFY_{timestamp}.json'
        engine.export_analysis(analysis, json_file, format='json')
        print(f"✓ Exported to JSON: {json_file}")

        # Excel export (if openpyxl is installed)
        try:
            excel_file = f'analysis_INFY_{timestamp}.xlsx'
            engine.export_analysis(analysis, excel_file, format='excel')
            print(f"✓ Exported to Excel: {excel_file}")
        except ImportError:
            print("! Excel export requires openpyxl package")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("TECHNICAL INDICATORS SYSTEM - EXAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates the complete technical indicators system")
    print("for analyzing real stocks with various indicators and classifications.")

    examples = [
        ("Basic Indicators", example_1_basic_indicators),
        ("Stock Classification", example_2_stock_classification),
        ("Multiple Stocks", example_3_fetch_multiple_stocks),
        ("Stocks by Type", example_4_stocks_by_type),
        ("Complete Analysis", example_5_complete_analysis),
        ("Compare Stocks", example_6_compare_stocks),
        ("Stock Screening", example_7_stock_screening),
        ("Sector Analysis", example_8_sector_analysis),
        ("Intraday Analysis", example_9_intraday_analysis),
        ("Custom Indicators", example_10_custom_indicators),
        ("Export Analysis", example_11_export_analysis),
    ]

    print("\n\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\n" + "-"*70)
    choice = input("\nEnter example number to run (or 'all' for all examples): ").strip().lower()

    if choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n⚠ Error in {name}: {str(e)}")
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            try:
                examples[idx][1]()
            except Exception as e:
                print(f"\n⚠ Error: {str(e)}")
        else:
            print("Invalid example number")
    else:
        print("Running Example 1 (Basic Indicators) by default...")
        example_1_basic_indicators()

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
