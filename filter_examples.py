"""
Filter System Examples
Demonstrates individual and combined filter usage
"""

from filter_manager import FilterManager, FilterPresets, CombineLogic
from filter_api import FilterAPI, quick_screen, preset_screen
from stock_filters import FilterRegistry, FilterType
from indicator_engine import IndicatorEngine


def example_1_individual_filters():
    """Example 1: Apply filters individually"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Individual Filter Application")
    print("="*70)

    manager = FilterManager()
    engine = IndicatorEngine()

    # Add individual filters
    manager.add_filter('rsi', {'min': 30, 'max': 70})
    manager.add_filter('price', {'min': 100, 'max': 5000})
    manager.add_filter('market_cap', {'categories': ['LARGE']})

    # Analyze a stock
    analysis = engine.analyze_stock('RELIANCE', period='1mo')

    if analysis:
        print(f"\nStock: {analysis['symbol']}")
        print(f"Price: ₹{analysis['latest_price']:.2f}")

        # Test each filter individually
        print("\nIndividual Filter Results:")
        print(f"  RSI Filter: {'✓ PASS' if manager.apply_individual(analysis, 'RSI Filter') else '✗ FAIL'}")
        print(f"  Price Filter: {'✓ PASS' if manager.apply_individual(analysis, 'Price Filter') else '✗ FAIL'}")
        print(f"  Market Cap Filter: {'✓ PASS' if manager.apply_individual(analysis, 'Market Cap Filter') else '✗ FAIL'}")

        # Test all filters combined
        print(f"\nCombined (AND): {'✓ PASS' if manager.apply_all(analysis, CombineLogic.AND) else '✗ FAIL'}")
        print(f"Combined (OR): {'✓ PASS' if manager.apply_all(analysis, CombineLogic.OR) else '✗ FAIL'}")


def example_2_filter_groups():
    """Example 2: Organize filters into groups"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Filter Groups")
    print("="*70)

    manager = FilterManager()

    # Group 1: Technical filters
    manager.add_group(
        'Technical Analysis',
        [
            {'name': 'rsi', 'parameters': {'min': 40, 'max': 60}},
            {'name': 'trend', 'parameters': {'direction': 'BULLISH'}},
            {'name': 'macd', 'parameters': {'signal': 'bullish'}}
        ],
        logic=CombineLogic.AND
    )

    # Group 2: Classification filters
    manager.add_group(
        'Stock Quality',
        [
            {'name': 'market_cap', 'parameters': {'categories': ['LARGE']}},
            {'name': 'liquidity', 'parameters': {'levels': ['HIGH']}},
        ],
        logic=CombineLogic.AND
    )

    print("\nFilter Groups Created:")
    info = manager.get_info()
    for group in info['groups']:
        print(f"\n  {group['name']} ({group['logic']})")
        print(f"    Filters: {group['filter_count']}")
        for f in group['filters']:
            print(f"      - {f['name']}")


def example_3_enable_disable():
    """Example 3: Enable/Disable filters dynamically"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Enable/Disable Filters")
    print("="*70)

    manager = FilterManager()
    engine = IndicatorEngine()

    # Add multiple filters
    manager.add_filter('rsi', {'min': 30, 'max': 70})
    manager.add_filter('price', {'min': 100, 'max': 5000})
    manager.add_filter('trend', {'direction': 'BULLISH'})

    analysis = engine.analyze_stock('TCS', period='1mo')

    if analysis:
        print(f"\nStock: {analysis['symbol']}")

        # Test with all filters
        print("\nWith all filters enabled:")
        result1 = manager.apply_all(analysis)
        print(f"  Result: {'✓ PASS' if result1 else '✗ FAIL'}")

        # Disable trend filter
        manager.disable_filter('Trend Filter')
        print("\nWith trend filter disabled:")
        result2 = manager.apply_all(analysis)
        print(f"  Result: {'✓ PASS' if result2 else '✗ FAIL'}")

        # Enable it back
        manager.enable_filter('Trend Filter')
        print("\nWith trend filter re-enabled:")
        result3 = manager.apply_all(analysis)
        print(f"  Result: {'✓ PASS' if result3 else '✗ FAIL'}")


def example_4_filter_by_type():
    """Example 4: Apply filters by type"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Filter by Type")
    print("="*70)

    manager = FilterManager()
    engine = IndicatorEngine()

    # Add filters of different types
    manager.add_filter('rsi', {'min': 30, 'max': 70})  # Technical
    manager.add_filter('macd', {'signal': 'bullish'})  # Technical
    manager.add_filter('price', {'min': 100, 'max': 5000})  # Price
    manager.add_filter('market_cap', {'categories': ['LARGE']})  # Classification

    analysis = engine.analyze_stock('INFY', period='1mo')

    if analysis:
        print(f"\nStock: {analysis['symbol']}")

        # Apply only technical filters
        print("\nTechnical filters only:")
        result_tech = manager.apply_by_type(analysis, FilterType.TECHNICAL)
        print(f"  Result: {'✓ PASS' if result_tech else '✗ FAIL'}")

        # Apply only price filters
        print("\nPrice filters only:")
        result_price = manager.apply_by_type(analysis, FilterType.PRICE)
        print(f"  Result: {'✓ PASS' if result_price else '✗ FAIL'}")

        # Apply only classification filters
        print("\nClassification filters only:")
        result_class = manager.apply_by_type(analysis, FilterType.CLASSIFICATION)
        print(f"  Result: {'✓ PASS' if result_class else '✗ FAIL'}")


def example_5_update_parameters():
    """Example 5: Update filter parameters dynamically"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Update Filter Parameters")
    print("="*70)

    manager = FilterManager()
    engine = IndicatorEngine()

    # Add RSI filter
    manager.add_filter('rsi', {'min': 30, 'max': 70})

    analysis = engine.analyze_stock('WIPRO', period='1mo')

    if analysis:
        rsi_value = analysis['indicators'].get('RSI_14', 0)
        print(f"\nStock: {analysis['symbol']}")
        print(f"RSI Value: {rsi_value:.2f}")

        # Test with original parameters
        print("\nWith RSI range 30-70:")
        result1 = manager.apply_individual(analysis, 'RSI Filter')
        print(f"  Result: {'✓ PASS' if result1 else '✗ FAIL'}")

        # Update to tighter range
        manager.update_filter_parameters('RSI Filter', min=40, max=60)
        print("\nWith RSI range 40-60:")
        result2 = manager.apply_individual(analysis, 'RSI Filter')
        print(f"  Result: {'✓ PASS' if result2 else '✗ FAIL'}")

        # Update to oversold range
        manager.update_filter_parameters('RSI Filter', min=0, max=35)
        print("\nWith RSI range 0-35 (oversold):")
        result3 = manager.apply_individual(analysis, 'RSI Filter')
        print(f"  Result: {'✓ PASS' if result3 else '✗ FAIL'}")


def example_6_screen_multiple_stocks():
    """Example 6: Screen multiple stocks"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Screen Multiple Stocks")
    print("="*70)

    manager = FilterManager()
    engine = IndicatorEngine()

    # Setup filters
    manager.add_filter('rsi', {'min': 30, 'max': 70})
    manager.add_filter('trend', {'direction': 'BULLISH'})
    manager.add_filter('liquidity', {'levels': ['HIGH']})

    # List of stocks to screen
    symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

    # Analyze all stocks
    analyses = engine.analyze_multiple_stocks(symbols, period='1mo')

    # Screen with filters
    passed = manager.screen_stocks(analyses)

    print(f"\nScreened {len(analyses)} stocks")
    print(f"Passed: {len(passed)} stocks\n")

    for symbol, analysis in passed.items():
        print(f"✓ {symbol}")
        print(f"    Price: ₹{analysis['latest_price']:.2f} ({analysis['price_change_pct']:.2f}%)")
        print(f"    RSI: {analysis['indicators'].get('RSI_14', 0):.2f}")
        print(f"    Trend: {analysis['signals']['trend']}")


def example_7_detailed_screening():
    """Example 7: Screen with detailed results"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Detailed Screening Results")
    print("="*70)

    manager = FilterManager()
    engine = IndicatorEngine()

    # Setup filters
    manager.add_filter('rsi', {'min': 40, 'max': 80})
    manager.add_filter('volume', {'state': 'HIGH'})

    symbols = ['TCS', 'INFY', 'WIPRO']
    analyses = engine.analyze_multiple_stocks(symbols, period='1mo')

    # Get detailed results
    results = manager.screen_with_details(analyses)

    print(f"\nScreening Results:")
    print(f"  Total: {results['filter_summary']['total_stocks']}")
    print(f"  Passed: {results['filter_summary']['passed']}")
    print(f"  Failed: {results['filter_summary']['failed']}")
    print(f"  Pass Rate: {results['filter_summary']['pass_rate']:.1f}%")

    print("\nPassed Stocks:")
    for symbol, data in results['passed'].items():
        print(f"\n  {symbol}:")
        print(f"    Price: ₹{data['analysis']['latest_price']:.2f}")
        print(f"    Filter Results:")
        for filter_name, filter_result in data['filter_results']['individual_filters'].items():
            status = "✓" if filter_result['passed'] else "✗"
            print(f"      {status} {filter_name}: {filter_result['passed']}")


def example_8_using_presets():
    """Example 8: Use preset filter configurations"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Using Filter Presets")
    print("="*70)

    # Momentum trading preset
    print("\n1. Momentum Trading Preset:")
    manager1 = FilterPresets.momentum_trading()
    print(f"   Filters: {len(manager1.filters)}")
    for f in manager1.filters:
        print(f"     - {f.config.name}")

    # Value investing preset
    print("\n2. Value Investing Preset:")
    manager2 = FilterPresets.value_investing()
    print(f"   Filters: {len(manager2.filters)}")
    for f in manager2.filters:
        print(f"     - {f.config.name}")

    # Oversold bounce preset
    print("\n3. Oversold Bounce Preset:")
    manager3 = FilterPresets.oversold_bounce()
    print(f"   Filters: {len(manager3.filters)}")
    for f in manager3.filters:
        print(f"     - {f.config.name}")


def example_9_web_api():
    """Example 9: Using Web API"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Web API Usage")
    print("="*70)

    api = FilterAPI()

    # Get available filters
    filters = api.get_available_filters()
    print(f"\nAvailable Filters: {filters['count']}")
    print(f"Filter Types: {', '.join(filters['types'])}")

    # Create a manager
    api.create_manager('my_manager')

    # Add filters via API
    api.add_filter('my_manager', 'rsi', {'min': 30, 'max': 70})
    api.add_filter('my_manager', 'trend', {'direction': 'BULLISH'})

    # Screen stocks
    results = api.screen_stocks(
        'my_manager',
        ['RELIANCE', 'TCS', 'INFY'],
        period='1mo'
    )

    print(f"\nScreening Results:")
    print(f"  Total: {results['summary']['total_stocks']}")
    print(f"  Passed: {results['summary']['passed']}")
    print(f"  Pass Rate: {results['summary']['pass_rate']:.1f}%")

    print("\nPassed Stocks:")
    for symbol, data in results['passed'].items():
        print(f"  {symbol}: ₹{data['price']:.2f} ({data['change_pct']:.2f}%) - {data['trend']}")


def example_10_quick_screen():
    """Example 10: Quick screening without manager"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Quick Screen")
    print("="*70)

    # Define filters inline
    filters = [
        {'name': 'rsi', 'parameters': {'min': 35, 'max': 65}},
        {'name': 'volume', 'parameters': {'state': 'HIGH'}},
        {'name': 'liquidity', 'parameters': {'levels': ['HIGH']}}
    ]

    # Quick screen
    results = quick_screen(
        symbols=['RELIANCE', 'TCS', 'INFY', 'HDFCBANK'],
        filters=filters,
        period='1mo',
        logic='AND'
    )

    print(f"\nQuick Screen Results:")
    print(f"  Passed: {results['summary']['passed']}/{results['summary']['total_stocks']}")

    for symbol, data in results['passed'].items():
        print(f"  ✓ {symbol}: ₹{data['price']:.2f} - {data['trend']}")


def example_11_preset_screen():
    """Example 11: Screen with preset"""
    print("\n" + "="*70)
    print("EXAMPLE 11: Preset Screening")
    print("="*70)

    # Screen using preset
    results = preset_screen(
        preset='momentum_trading',
        symbols=['RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HCLTECH'],
        period='1mo'
    )

    print(f"\nMomentum Trading Screen:")
    print(f"  Passed: {results['summary']['passed']}/{results['summary']['total_stocks']}")

    for symbol, data in results['passed'].items():
        print(f"  ✓ {symbol}: ₹{data['price']:.2f} ({data['change_pct']:.2f}%)")
        print(f"      Trend: {data['trend']}, Momentum: {data['momentum']}")


def example_12_sector_screening():
    """Example 12: Screen by sector"""
    print("\n" + "="*70)
    print("EXAMPLE 12: Sector Screening")
    print("="*70)

    api = FilterAPI()

    # Create manager for IT sector
    api.create_manager('it_screen')

    # Add filters
    api.add_filter('it_screen', 'rsi', {'min': 40, 'max': 60})
    api.add_filter('it_screen', 'trend', {'direction': 'BULLISH'})

    # Screen IT sector
    results = api.screen_by_type(
        'it_screen',
        stock_type='IT',
        period='1mo',
        limit=10
    )

    print(f"\nIT Sector Screening:")
    print(f"  Type: {results['stock_type']}")
    print(f"  Passed: {results['summary']['passed']}/{results['summary']['total_stocks']}")

    print("\nPassed IT Stocks:")
    for symbol, data in results['passed'].items():
        print(f"  {symbol}: ₹{data['price']:.2f} - {data['sector']}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FILTER SYSTEM EXAMPLES")
    print("="*70)

    examples = [
        ("Individual Filters", example_1_individual_filters),
        ("Filter Groups", example_2_filter_groups),
        ("Enable/Disable Filters", example_3_enable_disable),
        ("Filter by Type", example_4_filter_by_type),
        ("Update Parameters", example_5_update_parameters),
        ("Screen Multiple Stocks", example_6_screen_multiple_stocks),
        ("Detailed Screening", example_7_detailed_screening),
        ("Using Presets", example_8_using_presets),
        ("Web API", example_9_web_api),
        ("Quick Screen", example_10_quick_screen),
        ("Preset Screen", example_11_preset_screen),
        ("Sector Screening", example_12_sector_screening),
    ]

    print("\n\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\n" + "-"*70)
    choice = input("\nEnter example number (or 'all' for all): ").strip().lower()

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
        print("Running Example 1 by default...")
        example_1_individual_filters()

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
