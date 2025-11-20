# Advanced Stock Filtering System

A flexible, modular filtering system that allows you to apply filters individually or in combination to screen stocks based on technical, fundamental, and classification criteria.

## 🎯 Key Features

### **Individual Filter Application**
- Apply each filter separately
- Test filters one by one
- Enable/disable filters dynamically
- Update filter parameters on the fly

### **Combined Filter Application**
- Combine filters with AND/OR logic
- Create filter groups
- Mix different filter types
- Hierarchical filter organization

### **Filter Organization by Type**
- **Technical**: RSI, MACD, Bollinger Bands, ADX, VWAP, Stochastic, etc.
- **Price**: Price range, price change percentage
- **Volume**: Volume state, volume spikes
- **Classification**: Market cap, sector, liquidity, volatility
- **Fundamental**: P/E ratio, P/B ratio, Beta

### **Web-Friendly API**
- REST-style JSON API
- Easy integration with web frontends
- Django-compatible
- Session-based filter management

## 📦 Modules

### 1. `stock_filters.py`
Individual filter classes - 20+ filters organized by type.

**Available Filters:**

**Technical Filters:**
- `rsi` - Relative Strength Index
- `macd` - MACD signal
- `bollinger_bands` - Bollinger Bands position
- `atr` - Average True Range (volatility)
- `vwap` - VWAP position
- `stochastic` - Stochastic oscillator
- `adx` - Trend strength
- `trend` - Trend direction
- `momentum` - Momentum state

**Price Filters:**
- `price` - Price range
- `price_change` - Price change percentage

**Volume Filters:**
- `volume` - Volume state
- `volume_spike` - Volume spike detection

**Classification Filters:**
- `market_cap` - Market capitalization
- `sector` - Sector filter
- `liquidity` - Liquidity level
- `volatility` - Volatility level
- `index_membership` - Index membership (NIFTY50, etc.)

**Fundamental Filters:**
- `pe_ratio` - P/E ratio
- `pb_ratio` - P/B ratio
- `beta` - Market risk (Beta)

### 2. `filter_manager.py`
Filter management and composition engine.

**Key Classes:**
- `FilterManager` - Main filter management class
- `FilterGroup` - Group filters with AND/OR logic
- `FilterPresets` - Pre-configured filter setups

**Presets Available:**
- `momentum_trading` - For momentum strategies
- `value_investing` - For value investing
- `oversold_bounce` - For oversold bounce plays
- `breakout_trading` - For breakout strategies
- `swing_trading` - For swing trading
- `conservative` - Low-risk conservative filters

### 3. `filter_api.py`
Web-friendly API for filter operations.

**API Methods:**
- `get_available_filters()` - List all filters
- `create_manager()` - Create filter manager
- `add_filter()` - Add individual filter
- `add_filter_group()` - Add filter group
- `enable_filter()` / `disable_filter()` - Toggle filters
- `update_filter_parameters()` - Update filter settings
- `screen_stocks()` - Screen stocks with filters
- `test_filter_on_stock()` - Test filters on single stock

### 4. `filter_examples.py`
12 comprehensive examples demonstrating all features.

## 🚀 Quick Start

### Example 1: Individual Filter Application

```python
from filter_manager import FilterManager
from indicator_engine import IndicatorEngine

# Create manager
manager = FilterManager()
engine = IndicatorEngine()

# Add individual filters
manager.add_filter('rsi', {'min': 30, 'max': 70})
manager.add_filter('price', {'min': 100, 'max': 5000})
manager.add_filter('trend', {'direction': 'BULLISH'})

# Analyze a stock
analysis = engine.analyze_stock('RELIANCE', period='1mo')

# Test each filter individually
print(f"RSI Filter: {manager.apply_individual(analysis, 'RSI Filter')}")
print(f"Price Filter: {manager.apply_individual(analysis, 'Price Filter')}")
print(f"Trend Filter: {manager.apply_individual(analysis, 'Trend Filter')}")

# Test all filters combined
print(f"All filters (AND): {manager.apply_all(analysis, CombineLogic.AND)}")
print(f"All filters (OR): {manager.apply_all(analysis, CombineLogic.OR)}")
```

### Example 2: Filter Groups

```python
from filter_manager import FilterManager, CombineLogic

manager = FilterManager()

# Create a technical analysis group
manager.add_group(
    'Technical Analysis',
    [
        {'name': 'rsi', 'parameters': {'min': 40, 'max': 60}},
        {'name': 'trend', 'parameters': {'direction': 'BULLISH'}},
        {'name': 'macd', 'parameters': {'signal': 'bullish'}}
    ],
    logic=CombineLogic.AND
)

# Create a quality group
manager.add_group(
    'Stock Quality',
    [
        {'name': 'market_cap', 'parameters': {'categories': ['LARGE']}},
        {'name': 'liquidity', 'parameters': {'levels': ['HIGH']}},
    ],
    logic=CombineLogic.AND
)

# Screen stocks
passed = manager.screen_stocks(analyses)
```

### Example 3: Enable/Disable Filters

```python
manager = FilterManager()

# Add filters
manager.add_filter('rsi', {'min': 30, 'max': 70})
manager.add_filter('trend', {'direction': 'BULLISH'})

# Test with all filters
result1 = manager.apply_all(analysis)

# Disable one filter
manager.disable_filter('Trend Filter')
result2 = manager.apply_all(analysis)

# Re-enable it
manager.enable_filter('Trend Filter')
result3 = manager.apply_all(analysis)
```

### Example 4: Update Parameters Dynamically

```python
manager = FilterManager()

# Add RSI filter
manager.add_filter('rsi', {'min': 30, 'max': 70})

# Test with original parameters
result1 = manager.apply_individual(analysis, 'RSI Filter')

# Update to tighter range
manager.update_filter_parameters('RSI Filter', min=40, max=60)
result2 = manager.apply_individual(analysis, 'RSI Filter')

# Update to oversold range
manager.update_filter_parameters('RSI Filter', min=0, max=35)
result3 = manager.apply_individual(analysis, 'RSI Filter')
```

### Example 5: Screen Multiple Stocks

```python
from filter_manager import FilterManager
from indicator_engine import IndicatorEngine

manager = FilterManager()
engine = IndicatorEngine()

# Setup filters
manager.add_filter('rsi', {'min': 30, 'max': 70})
manager.add_filter('trend', {'direction': 'BULLISH'})
manager.add_filter('liquidity', {'levels': ['HIGH']})

# Analyze stocks
symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
analyses = engine.analyze_multiple_stocks(symbols, period='1mo')

# Screen with filters
passed = manager.screen_stocks(analyses)

print(f"Passed: {len(passed)}/{len(analyses)} stocks")
for symbol, analysis in passed.items():
    print(f"✓ {symbol}: ₹{analysis['latest_price']:.2f}")
```

### Example 6: Using Presets

```python
from filter_manager import FilterPresets

# Use momentum trading preset
manager = FilterPresets.momentum_trading()

# Or value investing preset
manager = FilterPresets.value_investing()

# Or create custom from preset
manager = FilterPresets.oversold_bounce()
# Then modify it
manager.update_filter_parameters('RSI Filter', min=25, max=30)
```

### Example 7: Web API Usage

```python
from filter_api import FilterAPI

api = FilterAPI()

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

print(f"Passed: {results['summary']['passed']}")
for symbol, data in results['passed'].items():
    print(f"{symbol}: ₹{data['price']:.2f}")
```

### Example 8: Quick Screen (No Manager Needed)

```python
from filter_api import quick_screen

# Define filters inline
filters = [
    {'name': 'rsi', 'parameters': {'min': 35, 'max': 65}},
    {'name': 'volume', 'parameters': {'state': 'HIGH'}},
    {'name': 'liquidity', 'parameters': {'levels': ['HIGH']}}
]

# Quick screen
results = quick_screen(
    symbols=['RELIANCE', 'TCS', 'INFY'],
    filters=filters,
    logic='AND'
)
```

### Example 9: Preset Screen

```python
from filter_api import preset_screen

# Screen using preset
results = preset_screen(
    preset='momentum_trading',
    symbols=['RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HCLTECH']
)
```

### Example 10: Filter by Type

```python
manager = FilterManager()

# Add multiple filter types
manager.add_filter('rsi', {'min': 30, 'max': 70})  # Technical
manager.add_filter('price', {'min': 100})          # Price
manager.add_filter('market_cap', {'categories': ['LARGE']})  # Classification

# Apply only technical filters
result_tech = manager.apply_by_type(analysis, FilterType.TECHNICAL)

# Apply only price filters
result_price = manager.apply_by_type(analysis, FilterType.PRICE)
```

## 📊 Filter Configuration Reference

### RSI Filter
```python
{
    'name': 'rsi',
    'parameters': {
        'min': 30,      # Minimum RSI value
        'max': 70,      # Maximum RSI value
        'period': 14    # RSI period
    }
}
```

### MACD Filter
```python
{
    'name': 'macd',
    'parameters': {
        'signal': 'bullish'  # 'bullish', 'bearish', or 'any'
    }
}
```

### Bollinger Bands Filter
```python
{
    'name': 'bollinger_bands',
    'parameters': {
        'position': 'middle',  # 'upper', 'lower', 'middle', 'outside_upper', 'outside_lower'
        'period': 20
    }
}
```

### Price Filter
```python
{
    'name': 'price',
    'parameters': {
        'min': 100,    # Minimum price
        'max': 5000    # Maximum price
    }
}
```

### Market Cap Filter
```python
{
    'name': 'market_cap',
    'parameters': {
        'categories': ['LARGE', 'MID']  # List of allowed categories
    }
}
```

### Sector Filter
```python
{
    'name': 'sector',
    'parameters': {
        'sectors': ['Technology', 'Financial Services']  # List of allowed sectors
    }
}
```

### Trend Filter
```python
{
    'name': 'trend',
    'parameters': {
        'direction': 'BULLISH'  # 'BULLISH', 'BEARISH', 'NEUTRAL', or 'any'
    }
}
```

### Volume Filter
```python
{
    'name': 'volume',
    'parameters': {
        'state': 'HIGH'  # 'HIGH', 'NORMAL', 'LOW', or 'any'
    }
}
```

## 🔧 Advanced Usage

### Detailed Screening with Results

```python
manager = FilterManager()
manager.add_filter('rsi', {'min': 40, 'max': 80})
manager.add_filter('volume', {'state': 'HIGH'})

analyses = engine.analyze_multiple_stocks(symbols, period='1mo')

# Get detailed results
results = manager.screen_with_details(analyses)

print(f"Total: {results['filter_summary']['total_stocks']}")
print(f"Passed: {results['filter_summary']['passed']}")
print(f"Pass Rate: {results['filter_summary']['pass_rate']:.1f}%")

# See which filters passed/failed for each stock
for symbol, data in results['passed'].items():
    print(f"\n{symbol}:")
    for filter_name, filter_result in data['filter_results']['individual_filters'].items():
        print(f"  {filter_name}: {'✓' if filter_result['passed'] else '✗'}")
```

### Export/Import Configuration

```python
# Export configuration
manager.export_config('my_filters.json')

# Import configuration
new_manager = FilterManager()
new_manager.import_config('my_filters.json')
```

### Sector/Index Screening

```python
api = FilterAPI()
api.create_manager('sector_screen')

# Add filters
api.add_filter('sector_screen', 'rsi', {'min': 40, 'max': 60})
api.add_filter('sector_screen', 'trend', {'direction': 'BULLISH'})

# Screen by sector
results = api.screen_by_type(
    'sector_screen',
    stock_type='IT',  # or 'BANKING', 'PHARMA', 'NIFTY50', etc.
    limit=10
)
```

### Test Filter on Single Stock

```python
api = FilterAPI()
api.create_manager('test')
api.add_filter('test', 'rsi', {'min': 30, 'max': 70})

# Test on single stock
result = api.test_filter_on_stock('test', 'RELIANCE')

print(f"Stock: {result['symbol']}")
print(f"Passed: {result['passed']}")
print(f"Filter Results: {result['filter_results']}")
```

## 🎨 Integration with Django

### Django View Example

```python
# views.py
from django.http import JsonResponse
from filter_api import FilterAPI

api = FilterAPI()

def create_filter_view(request):
    """Create a new filter manager"""
    manager_id = request.POST.get('manager_id')
    preset = request.POST.get('preset', None)

    result = api.create_manager(manager_id, preset)
    return JsonResponse(result)

def add_filter_view(request):
    """Add a filter to manager"""
    manager_id = request.POST.get('manager_id')
    filter_name = request.POST.get('filter_name')
    parameters = json.loads(request.POST.get('parameters', '{}'))

    result = api.add_filter(manager_id, filter_name, parameters)
    return JsonResponse(result)

def screen_stocks_view(request):
    """Screen stocks with filters"""
    manager_id = request.POST.get('manager_id')
    symbols = json.loads(request.POST.get('symbols'))
    period = request.POST.get('period', '1mo')

    results = api.screen_stocks(manager_id, symbols, period)
    return JsonResponse(results)
```

### JavaScript Frontend Example

```javascript
// Create manager
fetch('/api/filters/create/', {
    method: 'POST',
    body: JSON.stringify({
        manager_id: 'user123',
        preset: 'momentum_trading'
    })
});

// Add filter
fetch('/api/filters/add/', {
    method: 'POST',
    body: JSON.stringify({
        manager_id: 'user123',
        filter_name: 'rsi',
        parameters: {min: 30, max: 70}
    })
});

// Screen stocks
fetch('/api/filters/screen/', {
    method: 'POST',
    body: JSON.stringify({
        manager_id: 'user123',
        symbols: ['RELIANCE', 'TCS', 'INFY']
    })
})
.then(response => response.json())
.then(data => {
    console.log('Passed stocks:', data.passed);
    console.log('Pass rate:', data.summary.pass_rate);
});
```

## 📝 Filter Combinations

### Example 1: Momentum + Quality
```python
manager.add_filter('rsi', {'min': 50, 'max': 80})
manager.add_filter('trend', {'direction': 'BULLISH'})
manager.add_filter('volume', {'state': 'HIGH'})
manager.add_filter('market_cap', {'categories': ['LARGE']})
manager.add_filter('liquidity', {'levels': ['HIGH']})
```

### Example 2: Oversold Value Plays
```python
manager.add_filter('rsi', {'min': 20, 'max': 35})
manager.add_filter('pe_ratio', {'min': 0, 'max': 15})
manager.add_filter('pb_ratio', {'min': 0, 'max': 2})
manager.add_filter('trend', {'direction': 'BULLISH'})
```

### Example 3: Breakout Candidates
```python
manager.add_filter('adx', {'min': 25})
manager.add_filter('volume_spike', {'required': True})
manager.add_filter('bollinger_bands', {'position': 'outside_upper'})
manager.add_filter('liquidity', {'levels': ['HIGH']})
```

## 🧪 Run Examples

```bash
python filter_examples.py
```

This runs 12 interactive examples demonstrating all features.

## 📚 Available Filter Types

| Type | Filters |
|------|---------|
| **Technical** | rsi, macd, bollinger_bands, atr, vwap, stochastic, adx, trend, momentum |
| **Price** | price, price_change |
| **Volume** | volume, volume_spike |
| **Classification** | market_cap, sector, liquidity, volatility, index_membership |
| **Fundamental** | pe_ratio, pb_ratio, beta |

## 🎯 Filter Presets

| Preset | Description | Filters |
|--------|-------------|---------|
| `momentum_trading` | Momentum strategies | RSI (40-80), Trend (Bullish), Volume (High), Liquidity (High) |
| `value_investing` | Value investing | PE (0-20), PB (0-3), Market Cap (Large/Mid) |
| `oversold_bounce` | Oversold bounce plays | RSI (20-35), Trend (Bullish), Volume Spike |
| `breakout_trading` | Breakout strategies | ADX (>25), Volume (High), BB (Outside Upper), Liquidity (High) |
| `swing_trading` | Swing trading | RSI (30-70), MACD (Bullish), Trend (Bullish), ATR (10-100) |
| `conservative` | Low-risk stocks | Market Cap (Large), Liquidity (High), Volatility (Low/Medium), Beta (<1.2) |

## 💡 Tips

1. **Start with presets** - Use pre-configured filter sets and modify as needed
2. **Test individually** - Test each filter separately before combining
3. **Use groups** - Organize related filters into groups
4. **Enable/disable dynamically** - Toggle filters without removing them
5. **Export/import** - Save successful filter configurations
6. **Detailed screening** - Use detailed results to understand why stocks pass/fail

## 🔗 Integration with Existing System

```python
# Use with existing intraday_trading_system.py
from intraday_trading_system import IntradayTradingSystem
from filter_manager import FilterManager

trading_system = IntradayTradingSystem()
filter_manager = FilterManager()

# Get liquid stocks from trading system
symbols = trading_system.get_liquid_stocks()

# Apply your custom filters
filter_manager.add_filter('rsi', {'min': 40, 'max': 60})
passed = filter_manager.screen_stocks(analyses)
```

---

**Happy Filtering! 🎯**
