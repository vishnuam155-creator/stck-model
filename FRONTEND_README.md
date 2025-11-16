# Trading Signal Frontend - Django Web Application

A modern, responsive Django web application for visualizing trading signals from the intraday trading system.

## Features

- **Modern Dashboard**: Real-time analytics, signal overview, and financial metrics
- **Signal List View**: Filterable and searchable list of all trading signals
- **Signal Detail View**: Comprehensive view of individual signals with technical indicators
- **Responsive Design**: Beautiful UI built with Tailwind CSS
- **REST API**: JSON endpoints for real-time data access
- **Admin Panel**: Full Django admin for data management

## Setup Instructions

### 1. Install Dependencies

The required packages are already installed:
- Django 5.2.8
- Django REST Framework
- Django Extensions
- Pandas, Plotly

### 2. Database Setup

The database is already migrated and ready to use. If you need to reset:

```bash
# Run migrations (already done)
python manage.py migrate

# Create a superuser for admin access
python manage.py createsuperuser
```

### 3. Import Trading Signals

Import signals from the intraday trading system:

```bash
# Option 1: Run the scanner and import results
python manage.py import_signals --capital 100000 --stocks 50

# Option 2: Import from existing CSV file
python manage.py import_signals --csv-only --output intraday_signals.csv

# With custom parameters
python manage.py import_signals --capital 200000 --stocks 100 --output my_signals.csv
```

### 4. Start the Development Server

```bash
python manage.py runserver
```

The application will be available at: **http://127.0.0.1:8000**

## Application Structure

```
trading_platform/          # Django project
├── settings.py           # Project settings
├── urls.py              # URL routing
└── wsgi.py              # WSGI config

signals/                  # Main app
├── models.py            # Data models
├── views.py             # View logic
├── admin.py             # Admin configuration
├── urls.py              # App URL routing
├── templates/           # HTML templates
│   └── signals/
│       ├── base.html           # Base template
│       ├── dashboard.html      # Dashboard view
│       ├── signals_list.html   # Signals list
│       └── signal_detail.html  # Signal detail
└── management/
    └── commands/
        └── import_signals.py   # Data import command

static/                  # Static files (CSS, JS, images)
media/                   # User uploaded files
```

## Pages & Features

### 1. Dashboard (/)
**URL**: http://127.0.0.1:8000/

Features:
- Total signals count (Buy/Sell breakdown)
- Average Risk:Reward ratio
- Financial overview (capital required, potential profit, ROI)
- Strategy breakdown (ORB, VWAP Pullback, Confluence Reversal)
- Top 10 signals by confidence
- Recent scan sessions history
- Live market time (IST)

### 2. Signals List (/signals/)
**URL**: http://127.0.0.1:8000/signals/

Features:
- All trading signals in card view
- Filters:
  - Search by symbol (e.g., RELIANCE, TCS)
  - Signal type (BUY/SELL)
  - Strategy (ORB, VWAP, Confluence)
  - Minimum confidence score
- Each signal card shows:
  - Symbol and signal type
  - Price levels (current, entry, stop loss, target)
  - Position size and capital required
  - Technical indicators (RSI, VWAP, Volume)
  - Confluence factors
  - Confidence score

### 3. Signal Detail (/signals/<id>/)
**URL**: http://127.0.0.1:8000/signals/1/

Features:
- Complete signal information
- Price levels with visual risk:reward ratio
- Position details (size, capital, risk, reward)
- All technical indicators (RSI, VWAP, ATR, MACD, EMAs, etc.)
- Bollinger Bands, ORB levels, Support/Resistance
- Confluence factors with visual badges
- Related news with sentiment analysis
- Recent signals for the same symbol
- Signal metadata (creation date, status, ID)

### 4. Admin Panel (/admin/)
**URL**: http://127.0.0.1:8000/admin/

Features:
- Manage all trading signals
- View market data
- Track scan sessions
- Manage news items
- Full CRUD operations

## API Endpoints

### GET /api/signals/
Returns all active trading signals in JSON format.

**Response**:
```json
[
  {
    "id": 1,
    "symbol": "RELIANCE",
    "signal_type": "BUY",
    "strategy": "Opening Range Breakout",
    "entry_price": 2450.0,
    "stop_loss": 2435.0,
    "target_price": 2480.0,
    "current_price": 2445.5,
    "rrr": 2.0,
    "confidence_score": 85.0,
    "position_size": 67,
    "capital_required": 164150.0,
    "timestamp": "2025-11-16T09:30:15+05:30"
  }
]
```

### GET /api/dashboard-stats/
Returns dashboard statistics.

**Response**:
```json
{
  "total_signals": 25,
  "buy_signals": 15,
  "sell_signals": 10,
  "total_capital_required": 2500000.0,
  "total_potential_profit": 125000.0,
  "avg_rrr": 2.3,
  "strategy_breakdown": [
    {"strategy": "ORB", "count": 12},
    {"strategy": "VWAP_PULLBACK", "count": 8},
    {"strategy": "CONFLUENCE_REVERSAL", "count": 5}
  ]
}
```

## Database Models

### TradingSignal
Main model storing trading signals with:
- Basic info (symbol, signal type, strategy, timestamp)
- Price levels (entry, stop loss, target, current)
- Risk management (RRR, position size, capital required)
- Technical indicators (RSI, VWAP, ATR, MACD, EMAs, etc.)
- Support/Resistance levels
- ORB levels (high, low, midpoint)
- Bollinger Bands
- Confidence score and confluence factors

### MarketData
Market-wide data:
- NIFTY 50, NIFTY Bank indices
- PCR (Put-Call Ratio)
- India VIX
- Market sentiment (Bullish/Bearish/Neutral)

### ScanSession
Tracking each scanning session:
- Timestamp
- Total signals count
- Buy/Sell signal breakdown
- Total capital required
- Total potential profit
- Scan duration
- Status (Running/Completed/Failed)

### NewsItem
News articles for stocks:
- Symbol, title, URL
- Published date, source
- Sentiment (Positive/Negative/Neutral)
- Sentiment score (-2.0 to +2.0)

## Workflow

### Daily Trading Workflow

1. **Morning (Before Market Opens)**
   ```bash
   # Run the intraday scanner
   python manage.py import_signals --capital 100000 --stocks 50
   ```

2. **Review Signals**
   - Open http://127.0.0.1:8000
   - Review dashboard for overview
   - Check top signals by confidence
   - Filter signals by strategy/confidence

3. **Analyze Individual Signals**
   - Click on any signal to view details
   - Review all technical indicators
   - Check confluence factors
   - Read related news

4. **Execute Trades**
   - Use the signal details for trade execution
   - Entry price, stop loss, target are clearly displayed
   - Position size is calculated based on your capital

### Continuous Monitoring

The dashboard auto-refreshes statistics every 30 seconds to show the latest data.

## Customization

### Change Capital Amount
Edit the import command:
```bash
python manage.py import_signals --capital 200000
```

### Add Custom Filters
Edit `signals/views.py` to add more filter options.

### Customize UI Colors
Edit `signals/templates/signals/base.html` and modify the Tailwind classes.

### Add More Indicators
1. Add fields to `TradingSignal` model in `signals/models.py`
2. Run migrations: `python manage.py makemigrations && python manage.py migrate`
3. Update templates to display new indicators

## Technologies Used

- **Backend**: Django 5.2.8, Django REST Framework
- **Frontend**: Tailwind CSS, Alpine.js, Font Awesome
- **Charts**: Chart.js, Plotly
- **Database**: SQLite (can be changed to PostgreSQL/MySQL)
- **Icons**: Font Awesome 6

## Design Features

- **Gradient Backgrounds**: Purple to blue gradients for modern look
- **Card Shadows**: Subtle shadows with hover effects
- **Color Coding**:
  - Green for BUY signals and profits
  - Red for SELL signals and losses
  - Purple for neutral metrics
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Live Clock**: Shows current IST time
- **Animations**: Smooth transitions and hover effects

## Troubleshooting

### No signals showing up
- Run the import command: `python manage.py import_signals`
- Check if the CSV file was generated by the trading system

### Admin panel not accessible
- Create a superuser: `python manage.py createsuperuser`
- Use the credentials to log in

### Static files not loading
- Run: `python manage.py collectstatic`
- Make sure `DEBUG = True` in settings.py for development

### Database errors
- Delete `db.sqlite3` and run migrations again:
  ```bash
  rm db.sqlite3
  python manage.py migrate
  ```

## Next Steps

1. **Add Real-time Updates**: Use WebSockets (Django Channels) for live signal updates
2. **Add Charts**: Integrate TradingView or Plotly charts for price visualization
3. **Add Backtesting**: Show historical signal performance
4. **Add Alerts**: Email/SMS notifications for high-confidence signals
5. **Add Portfolio Tracking**: Track executed trades and P&L
6. **Deploy to Production**: Use Gunicorn + Nginx for production deployment

## Support

For issues or questions:
- Check the main README.md
- Review Django documentation: https://docs.djangoproject.com/
- Check the trading system code in `intraday_trading_system.py`

---

**Happy Trading!** 📈
