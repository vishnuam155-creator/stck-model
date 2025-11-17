#!/usr/bin/env python3
"""
Generate demo trading signals for testing the Django frontend
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, '/home/user/stck-model')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_platform.settings')
django.setup()

from django.utils import timezone
from signals.models import TradingSignal, ScanSession
from decimal import Decimal
from datetime import timedelta
import random

# Demo stock data
DEMO_STOCKS = [
    {
        'symbol': 'RELIANCE',
        'signal_type': 'BUY',
        'strategy': 'ORB',
        'entry': 2450.00,
        'stop_loss': 2435.00,
        'target': 2480.00,
        'current': 2445.50,
        'rsi': 62.5,
        'vwap': 2448.25,
        'atr': 15.50,
        'volume_ratio': 3.2,
        'confidence': 85.0,
        'confluences': 'RSI_BULLISH,HIGH_VOLUME,TREND_STRONG',
    },
    {
        'symbol': 'TCS',
        'signal_type': 'BUY',
        'strategy': 'VWAP_PULLBACK',
        'entry': 3680.00,
        'stop_loss': 3650.00,
        'target': 3740.00,
        'current': 3675.00,
        'rsi': 45.0,
        'vwap': 3678.00,
        'atr': 22.30,
        'volume_ratio': 2.8,
        'confidence': 78.0,
        'confluences': 'VWAP_SUPPORT,RSI_NEUTRAL,EMA_SUPPORT',
    },
    {
        'symbol': 'HDFCBANK',
        'signal_type': 'BUY',
        'strategy': 'CONFLUENCE_REVERSAL',
        'entry': 1650.00,
        'stop_loss': 1635.00,
        'target': 1680.00,
        'current': 1648.00,
        'rsi': 38.5,
        'vwap': 1649.50,
        'atr': 12.40,
        'volume_ratio': 4.1,
        'confidence': 82.0,
        'confluences': 'SUPPORT_LEVEL,RSI_OVERSOLD,MACD_BULLISH,HIGH_VOLUME',
    },
    {
        'symbol': 'INFY',
        'signal_type': 'SELL',
        'strategy': 'CONFLUENCE_REVERSAL',
        'entry': 1580.00,
        'stop_loss': 1595.00,
        'target': 1550.00,
        'current': 1582.00,
        'rsi': 72.5,
        'vwap': 1577.00,
        'atr': 18.20,
        'volume_ratio': 3.5,
        'confidence': 80.0,
        'confluences': 'RESISTANCE_LEVEL,RSI_OVERBOUGHT,MACD_BEARISH',
    },
    {
        'symbol': 'ICICIBANK',
        'signal_type': 'BUY',
        'strategy': 'ORB',
        'entry': 1050.00,
        'stop_loss': 1040.00,
        'target': 1070.00,
        'current': 1048.00,
        'rsi': 58.0,
        'vwap': 1049.00,
        'atr': 8.50,
        'volume_ratio': 5.2,
        'confidence': 88.0,
        'confluences': 'ORB_BREAKOUT,HIGH_VOLUME,TREND_BULLISH',
    },
    {
        'symbol': 'SBIN',
        'signal_type': 'BUY',
        'strategy': 'VWAP_PULLBACK',
        'entry': 625.00,
        'stop_loss': 615.00,
        'target': 645.00,
        'current': 623.00,
        'rsi': 52.0,
        'vwap': 624.00,
        'atr': 6.80,
        'volume_ratio': 2.9,
        'confidence': 75.0,
        'confluences': 'VWAP_BOUNCE,VOLUME_CONFIRMATION',
    },
    {
        'symbol': 'TATASTEEL',
        'signal_type': 'SELL',
        'strategy': 'CONFLUENCE_REVERSAL',
        'entry': 115.00,
        'stop_loss': 118.00,
        'target': 109.00,
        'current': 115.50,
        'rsi': 68.0,
        'vwap': 114.00,
        'atr': 2.10,
        'volume_ratio': 3.8,
        'confidence': 77.0,
        'confluences': 'RESISTANCE,RSI_HIGH,BELOW_VWAP',
    },
    {
        'symbol': 'BHARTIARTL',
        'signal_type': 'BUY',
        'strategy': 'ORB',
        'entry': 1210.00,
        'stop_loss': 1200.00,
        'target': 1230.00,
        'current': 1208.00,
        'rsi': 55.0,
        'vwap': 1209.00,
        'atr': 9.20,
        'volume_ratio': 4.5,
        'confidence': 83.0,
        'confluences': 'ORB_HIGH,VOLUME_SPIKE,TREND_UP',
    },
    {
        'symbol': 'LT',
        'signal_type': 'BUY',
        'strategy': 'VWAP_PULLBACK',
        'entry': 3450.00,
        'stop_loss': 3425.00,
        'target': 3500.00,
        'current': 3445.00,
        'rsi': 48.0,
        'vwap': 3448.00,
        'atr': 28.50,
        'volume_ratio': 2.5,
        'confidence': 72.0,
        'confluences': 'VWAP_SUPPORT,EMA_CROSS',
    },
    {
        'symbol': 'ITC',
        'signal_type': 'BUY',
        'strategy': 'CONFLUENCE_REVERSAL',
        'entry': 465.00,
        'stop_loss': 460.00,
        'target': 475.00,
        'current': 464.00,
        'rsi': 42.0,
        'vwap': 464.50,
        'atr': 4.20,
        'volume_ratio': 3.1,
        'confidence': 79.0,
        'confluences': 'SUPPORT_STRONG,RSI_BULLISH,VOLUME_UP',
    },
]

def calculate_position_details(entry, stop_loss, target, capital=100000, max_risk_pct=1.0):
    """Calculate position size and risk/reward"""
    # Calculate risk per share
    risk_per_share = abs(entry - stop_loss)

    # Max risk amount (1% of capital)
    max_risk = capital * (max_risk_pct / 100)

    # Position size
    position_size = int(max_risk / risk_per_share)

    # Capital required
    capital_required = entry * position_size

    # Risk and reward amounts
    risk_amount = risk_per_share * position_size
    reward_amount = abs(target - entry) * position_size

    # RRR
    rrr = reward_amount / risk_amount if risk_amount > 0 else 0

    return {
        'position_size': position_size,
        'capital_required': capital_required,
        'risk_amount': risk_amount,
        'reward_amount': reward_amount,
        'rrr': rrr,
        'max_risk': max_risk,
    }

def create_demo_signals():
    """Create demo trading signals"""

    print("Creating demo trading signals...")

    # Create scan session
    scan_session = ScanSession.objects.create(
        timestamp=timezone.now(),
        status='COMPLETED',
    )

    signals_created = 0
    total_capital = 0
    total_profit = 0
    buy_count = 0
    sell_count = 0

    for stock_data in DEMO_STOCKS:
        # Calculate position details
        details = calculate_position_details(
            entry=stock_data['entry'],
            stop_loss=stock_data['stop_loss'],
            target=stock_data['target']
        )

        # Create signal
        signal = TradingSignal.objects.create(
            symbol=stock_data['symbol'],
            signal_type=stock_data['signal_type'],
            strategy=stock_data['strategy'],
            timestamp=timezone.now() - timedelta(minutes=random.randint(5, 60)),

            # Price levels
            entry_price=Decimal(str(stock_data['entry'])),
            stop_loss=Decimal(str(stock_data['stop_loss'])),
            target_price=Decimal(str(stock_data['target'])),
            current_price=Decimal(str(stock_data['current'])),

            # Risk management
            risk_amount=Decimal(str(details['risk_amount'])),
            reward_amount=Decimal(str(details['reward_amount'])),
            rrr=Decimal(str(details['rrr'])),
            position_size=details['position_size'],
            capital_required=Decimal(str(details['capital_required'])),
            max_risk=Decimal(str(details['max_risk'])),

            # Technical indicators
            rsi=Decimal(str(stock_data['rsi'])),
            vwap=Decimal(str(stock_data['vwap'])),
            atr=Decimal(str(stock_data['atr'])),
            volume_ratio=Decimal(str(stock_data['volume_ratio'])),

            # Confluence and confidence
            confidence_score=Decimal(str(stock_data['confidence'])),
            confluences=stock_data['confluences'],

            is_active=True,
        )

        signals_created += 1
        total_capital += float(details['capital_required'])
        total_profit += float(details['reward_amount'])

        if stock_data['signal_type'] == 'BUY':
            buy_count += 1
        else:
            sell_count += 1

        print(f"✓ Created: {stock_data['symbol']} - {stock_data['signal_type']} ({stock_data['strategy']})")

    # Update scan session
    scan_session.total_signals = signals_created
    scan_session.buy_signals = buy_count
    scan_session.sell_signals = sell_count
    scan_session.total_capital_required = Decimal(str(total_capital))
    scan_session.total_potential_profit = Decimal(str(total_profit))
    scan_session.save()

    print(f"\n✅ Successfully created {signals_created} demo signals!")
    print(f"   Buy signals: {buy_count}")
    print(f"   Sell signals: {sell_count}")
    print(f"   Total capital required: ₹{total_capital:,.2f}")
    print(f"   Total potential profit: ₹{total_profit:,.2f}")
    print(f"\n🌐 View the dashboard at: http://127.0.0.1:8000")

if __name__ == '__main__':
    create_demo_signals()
