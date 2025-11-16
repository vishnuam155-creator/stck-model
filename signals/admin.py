from django.contrib import admin
from .models import TradingSignal, MarketData, ScanSession, NewsItem


@admin.register(TradingSignal)
class TradingSignalAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'signal_type', 'strategy', 'entry_price', 'target_price',
                    'rrr', 'confidence_score', 'timestamp', 'is_active']
    list_filter = ['signal_type', 'strategy', 'is_active', 'timestamp']
    search_fields = ['symbol']
    ordering = ['-confidence_score', '-timestamp']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'nifty_50', 'pcr_ratio', 'india_vix', 'market_sentiment']
    list_filter = ['market_sentiment', 'timestamp']
    ordering = ['-timestamp']


@admin.register(ScanSession)
class ScanSessionAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'total_signals', 'buy_signals', 'sell_signals',
                    'total_capital_required', 'status']
    list_filter = ['status', 'timestamp']
    ordering = ['-timestamp']


@admin.register(NewsItem)
class NewsItemAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'title', 'sentiment', 'sentiment_score', 'published_at', 'source']
    list_filter = ['sentiment', 'source', 'published_at']
    search_fields = ['symbol', 'title']
    ordering = ['-published_at']
