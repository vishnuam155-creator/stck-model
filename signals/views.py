from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Count, Sum, Avg, Q
from django.utils import timezone
from datetime import timedelta
from .models import TradingSignal, ScanSession, MarketData, NewsItem
from rest_framework.decorators import api_view
from rest_framework.response import Response


def dashboard(request):
    """Main dashboard view with analytics"""

    # Get today's signals
    today = timezone.now().date()
    signals_today = TradingSignal.objects.filter(
        timestamp__date=today,
        is_active=True
    )

    # Get latest scan session
    latest_scan = ScanSession.objects.filter(status='COMPLETED').first()

    # Calculate metrics
    total_signals = signals_today.count()
    buy_signals = signals_today.filter(signal_type='BUY').count()
    sell_signals = signals_today.filter(signal_type='SELL').count()

    # Strategy breakdown
    strategy_breakdown = signals_today.values('strategy').annotate(
        count=Count('id')
    ).order_by('-count')

    # Top signals by confidence
    top_signals = signals_today.order_by('-confidence_score')[:10]

    # Risk/Reward analytics
    total_capital_required = signals_today.aggregate(
        total=Sum('capital_required')
    )['total'] or 0

    total_potential_profit = signals_today.aggregate(
        total=Sum('reward_amount')
    )['total'] or 0

    avg_rrr = signals_today.aggregate(
        avg=Avg('rrr')
    )['avg'] or 0

    # Calculate potential ROI
    potential_roi = 0
    if total_capital_required > 0:
        potential_roi = (float(total_potential_profit) / float(total_capital_required)) * 100

    # Get latest market data
    latest_market_data = MarketData.objects.first()

    # Recent scan sessions
    recent_scans = ScanSession.objects.filter(
        status='COMPLETED'
    ).order_by('-timestamp')[:5]

    context = {
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'strategy_breakdown': strategy_breakdown,
        'top_signals': top_signals,
        'total_capital_required': total_capital_required,
        'total_potential_profit': total_potential_profit,
        'potential_roi': potential_roi,
        'avg_rrr': avg_rrr,
        'latest_scan': latest_scan,
        'latest_market_data': latest_market_data,
        'recent_scans': recent_scans,
    }

    return render(request, 'signals/dashboard.html', context)


def signals_list(request):
    """List all trading signals with filtering"""

    # Get filter parameters
    signal_type = request.GET.get('signal_type', '')
    strategy = request.GET.get('strategy', '')
    min_confidence = request.GET.get('min_confidence', 0)
    search = request.GET.get('search', '')

    # Start with all active signals
    signals = TradingSignal.objects.filter(is_active=True)

    # Apply filters
    if signal_type:
        signals = signals.filter(signal_type=signal_type)

    if strategy:
        signals = signals.filter(strategy=strategy)

    if min_confidence:
        try:
            signals = signals.filter(confidence_score__gte=float(min_confidence))
        except ValueError:
            pass

    if search:
        signals = signals.filter(
            Q(symbol__icontains=search) |
            Q(confluences__icontains=search)
        )

    # Order by confidence score
    signals = signals.order_by('-confidence_score', '-timestamp')

    context = {
        'signals': signals,
        'signal_type': signal_type,
        'strategy': strategy,
        'min_confidence': min_confidence,
        'search': search,
        'signal_types': TradingSignal.SIGNAL_TYPES,
        'strategies': TradingSignal.STRATEGY_TYPES,
    }

    return render(request, 'signals/signals_list.html', context)


def signal_detail(request, pk):
    """Detail view for a single trading signal"""

    signal = get_object_or_404(TradingSignal, pk=pk)

    # Get related news for this symbol
    news_items = NewsItem.objects.filter(
        symbol=signal.symbol
    ).order_by('-published_at')[:10]

    # Get other signals for the same symbol (last 7 days)
    seven_days_ago = timezone.now() - timedelta(days=7)
    related_signals = TradingSignal.objects.filter(
        symbol=signal.symbol,
        timestamp__gte=seven_days_ago
    ).exclude(pk=pk).order_by('-timestamp')[:5]

    context = {
        'signal': signal,
        'news_items': news_items,
        'related_signals': related_signals,
    }

    return render(request, 'signals/signal_detail.html', context)


# API Views
@api_view(['GET'])
def api_signals(request):
    """API endpoint to get all signals"""

    signals = TradingSignal.objects.filter(is_active=True).order_by('-confidence_score')

    data = []
    for signal in signals:
        data.append({
            'id': signal.id,
            'symbol': signal.symbol,
            'signal_type': signal.signal_type,
            'strategy': signal.get_strategy_display(),
            'entry_price': float(signal.entry_price),
            'stop_loss': float(signal.stop_loss),
            'target_price': float(signal.target_price),
            'current_price': float(signal.current_price),
            'rrr': float(signal.rrr),
            'confidence_score': float(signal.confidence_score),
            'position_size': signal.position_size,
            'capital_required': float(signal.capital_required),
            'timestamp': signal.timestamp.isoformat(),
        })

    return Response(data)


@api_view(['GET'])
def api_dashboard_stats(request):
    """API endpoint for dashboard statistics"""

    today = timezone.now().date()
    signals_today = TradingSignal.objects.filter(
        timestamp__date=today,
        is_active=True
    )

    stats = {
        'total_signals': signals_today.count(),
        'buy_signals': signals_today.filter(signal_type='BUY').count(),
        'sell_signals': signals_today.filter(signal_type='SELL').count(),
        'total_capital_required': float(signals_today.aggregate(
            total=Sum('capital_required')
        )['total'] or 0),
        'total_potential_profit': float(signals_today.aggregate(
            total=Sum('reward_amount')
        )['total'] or 0),
        'avg_rrr': float(signals_today.aggregate(
            avg=Avg('rrr')
        )['avg'] or 0),
    }

    # Strategy breakdown
    strategy_breakdown = list(signals_today.values('strategy').annotate(
        count=Count('id')
    ).order_by('-count'))

    stats['strategy_breakdown'] = strategy_breakdown

    return Response(stats)
