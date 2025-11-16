from django.urls import path
from . import views

app_name = 'signals'

urlpatterns = [
    # Web views
    path('', views.dashboard, name='dashboard'),
    path('signals/', views.signals_list, name='signals_list'),
    path('signals/<int:pk>/', views.signal_detail, name='signal_detail'),

    # API endpoints
    path('api/signals/', views.api_signals, name='api_signals'),
    path('api/dashboard-stats/', views.api_dashboard_stats, name='api_dashboard_stats'),
]
