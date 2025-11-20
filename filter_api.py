"""
Web-Friendly Filter API
REST-style API for filter operations
Can be integrated with Django views
"""

from typing import Dict, Any, List, Optional
import json

from filter_manager import FilterManager, FilterPresets, CombineLogic
from stock_filters import FilterRegistry, FilterType
from indicator_engine import IndicatorEngine


class FilterAPI:
    """
    Web-friendly API for filter operations
    Designed to work with JSON requests/responses
    """

    def __init__(self):
        """Initialize filter API"""
        self.managers: Dict[str, FilterManager] = {}
        self.engine = IndicatorEngine()

    # ==================== FILTER INFO ====================

    def get_available_filters(self) -> Dict[str, Any]:
        """
        Get list of all available filters

        Returns:
            {
                'filters': [
                    {
                        'name': 'rsi',
                        'type': 'technical',
                        'description': 'Filter by RSI...',
                        'default_parameters': {...}
                    },
                    ...
                ],
                'types': ['technical', 'fundamental', 'classification', ...]
            }
        """
        filters = []

        for filter_name in FilterRegistry.list_filters():
            filter_instance = FilterRegistry.get_filter(filter_name)
            filters.append({
                'name': filter_name,
                'type': filter_instance.config.filter_type.value,
                'description': filter_instance.config.description,
                'default_parameters': filter_instance.config.parameters
            })

        types = list(set(f['type'] for f in filters))

        return {
            'filters': filters,
            'types': types,
            'count': len(filters)
        }

    def get_filters_by_type(self, filter_type: str) -> Dict[str, Any]:
        """
        Get filters of a specific type

        Args:
            filter_type: Type of filters ('technical', 'fundamental', etc.)

        Returns:
            {
                'type': 'technical',
                'filters': [...]
            }
        """
        try:
            ftype = FilterType(filter_type.lower())
            filter_names = FilterRegistry.get_filters_by_type(ftype)

            filters = []
            for name in filter_names:
                filter_instance = FilterRegistry.get_filter(name)
                filters.append({
                    'name': name,
                    'description': filter_instance.config.description,
                    'default_parameters': filter_instance.config.parameters
                })

            return {
                'type': filter_type,
                'filters': filters,
                'count': len(filters)
            }
        except ValueError:
            return {'error': f'Invalid filter type: {filter_type}'}

    def get_presets(self) -> Dict[str, Any]:
        """
        Get available filter presets

        Returns:
            {
                'presets': [
                    {
                        'name': 'momentum_trading',
                        'description': 'Filters for momentum trading'
                    },
                    ...
                ]
            }
        """
        presets = [
            {
                'name': 'momentum_trading',
                'description': 'Filters for momentum trading strategies'
            },
            {
                'name': 'value_investing',
                'description': 'Filters for value investing'
            },
            {
                'name': 'oversold_bounce',
                'description': 'Filters for oversold bounce plays'
            },
            {
                'name': 'breakout_trading',
                'description': 'Filters for breakout trading'
            },
            {
                'name': 'swing_trading',
                'description': 'Filters for swing trading'
            },
            {
                'name': 'conservative',
                'description': 'Conservative filters for low-risk stocks'
            }
        ]

        return {'presets': presets, 'count': len(presets)}

    # ==================== MANAGER OPERATIONS ====================

    def create_manager(self, manager_id: str, preset: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new filter manager

        Args:
            manager_id: Unique ID for this manager
            preset: Optional preset name

        Returns:
            {
                'success': True,
                'manager_id': 'abc123',
                'message': 'Manager created'
            }
        """
        if preset:
            manager = FilterPresets.__dict__[preset]() if preset in FilterPresets.__dict__ else FilterManager()
        else:
            manager = FilterManager()

        self.managers[manager_id] = manager

        return {
            'success': True,
            'manager_id': manager_id,
            'preset': preset,
            'message': f'Manager created{" with preset: " + preset if preset else ""}'
        }

    def delete_manager(self, manager_id: str) -> Dict[str, Any]:
        """Delete a filter manager"""
        if manager_id in self.managers:
            del self.managers[manager_id]
            return {'success': True, 'message': f'Manager {manager_id} deleted'}
        return {'success': False, 'error': f'Manager {manager_id} not found'}

    def get_manager_info(self, manager_id: str) -> Dict[str, Any]:
        """Get information about a manager"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        return {
            'manager_id': manager_id,
            'info': manager.get_info()
        }

    # ==================== ADD/REMOVE FILTERS ====================

    def add_filter(
        self,
        manager_id: str,
        filter_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Add a filter to manager

        Args:
            manager_id: Manager ID
            filter_name: Name of filter to add
            parameters: Filter parameters
            enabled: Whether filter is enabled

        Returns:
            {
                'success': True,
                'filter': {
                    'name': 'rsi',
                    'parameters': {...}
                }
            }
        """
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]

        try:
            filter_obj = manager.add_filter(filter_name, parameters, enabled)
            return {
                'success': True,
                'filter': filter_obj.get_info()
            }
        except ValueError as e:
            return {'error': str(e)}

    def remove_filter(self, manager_id: str, filter_name: str) -> Dict[str, Any]:
        """Remove a filter from manager"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        manager.remove_filter(filter_name)

        return {
            'success': True,
            'message': f'Filter {filter_name} removed'
        }

    def add_filter_group(
        self,
        manager_id: str,
        group_name: str,
        filters: List[Dict[str, Any]],
        logic: str = 'AND'
    ) -> Dict[str, Any]:
        """
        Add a filter group

        Args:
            manager_id: Manager ID
            group_name: Name for the group
            filters: List of filter configs
            logic: Combination logic ('AND' or 'OR')

        Returns:
            {
                'success': True,
                'group': {...}
            }
        """
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        combine_logic = CombineLogic.AND if logic.upper() == 'AND' else CombineLogic.OR

        try:
            group = manager.add_group(group_name, filters, combine_logic)
            return {
                'success': True,
                'group': group.get_info()
            }
        except Exception as e:
            return {'error': str(e)}

    # ==================== ENABLE/DISABLE ====================

    def enable_filter(self, manager_id: str, filter_name: str) -> Dict[str, Any]:
        """Enable a filter"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        manager.enable_filter(filter_name)

        return {'success': True, 'message': f'Filter {filter_name} enabled'}

    def disable_filter(self, manager_id: str, filter_name: str) -> Dict[str, Any]:
        """Disable a filter"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        manager.disable_filter(filter_name)

        return {'success': True, 'message': f'Filter {filter_name} disabled'}

    def enable_group(self, manager_id: str, group_name: str) -> Dict[str, Any]:
        """Enable a filter group"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        manager.enable_group(group_name)

        return {'success': True, 'message': f'Group {group_name} enabled'}

    def disable_group(self, manager_id: str, group_name: str) -> Dict[str, Any]:
        """Disable a filter group"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        manager.disable_group(group_name)

        return {'success': True, 'message': f'Group {group_name} disabled'}

    # ==================== UPDATE PARAMETERS ====================

    def update_filter_parameters(
        self,
        manager_id: str,
        filter_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update filter parameters"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        manager.update_filter_parameters(filter_name, **parameters)

        return {
            'success': True,
            'message': f'Filter {filter_name} parameters updated',
            'parameters': parameters
        }

    # ==================== SCREENING ====================

    def screen_stocks(
        self,
        manager_id: str,
        symbols: List[str],
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE',
        logic: str = 'AND',
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Screen stocks using configured filters

        Args:
            manager_id: Manager ID
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            exchange: Exchange suffix
            logic: Combination logic ('AND' or 'OR')
            include_details: Include detailed filter results

        Returns:
            {
                'passed': {...},
                'summary': {...},
                'details': {...}  # if include_details=True
            }
        """
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        combine_logic = CombineLogic.AND if logic.upper() == 'AND' else CombineLogic.OR

        # Analyze stocks
        analyses = self.engine.analyze_multiple_stocks(
            symbols, period, interval, exchange
        )

        # Screen
        if include_details:
            results = manager.screen_with_details(analyses, combine_logic)
        else:
            passed = manager.screen_stocks(analyses, combine_logic)
            results = {
                'passed': {
                    symbol: {
                        'price': analysis['latest_price'],
                        'change_pct': analysis['price_change_pct'],
                        'trend': analysis['signals']['trend'],
                        'momentum': analysis['signals']['momentum']
                    }
                    for symbol, analysis in passed.items()
                },
                'summary': {
                    'total_stocks': len(analyses),
                    'passed': len(passed),
                    'failed': len(analyses) - len(passed),
                    'pass_rate': len(passed) / len(analyses) * 100 if analyses else 0
                }
            }

        return results

    def screen_by_type(
        self,
        manager_id: str,
        stock_type: str,
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE',
        limit: Optional[int] = None,
        logic: str = 'AND'
    ) -> Dict[str, Any]:
        """
        Screen stocks by type (NIFTY50, IT sector, etc.)

        Args:
            manager_id: Manager ID
            stock_type: Type of stocks
            period: Data period
            interval: Data interval
            exchange: Exchange suffix
            limit: Max stocks to analyze
            logic: Combination logic

        Returns:
            Screening results
        """
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        combine_logic = CombineLogic.AND if logic.upper() == 'AND' else CombineLogic.OR

        # Analyze by type
        analyses = self.engine.analyze_by_stock_type(
            stock_type, period, interval, exchange, limit=limit
        )

        # Screen
        passed = manager.screen_stocks(analyses, combine_logic)

        return {
            'stock_type': stock_type,
            'passed': {
                symbol: {
                    'price': analysis['latest_price'],
                    'change_pct': analysis['price_change_pct'],
                    'trend': analysis['signals']['trend'],
                    'momentum': analysis['signals']['momentum'],
                    'sector': analysis['classification']['sector'] if analysis.get('classification') else None
                }
                for symbol, analysis in passed.items()
            },
            'summary': {
                'total_stocks': len(analyses),
                'passed': len(passed),
                'failed': len(analyses) - len(passed),
                'pass_rate': len(passed) / len(analyses) * 100 if analyses else 0
            }
        }

    # ==================== TEST FILTER ====================

    def test_filter_on_stock(
        self,
        manager_id: str,
        symbol: str,
        period: str = '1mo',
        interval: str = '1d',
        exchange: str = 'NSE'
    ) -> Dict[str, Any]:
        """
        Test current filters on a single stock

        Args:
            manager_id: Manager ID
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            exchange: Exchange suffix

        Returns:
            {
                'symbol': 'RELIANCE',
                'passed': True/False,
                'filter_results': {...}
            }
        """
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]

        # Analyze stock
        analysis = self.engine.analyze_stock(symbol, period, interval, exchange)

        if not analysis:
            return {'error': f'Failed to analyze {symbol}'}

        # Get detailed filter results
        filter_results = manager.get_filter_results(analysis)

        return {
            'symbol': symbol,
            'passed': filter_results['overall_pass'],
            'price': analysis['latest_price'],
            'change_pct': analysis['price_change_pct'],
            'filter_results': filter_results
        }

    # ==================== EXPORT/IMPORT ====================

    def export_config(self, manager_id: str) -> Dict[str, Any]:
        """Export manager configuration as JSON"""
        if manager_id not in self.managers:
            return {'error': f'Manager {manager_id} not found'}

        manager = self.managers[manager_id]
        return manager.get_info()

    def import_config(self, manager_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Import manager configuration from JSON"""
        try:
            # Create new manager
            manager = FilterManager()

            # Import individual filters
            for filter_config in config.get('individual_filters', []):
                manager.add_filter(
                    filter_config['name'],
                    filter_config.get('parameters'),
                    filter_config.get('enabled', True)
                )

            # Import groups
            for group_config in config.get('groups', []):
                logic = CombineLogic.AND if group_config['logic'] == 'AND' else CombineLogic.OR
                manager.add_group(
                    group_config['name'],
                    group_config['filters'],
                    logic
                )

            self.managers[manager_id] = manager

            return {
                'success': True,
                'manager_id': manager_id,
                'message': 'Configuration imported'
            }
        except Exception as e:
            return {'error': str(e)}


# ==================== CONVENIENCE FUNCTIONS ====================

# Global API instance
_api_instance = None


def get_api() -> FilterAPI:
    """Get global API instance"""
    global _api_instance
    if _api_instance is None:
        _api_instance = FilterAPI()
    return _api_instance


# Quick functions for common operations
def quick_screen(
    symbols: List[str],
    filters: List[Dict[str, Any]],
    period: str = '1mo',
    logic: str = 'AND'
) -> Dict[str, Any]:
    """
    Quick screening with temporary manager

    Args:
        symbols: List of stock symbols
        filters: List of filter configs
        period: Data period
        logic: Combination logic

    Returns:
        Screening results
    """
    api = get_api()
    manager_id = 'temp_' + str(hash(str(filters)))

    # Create manager
    api.create_manager(manager_id)

    # Add filters
    for filter_config in filters:
        api.add_filter(
            manager_id,
            filter_config['name'],
            filter_config.get('parameters'),
            filter_config.get('enabled', True)
        )

    # Screen
    results = api.screen_stocks(manager_id, symbols, period, logic=logic)

    # Cleanup
    api.delete_manager(manager_id)

    return results


def preset_screen(
    preset: str,
    symbols: List[str],
    period: str = '1mo'
) -> Dict[str, Any]:
    """
    Quick screening with preset filters

    Args:
        preset: Preset name
        symbols: List of stock symbols
        period: Data period

    Returns:
        Screening results
    """
    api = get_api()
    manager_id = f'preset_{preset}'

    # Create manager with preset
    api.create_manager(manager_id, preset)

    # Screen
    results = api.screen_stocks(manager_id, symbols, period)

    # Cleanup
    api.delete_manager(manager_id)

    return results
