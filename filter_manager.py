"""
Filter Manager and Composer
Allows combining filters individually or in groups with AND/OR logic
"""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import json

from stock_filters import (
    BaseFilter, FilterRegistry, FilterType, FilterConfig
)


class CombineLogic(Enum):
    """Logic for combining filters"""
    AND = "AND"  # All filters must pass
    OR = "OR"    # At least one filter must pass


@dataclass
class FilterGroup:
    """Group of filters with combination logic"""
    name: str
    filters: List[BaseFilter] = field(default_factory=list)
    logic: CombineLogic = CombineLogic.AND
    enabled: bool = True

    def add_filter(self, filter_obj: BaseFilter):
        """Add a filter to this group"""
        self.filters.append(filter_obj)

    def remove_filter(self, filter_name: str):
        """Remove a filter by name"""
        self.filters = [f for f in self.filters if f.config.name != filter_name]

    def apply(self, analysis: Dict[str, Any]) -> bool:
        """Apply all filters in this group"""
        if not self.enabled or not self.filters:
            return True

        results = [f.apply(analysis) for f in self.filters if f.enabled]

        if not results:
            return True

        if self.logic == CombineLogic.AND:
            return all(results)
        else:  # OR
            return any(results)

    def get_info(self) -> Dict[str, Any]:
        """Get group information"""
        return {
            'name': self.name,
            'logic': self.logic.value,
            'enabled': self.enabled,
            'filter_count': len(self.filters),
            'filters': [f.get_info() for f in self.filters]
        }


class FilterManager:
    """
    Manage filters individually or in groups
    Apply filters one by one or in combination
    """

    def __init__(self):
        """Initialize filter manager"""
        self.filters: List[BaseFilter] = []
        self.groups: List[FilterGroup] = []
        self.group_logic: CombineLogic = CombineLogic.AND

    # ==================== ADD FILTERS ====================

    def add_filter(
        self,
        filter_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> BaseFilter:
        """
        Add an individual filter

        Args:
            filter_name: Name of filter (e.g., 'rsi', 'macd', 'price')
            parameters: Filter parameters
            enabled: Whether filter is enabled

        Returns:
            The created filter instance
        """
        # Create config
        filter_instance = FilterRegistry.get_filter(filter_name)

        if parameters:
            filter_instance.update_parameters(**parameters)

        filter_instance.enabled = enabled

        self.filters.append(filter_instance)
        return filter_instance

    def add_filter_instance(self, filter_obj: BaseFilter):
        """Add an existing filter instance"""
        self.filters.append(filter_obj)

    def add_group(
        self,
        group_name: str,
        filter_configs: List[Dict[str, Any]],
        logic: CombineLogic = CombineLogic.AND
    ) -> FilterGroup:
        """
        Add a filter group

        Args:
            group_name: Name for the group
            filter_configs: List of filter configs, each with:
                {
                    'name': 'rsi',
                    'parameters': {'min': 30, 'max': 70},
                    'enabled': True
                }
            logic: Combination logic (AND/OR)

        Returns:
            The created filter group
        """
        group = FilterGroup(name=group_name, logic=logic)

        for config in filter_configs:
            filter_obj = self.add_filter(
                config['name'],
                config.get('parameters'),
                config.get('enabled', True)
            )
            # Remove from individual filters and add to group
            self.filters.remove(filter_obj)
            group.add_filter(filter_obj)

        self.groups.append(group)
        return group

    # ==================== REMOVE FILTERS ====================

    def remove_filter(self, filter_name: str):
        """Remove a filter by name"""
        self.filters = [f for f in self.filters if f.config.name != filter_name]

    def remove_group(self, group_name: str):
        """Remove a group by name"""
        self.groups = [g for g in self.groups if g.name != group_name]

    def clear_all(self):
        """Clear all filters and groups"""
        self.filters.clear()
        self.groups.clear()

    # ==================== ENABLE/DISABLE ====================

    def enable_filter(self, filter_name: str):
        """Enable a specific filter"""
        for f in self.filters:
            if f.config.name == filter_name:
                f.enable()
                return

        # Check in groups
        for group in self.groups:
            for f in group.filters:
                if f.config.name == filter_name:
                    f.enable()
                    return

    def disable_filter(self, filter_name: str):
        """Disable a specific filter"""
        for f in self.filters:
            if f.config.name == filter_name:
                f.disable()
                return

        # Check in groups
        for group in self.groups:
            for f in group.filters:
                if f.config.name == filter_name:
                    f.disable()
                    return

    def enable_group(self, group_name: str):
        """Enable a filter group"""
        for group in self.groups:
            if group.name == group_name:
                group.enabled = True
                return

    def disable_group(self, group_name: str):
        """Disable a filter group"""
        for group in self.groups:
            if group.name == group_name:
                group.enabled = False
                return

    def enable_all(self):
        """Enable all filters and groups"""
        for f in self.filters:
            f.enable()
        for group in self.groups:
            group.enabled = True

    def disable_all(self):
        """Disable all filters and groups"""
        for f in self.filters:
            f.disable()
        for group in self.groups:
            group.enabled = False

    # ==================== UPDATE PARAMETERS ====================

    def update_filter_parameters(self, filter_name: str, **parameters):
        """Update parameters for a specific filter"""
        for f in self.filters:
            if f.config.name == filter_name:
                f.update_parameters(**parameters)
                return

        # Check in groups
        for group in self.groups:
            for f in group.filters:
                if f.config.name == filter_name:
                    f.update_parameters(**parameters)
                    return

    # ==================== APPLY FILTERS ====================

    def apply_individual(
        self,
        analysis: Dict[str, Any],
        filter_name: str
    ) -> bool:
        """
        Apply a single filter individually

        Args:
            analysis: Stock analysis dictionary
            filter_name: Name of filter to apply

        Returns:
            True if stock passes filter
        """
        for f in self.filters:
            if f.config.name == filter_name:
                return f.apply(analysis)

        # Check in groups
        for group in self.groups:
            for f in group.filters:
                if f.config.name == filter_name:
                    return f.apply(analysis)

        return False

    def apply_all(
        self,
        analysis: Dict[str, Any],
        logic: Optional[CombineLogic] = None
    ) -> bool:
        """
        Apply all filters

        Args:
            analysis: Stock analysis dictionary
            logic: Combination logic (AND/OR). If None, uses self.group_logic

        Returns:
            True if stock passes all filters
        """
        if logic is None:
            logic = self.group_logic

        # Apply individual filters
        individual_results = [f.apply(analysis) for f in self.filters if f.enabled]

        # Apply groups
        group_results = [g.apply(analysis) for g in self.groups if g.enabled]

        # Combine all results
        all_results = individual_results + group_results

        if not all_results:
            return True

        if logic == CombineLogic.AND:
            return all(all_results)
        else:  # OR
            return any(all_results)

    def apply_by_type(
        self,
        analysis: Dict[str, Any],
        filter_type: FilterType,
        logic: CombineLogic = CombineLogic.AND
    ) -> bool:
        """
        Apply all filters of a specific type

        Args:
            analysis: Stock analysis dictionary
            filter_type: Type of filters to apply
            logic: Combination logic

        Returns:
            True if stock passes filters
        """
        results = []

        # Apply individual filters of this type
        for f in self.filters:
            if f.enabled and f.config.filter_type == filter_type:
                results.append(f.apply(analysis))

        # Apply groups (check if any filter in group matches type)
        for group in self.groups:
            if not group.enabled:
                continue
            type_filters = [f for f in group.filters if f.config.filter_type == filter_type]
            if type_filters:
                # Create temporary group with only this type
                temp_results = [f.apply(analysis) for f in type_filters if f.enabled]
                if temp_results:
                    if group.logic == CombineLogic.AND:
                        results.append(all(temp_results))
                    else:
                        results.append(any(temp_results))

        if not results:
            return True

        if logic == CombineLogic.AND:
            return all(results)
        else:
            return any(results)

    def get_filter_results(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed results for each filter

        Args:
            analysis: Stock analysis dictionary

        Returns:
            Dictionary with results for each filter
        """
        results = {
            'individual_filters': {},
            'groups': {},
            'overall_pass': self.apply_all(analysis)
        }

        # Individual filters
        for f in self.filters:
            if f.enabled:
                results['individual_filters'][f.config.name] = {
                    'passed': f.apply(analysis),
                    'type': f.config.filter_type.value,
                    'parameters': f.config.parameters
                }

        # Groups
        for group in self.groups:
            if group.enabled:
                filter_results = {}
                for f in group.filters:
                    if f.enabled:
                        filter_results[f.config.name] = {
                            'passed': f.apply(analysis),
                            'parameters': f.config.parameters
                        }

                results['groups'][group.name] = {
                    'passed': group.apply(analysis),
                    'logic': group.logic.value,
                    'filters': filter_results
                }

        return results

    # ==================== SCREENING ====================

    def screen_stocks(
        self,
        analyses: Dict[str, Dict[str, Any]],
        logic: Optional[CombineLogic] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Screen multiple stocks using current filters

        Args:
            analyses: Dictionary mapping symbol to analysis
            logic: Combination logic

        Returns:
            Dictionary of stocks that pass filters
        """
        passed = {}

        for symbol, analysis in analyses.items():
            if self.apply_all(analysis, logic):
                passed[symbol] = analysis

        return passed

    def screen_with_details(
        self,
        analyses: Dict[str, Dict[str, Any]],
        logic: Optional[CombineLogic] = None
    ) -> Dict[str, Any]:
        """
        Screen stocks with detailed filter results

        Args:
            analyses: Dictionary mapping symbol to analysis
            logic: Combination logic

        Returns:
            Dictionary with passed stocks and filter details
        """
        results = {
            'passed': {},
            'failed': {},
            'filter_summary': {}
        }

        for symbol, analysis in analyses.items():
            filter_results = self.get_filter_results(analysis)

            if filter_results['overall_pass']:
                results['passed'][symbol] = {
                    'analysis': analysis,
                    'filter_results': filter_results
                }
            else:
                results['failed'][symbol] = {
                    'analysis': analysis,
                    'filter_results': filter_results
                }

        # Summary
        results['filter_summary'] = {
            'total_stocks': len(analyses),
            'passed': len(results['passed']),
            'failed': len(results['failed']),
            'pass_rate': len(results['passed']) / len(analyses) * 100 if analyses else 0
        }

        return results

    # ==================== INFO & EXPORT ====================

    def get_info(self) -> Dict[str, Any]:
        """Get information about all filters"""
        return {
            'individual_filters': [f.get_info() for f in self.filters],
            'groups': [g.get_info() for g in self.groups],
            'group_logic': self.group_logic.value,
            'total_filters': len(self.filters),
            'total_groups': len(self.groups)
        }

    def list_filters(self) -> List[str]:
        """List all filter names"""
        names = [f.config.name for f in self.filters]

        for group in self.groups:
            names.extend([f.config.name for f in group.filters])

        return names

    def list_filters_by_type(self) -> Dict[str, List[str]]:
        """List filters organized by type"""
        by_type = {}

        all_filters = self.filters.copy()
        for group in self.groups:
            all_filters.extend(group.filters)

        for f in all_filters:
            filter_type = f.config.filter_type.value
            if filter_type not in by_type:
                by_type[filter_type] = []
            by_type[filter_type].append(f.config.name)

        return by_type

    def export_config(self, filename: str):
        """Export filter configuration to JSON file"""
        config = {
            'individual_filters': [
                {
                    'name': f.config.name,
                    'type': f.config.filter_type.value,
                    'enabled': f.enabled,
                    'parameters': f.config.parameters
                }
                for f in self.filters
            ],
            'groups': [
                {
                    'name': g.name,
                    'logic': g.logic.value,
                    'enabled': g.enabled,
                    'filters': [
                        {
                            'name': f.config.name,
                            'type': f.config.filter_type.value,
                            'enabled': f.enabled,
                            'parameters': f.config.parameters
                        }
                        for f in g.filters
                    ]
                }
                for g in self.groups
            ],
            'group_logic': self.group_logic.value
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

    def import_config(self, filename: str):
        """Import filter configuration from JSON file"""
        with open(filename, 'r') as f:
            config = json.load(f)

        # Clear existing
        self.clear_all()

        # Import individual filters
        for filter_config in config.get('individual_filters', []):
            self.add_filter(
                filter_config['name'],
                filter_config.get('parameters'),
                filter_config.get('enabled', True)
            )

        # Import groups
        for group_config in config.get('groups', []):
            logic = CombineLogic.AND if group_config['logic'] == 'AND' else CombineLogic.OR
            self.add_group(
                group_config['name'],
                group_config['filters'],
                logic
            )

        # Set group logic
        self.group_logic = CombineLogic.AND if config.get('group_logic') == 'AND' else CombineLogic.OR


# ==================== PRESET FILTER CONFIGURATIONS ====================

class FilterPresets:
    """Pre-configured filter setups for common scenarios"""

    @staticmethod
    def momentum_trading() -> FilterManager:
        """Filters for momentum trading"""
        manager = FilterManager()

        manager.add_filter('rsi', {'min': 40, 'max': 80})
        manager.add_filter('trend', {'direction': 'BULLISH'})
        manager.add_filter('volume', {'state': 'HIGH'})
        manager.add_filter('liquidity', {'levels': ['HIGH']})

        return manager

    @staticmethod
    def value_investing() -> FilterManager:
        """Filters for value investing"""
        manager = FilterManager()

        manager.add_filter('pe_ratio', {'min': 0, 'max': 20})
        manager.add_filter('pb_ratio', {'min': 0, 'max': 3})
        manager.add_filter('market_cap', {'categories': ['LARGE', 'MID']})

        return manager

    @staticmethod
    def oversold_bounce() -> FilterManager:
        """Filters for oversold bounce plays"""
        manager = FilterManager()

        manager.add_filter('rsi', {'min': 20, 'max': 35})
        manager.add_filter('trend', {'direction': 'BULLISH'})
        manager.add_filter('volume_spike', {'required': True})

        return manager

    @staticmethod
    def breakout_trading() -> FilterManager:
        """Filters for breakout trading"""
        manager = FilterManager()

        manager.add_filter('adx', {'min': 25})
        manager.add_filter('volume', {'state': 'HIGH'})
        manager.add_filter('bollinger_bands', {'position': 'outside_upper'})
        manager.add_filter('liquidity', {'levels': ['HIGH']})

        return manager

    @staticmethod
    def swing_trading() -> FilterManager:
        """Filters for swing trading"""
        manager = FilterManager()

        manager.add_filter('rsi', {'min': 30, 'max': 70})
        manager.add_filter('macd', {'signal': 'bullish'})
        manager.add_filter('trend', {'direction': 'BULLISH'})
        manager.add_filter('atr', {'min': 10, 'max': 100})

        return manager

    @staticmethod
    def conservative() -> FilterManager:
        """Conservative filters for low-risk stocks"""
        manager = FilterManager()

        manager.add_filter('market_cap', {'categories': ['LARGE']})
        manager.add_filter('liquidity', {'levels': ['HIGH']})
        manager.add_filter('volatility', {'levels': ['LOW', 'MEDIUM']})
        manager.add_filter('beta', {'min': 0, 'max': 1.2})

        return manager


# Convenience functions
def create_filter_manager(preset: Optional[str] = None) -> FilterManager:
    """
    Create a filter manager with optional preset

    Args:
        preset: Preset name ('momentum_trading', 'value_investing', etc.)

    Returns:
        FilterManager instance
    """
    if preset:
        preset_method = getattr(FilterPresets, preset, None)
        if preset_method:
            return preset_method()

    return FilterManager()
