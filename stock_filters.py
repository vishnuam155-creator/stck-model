"""
Individual Stock Filters
Each filter can be applied individually or combined
Organized by type: Technical, Fundamental, Classification
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
import pandas as pd
from enum import Enum


class FilterType(Enum):
    """Filter category types"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    CLASSIFICATION = "classification"
    PRICE = "price"
    VOLUME = "volume"
    PATTERN = "pattern"


class FilterOperator(Enum):
    """Comparison operators for filters"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "=="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"


@dataclass
class FilterConfig:
    """Configuration for a filter"""
    name: str
    filter_type: FilterType
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BaseFilter(ABC):
    """Base class for all filters"""

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize filter

        Args:
            config: Filter configuration
        """
        self.config = config or self._default_config()
        self.enabled = self.config.enabled

    @abstractmethod
    def _default_config(self) -> FilterConfig:
        """Return default configuration"""
        pass

    @abstractmethod
    def apply(self, analysis: Dict[str, Any]) -> bool:
        """
        Apply filter to stock analysis

        Args:
            analysis: Stock analysis dictionary

        Returns:
            True if stock passes filter, False otherwise
        """
        pass

    def enable(self):
        """Enable this filter"""
        self.enabled = True
        self.config.enabled = True

    def disable(self):
        """Disable this filter"""
        self.enabled = False
        self.config.enabled = False

    def update_parameters(self, **kwargs):
        """Update filter parameters"""
        self.config.parameters.update(kwargs)

    def get_info(self) -> Dict[str, Any]:
        """Get filter information"""
        return {
            'name': self.config.name,
            'type': self.config.filter_type.value,
            'description': self.config.description,
            'enabled': self.enabled,
            'parameters': self.config.parameters
        }

    def _compare(self, value: float, operator: FilterOperator, threshold: Any) -> bool:
        """Helper method to compare values"""
        if operator == FilterOperator.GREATER_THAN:
            return value > threshold
        elif operator == FilterOperator.LESS_THAN:
            return value < threshold
        elif operator == FilterOperator.EQUAL:
            return value == threshold
        elif operator == FilterOperator.GREATER_EQUAL:
            return value >= threshold
        elif operator == FilterOperator.LESS_EQUAL:
            return value <= threshold
        elif operator == FilterOperator.BETWEEN:
            return threshold[0] <= value <= threshold[1]
        return False


# ==================== TECHNICAL FILTERS ====================

class RSIFilter(BaseFilter):
    """Filter stocks by RSI value"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="RSI Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by Relative Strength Index",
            parameters={
                'min': 30,
                'max': 70,
                'period': 14
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        rsi_key = f"RSI_{self.config.parameters['period']}"
        rsi = indicators.get(rsi_key)

        if rsi is None:
            return False

        min_rsi = self.config.parameters.get('min', 0)
        max_rsi = self.config.parameters.get('max', 100)

        return min_rsi <= rsi <= max_rsi


class MACDFilter(BaseFilter):
    """Filter stocks by MACD signal"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="MACD Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by MACD crossover",
            parameters={
                'signal': 'bullish'  # bullish, bearish, any
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_SIGNAL')

        if macd is None or macd_signal is None:
            return False

        signal_type = self.config.parameters.get('signal', 'any')

        if signal_type == 'bullish':
            return macd > macd_signal
        elif signal_type == 'bearish':
            return macd < macd_signal
        else:  # any
            return True


class BollingerBandsFilter(BaseFilter):
    """Filter stocks by Bollinger Bands position"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Bollinger Bands Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by position relative to Bollinger Bands",
            parameters={
                'position': 'middle',  # upper, lower, middle, outside_upper, outside_lower
                'period': 20
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        price = analysis.get('latest_price')
        bb_upper = indicators.get('BB_UPPER')
        bb_lower = indicators.get('BB_LOWER')
        bb_middle = indicators.get('BB_MIDDLE')

        if None in [price, bb_upper, bb_lower, bb_middle]:
            return False

        position = self.config.parameters.get('position', 'middle')

        if position == 'upper':
            return price >= bb_middle and price <= bb_upper
        elif position == 'lower':
            return price >= bb_lower and price <= bb_middle
        elif position == 'middle':
            return bb_lower < price < bb_upper
        elif position == 'outside_upper':
            return price > bb_upper
        elif position == 'outside_lower':
            return price < bb_lower

        return True


class ATRFilter(BaseFilter):
    """Filter stocks by ATR (volatility)"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="ATR Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by Average True Range (volatility)",
            parameters={
                'min': 0,
                'max': 1000,
                'period': 14
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        atr_key = f"ATR_{self.config.parameters['period']}"
        atr = indicators.get(atr_key)

        if atr is None:
            return False

        min_atr = self.config.parameters.get('min', 0)
        max_atr = self.config.parameters.get('max', 1000)

        return min_atr <= atr <= max_atr


class VWAPFilter(BaseFilter):
    """Filter stocks by VWAP position"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="VWAP Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by position relative to VWAP",
            parameters={
                'position': 'above'  # above, below, any
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        price = analysis.get('latest_price')
        vwap = indicators.get('VWAP')

        if price is None or vwap is None:
            return False

        position = self.config.parameters.get('position', 'any')

        if position == 'above':
            return price > vwap
        elif position == 'below':
            return price < vwap
        else:  # any
            return True


class StochasticFilter(BaseFilter):
    """Filter stocks by Stochastic oscillator"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Stochastic Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by Stochastic oscillator",
            parameters={
                'min_k': 0,
                'max_k': 100,
                'signal': 'any'  # bullish, bearish, any
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        stoch_k = indicators.get('STOCH_K')
        stoch_d = indicators.get('STOCH_D')

        if stoch_k is None:
            return False

        min_k = self.config.parameters.get('min_k', 0)
        max_k = self.config.parameters.get('max_k', 100)

        # Check K value range
        if not (min_k <= stoch_k <= max_k):
            return False

        # Check signal
        signal = self.config.parameters.get('signal', 'any')
        if signal == 'bullish' and stoch_d is not None:
            return stoch_k > stoch_d
        elif signal == 'bearish' and stoch_d is not None:
            return stoch_k < stoch_d

        return True


class ADXFilter(BaseFilter):
    """Filter stocks by ADX (trend strength)"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="ADX Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by Average Directional Index (trend strength)",
            parameters={
                'min': 25,  # Strong trend above 25
                'period': 14
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        indicators = analysis.get('indicators', {})
        adx = indicators.get('ADX')

        if adx is None:
            return False

        min_adx = self.config.parameters.get('min', 0)
        return adx >= min_adx


class TrendFilter(BaseFilter):
    """Filter stocks by trend direction"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Trend Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by trend direction",
            parameters={
                'direction': 'BULLISH'  # BULLISH, BEARISH, NEUTRAL, any
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        signals = analysis.get('signals', {})
        trend = signals.get('trend')

        if trend is None:
            return False

        expected_trend = self.config.parameters.get('direction', 'any').upper()

        if expected_trend == 'ANY':
            return True

        return trend == expected_trend


class MomentumFilter(BaseFilter):
    """Filter stocks by momentum"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Momentum Filter",
            filter_type=FilterType.TECHNICAL,
            description="Filter by momentum state",
            parameters={
                'state': 'BULLISH'  # BULLISH, BEARISH, OVERSOLD, OVERBOUGHT, any
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        signals = analysis.get('signals', {})
        momentum = signals.get('momentum')

        if momentum is None:
            return False

        expected_momentum = self.config.parameters.get('state', 'any').upper()

        if expected_momentum == 'ANY':
            return True

        return momentum == expected_momentum


# ==================== PRICE FILTERS ====================

class PriceFilter(BaseFilter):
    """Filter stocks by price range"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Price Filter",
            filter_type=FilterType.PRICE,
            description="Filter by price range",
            parameters={
                'min': 0,
                'max': 10000
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        price = analysis.get('latest_price')

        if price is None:
            return False

        min_price = self.config.parameters.get('min', 0)
        max_price = self.config.parameters.get('max', 100000)

        return min_price <= price <= max_price


class PriceChangeFilter(BaseFilter):
    """Filter stocks by price change percentage"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Price Change Filter",
            filter_type=FilterType.PRICE,
            description="Filter by price change percentage",
            parameters={
                'min': -100,
                'max': 100
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        price_change = analysis.get('price_change_pct')

        if price_change is None:
            return False

        min_change = self.config.parameters.get('min', -100)
        max_change = self.config.parameters.get('max', 100)

        return min_change <= price_change <= max_change


# ==================== VOLUME FILTERS ====================

class VolumeFilter(BaseFilter):
    """Filter stocks by volume state"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Volume Filter",
            filter_type=FilterType.VOLUME,
            description="Filter by volume state",
            parameters={
                'state': 'HIGH'  # HIGH, NORMAL, LOW, any
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        signals = analysis.get('signals', {})
        volume = signals.get('volume')

        if volume is None:
            return False

        expected_volume = self.config.parameters.get('state', 'any').upper()

        if expected_volume == 'ANY':
            return True

        return volume == expected_volume


class VolumeSpikeFilter(BaseFilter):
    """Filter stocks with volume spike"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Volume Spike Filter",
            filter_type=FilterType.VOLUME,
            description="Filter stocks with volume spike",
            parameters={
                'required': True
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        signals = analysis.get('signals', {})
        volume = signals.get('volume')

        if volume is None:
            return False

        required = self.config.parameters.get('required', False)

        if required:
            return volume == 'HIGH'
        else:
            return True


# ==================== CLASSIFICATION FILTERS ====================

class MarketCapFilter(BaseFilter):
    """Filter stocks by market capitalization"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Market Cap Filter",
            filter_type=FilterType.CLASSIFICATION,
            description="Filter by market capitalization category",
            parameters={
                'categories': ['LARGE']  # LARGE, MID, SMALL
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        market_cap = classification.get('market_cap_category')

        if market_cap is None:
            return False

        allowed_categories = self.config.parameters.get('categories', [])

        if not allowed_categories:
            return True

        return market_cap in allowed_categories


class SectorFilter(BaseFilter):
    """Filter stocks by sector"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Sector Filter",
            filter_type=FilterType.CLASSIFICATION,
            description="Filter by sector",
            parameters={
                'sectors': []  # List of allowed sectors
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        sector = classification.get('sector')

        if sector is None:
            return False

        allowed_sectors = self.config.parameters.get('sectors', [])

        if not allowed_sectors:
            return True

        return sector in allowed_sectors


class LiquidityFilter(BaseFilter):
    """Filter stocks by liquidity"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Liquidity Filter",
            filter_type=FilterType.CLASSIFICATION,
            description="Filter by liquidity level",
            parameters={
                'levels': ['HIGH']  # HIGH, MEDIUM, LOW
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        liquidity = classification.get('liquidity')

        if liquidity is None:
            return False

        allowed_levels = self.config.parameters.get('levels', [])

        if not allowed_levels:
            return True

        return liquidity in allowed_levels


class VolatilityFilter(BaseFilter):
    """Filter stocks by volatility"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Volatility Filter",
            filter_type=FilterType.CLASSIFICATION,
            description="Filter by volatility level",
            parameters={
                'levels': ['MEDIUM']  # HIGH, MEDIUM, LOW
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        volatility = classification.get('volatility')

        if volatility is None:
            return False

        allowed_levels = self.config.parameters.get('levels', [])

        if not allowed_levels:
            return True

        return volatility in allowed_levels


class IndexMembershipFilter(BaseFilter):
    """Filter stocks by index membership"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Index Membership Filter",
            filter_type=FilterType.CLASSIFICATION,
            description="Filter by index membership",
            parameters={
                'indices': ['NIFTY50']  # NIFTY50, NIFTY100
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        membership = classification.get('index_membership', [])

        required_indices = self.config.parameters.get('indices', [])

        if not required_indices:
            return True

        # Stock must be in at least one of the required indices
        return any(index in membership for index in required_indices)


# ==================== FUNDAMENTAL FILTERS ====================

class PERatioFilter(BaseFilter):
    """Filter stocks by P/E ratio"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="P/E Ratio Filter",
            filter_type=FilterType.FUNDAMENTAL,
            description="Filter by Price-to-Earnings ratio",
            parameters={
                'min': 0,
                'max': 100
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        pe_ratio = classification.get('pe_ratio')

        if pe_ratio is None or pe_ratio <= 0:
            return False

        min_pe = self.config.parameters.get('min', 0)
        max_pe = self.config.parameters.get('max', 1000)

        return min_pe <= pe_ratio <= max_pe


class PBRatioFilter(BaseFilter):
    """Filter stocks by P/B ratio"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="P/B Ratio Filter",
            filter_type=FilterType.FUNDAMENTAL,
            description="Filter by Price-to-Book ratio",
            parameters={
                'min': 0,
                'max': 10
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        pb_ratio = classification.get('pb_ratio')

        if pb_ratio is None or pb_ratio <= 0:
            return False

        min_pb = self.config.parameters.get('min', 0)
        max_pb = self.config.parameters.get('max', 1000)

        return min_pb <= pb_ratio <= max_pb


class BetaFilter(BaseFilter):
    """Filter stocks by Beta (market risk)"""

    def _default_config(self) -> FilterConfig:
        return FilterConfig(
            name="Beta Filter",
            filter_type=FilterType.FUNDAMENTAL,
            description="Filter by Beta (market risk)",
            parameters={
                'min': 0,
                'max': 2
            }
        )

    def apply(self, analysis: Dict[str, Any]) -> bool:
        if not self.enabled:
            return True

        classification = analysis.get('classification', {})
        if not classification:
            return False

        beta = classification.get('beta')

        if beta is None:
            return False

        min_beta = self.config.parameters.get('min', 0)
        max_beta = self.config.parameters.get('max', 10)

        return min_beta <= beta <= max_beta


# ==================== FILTER REGISTRY ====================

class FilterRegistry:
    """Registry of all available filters"""

    _filters = {
        # Technical
        'rsi': RSIFilter,
        'macd': MACDFilter,
        'bollinger_bands': BollingerBandsFilter,
        'atr': ATRFilter,
        'vwap': VWAPFilter,
        'stochastic': StochasticFilter,
        'adx': ADXFilter,
        'trend': TrendFilter,
        'momentum': MomentumFilter,

        # Price
        'price': PriceFilter,
        'price_change': PriceChangeFilter,

        # Volume
        'volume': VolumeFilter,
        'volume_spike': VolumeSpikeFilter,

        # Classification
        'market_cap': MarketCapFilter,
        'sector': SectorFilter,
        'liquidity': LiquidityFilter,
        'volatility': VolatilityFilter,
        'index_membership': IndexMembershipFilter,

        # Fundamental
        'pe_ratio': PERatioFilter,
        'pb_ratio': PBRatioFilter,
        'beta': BetaFilter,
    }

    @classmethod
    def get_filter(cls, filter_name: str, config: Optional[FilterConfig] = None) -> BaseFilter:
        """Get filter instance by name"""
        filter_class = cls._filters.get(filter_name.lower())
        if filter_class is None:
            raise ValueError(f"Unknown filter: {filter_name}")
        return filter_class(config)

    @classmethod
    def list_filters(cls) -> List[str]:
        """List all available filter names"""
        return list(cls._filters.keys())

    @classmethod
    def get_filters_by_type(cls, filter_type: FilterType) -> List[str]:
        """Get filter names by type"""
        result = []
        for name, filter_class in cls._filters.items():
            instance = filter_class()
            if instance.config.filter_type == filter_type:
                result.append(name)
        return result
