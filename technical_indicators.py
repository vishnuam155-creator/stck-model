"""
Comprehensive Technical Indicators Library
Supports 20+ indicators across all categories: Trend, Momentum, Volatility, Volume
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pandas_ta as ta


class TechnicalIndicators:
    """
    Complete technical indicators library with all major indicators.
    Categories: Trend, Momentum, Volatility, Volume, Custom
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV dataframe

        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df.copy()
        self.results = {}

    # ==================== TREND INDICATORS ====================

    def sma(self, period: int = 20, column: str = 'Close') -> pd.Series:
        """Simple Moving Average"""
        sma = self.df[column].rolling(window=period).mean()
        self.results[f'SMA_{period}'] = sma
        return sma

    def ema(self, period: int = 20, column: str = 'Close') -> pd.Series:
        """Exponential Moving Average"""
        ema = self.df[column].ewm(span=period, adjust=False).mean()
        self.results[f'EMA_{period}'] = ema
        return ema

    def wma(self, period: int = 20, column: str = 'Close') -> pd.Series:
        """Weighted Moving Average"""
        wma = ta.wma(self.df[column], length=period)
        self.results[f'WMA_{period}'] = wma
        return wma

    def dema(self, period: int = 20, column: str = 'Close') -> pd.Series:
        """Double Exponential Moving Average"""
        dema = ta.dema(self.df[column], length=period)
        self.results[f'DEMA_{period}'] = dema
        return dema

    def tema(self, period: int = 20, column: str = 'Close') -> pd.Series:
        """Triple Exponential Moving Average"""
        tema = ta.tema(self.df[column], length=period)
        self.results[f'TEMA_{period}'] = tema
        return tema

    def vwap(self) -> pd.Series:
        """Volume Weighted Average Price"""
        vwap = ta.vwap(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])
        self.results['VWAP'] = vwap
        return vwap

    def supertrend(self, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """SuperTrend Indicator"""
        supertrend_df = ta.supertrend(
            self.df['High'],
            self.df['Low'],
            self.df['Close'],
            length=period,
            multiplier=multiplier
        )
        if supertrend_df is not None and not supertrend_df.empty:
            trend = supertrend_df.iloc[:, 1]  # SUPERT_10_3.0 (direction)
            value = supertrend_df.iloc[:, 0]  # SUPERTd_10_3.0 (value)
            self.results['SUPERTREND'] = value
            self.results['SUPERTREND_DIRECTION'] = trend
            return value, trend
        return pd.Series(), pd.Series()

    def adx(self, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        adx_df = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=period)
        if adx_df is not None and not adx_df.empty:
            result = {
                'ADX': adx_df[f'ADX_{period}'],
                'DI_PLUS': adx_df[f'DMP_{period}'],
                'DI_MINUS': adx_df[f'DMN_{period}']
            }
            self.results.update(result)
            return result
        return {}

    def ichimoku(self) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        ichimoku_df = ta.ichimoku(self.df['High'], self.df['Low'], self.df['Close'])
        if ichimoku_df is not None and not ichimoku_df.empty:
            result = {
                'ICHIMOKU_CONVERSION': ichimoku_df[0].iloc[:, 0],  # Tenkan-sen
                'ICHIMOKU_BASE': ichimoku_df[0].iloc[:, 1],         # Kijun-sen
                'ICHIMOKU_SPAN_A': ichimoku_df[0].iloc[:, 2],       # Senkou Span A
                'ICHIMOKU_SPAN_B': ichimoku_df[0].iloc[:, 3],       # Senkou Span B
            }
            self.results.update(result)
            return result
        return {}

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, period: int = 14, column: str = 'Close') -> pd.Series:
        """Relative Strength Index"""
        rsi = ta.rsi(self.df[column], length=period)
        self.results[f'RSI_{period}'] = rsi
        return rsi

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence"""
        macd_df = ta.macd(self.df['Close'], fast=fast, slow=slow, signal=signal)
        if macd_df is not None and not macd_df.empty:
            result = {
                'MACD': macd_df[f'MACD_{fast}_{slow}_{signal}'],
                'MACD_SIGNAL': macd_df[f'MACDs_{fast}_{slow}_{signal}'],
                'MACD_HISTOGRAM': macd_df[f'MACDh_{fast}_{slow}_{signal}']
            }
            self.results.update(result)
            return result
        return {}

    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        stoch_df = ta.stoch(
            self.df['High'],
            self.df['Low'],
            self.df['Close'],
            k=k_period,
            d=d_period
        )
        if stoch_df is not None and not stoch_df.empty:
            result = {
                'STOCH_K': stoch_df[f'STOCHk_{k_period}_{d_period}_3'],
                'STOCH_D': stoch_df[f'STOCHd_{k_period}_{d_period}_3']
            }
            self.results.update(result)
            return result
        return {}

    def cci(self, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        cci = ta.cci(self.df['High'], self.df['Low'], self.df['Close'], length=period)
        self.results[f'CCI_{period}'] = cci
        return cci

    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R"""
        willr = ta.willr(self.df['High'], self.df['Low'], self.df['Close'], length=period)
        self.results[f'WILLIAMS_R_{period}'] = willr
        return willr

    def roc(self, period: int = 12, column: str = 'Close') -> pd.Series:
        """Rate of Change"""
        roc = ta.roc(self.df[column], length=period)
        self.results[f'ROC_{period}'] = roc
        return roc

    def mfi(self, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        mfi = ta.mfi(
            self.df['High'],
            self.df['Low'],
            self.df['Close'],
            self.df['Volume'],
            length=period
        )
        self.results[f'MFI_{period}'] = mfi
        return mfi

    def ultimate_oscillator(self) -> pd.Series:
        """Ultimate Oscillator"""
        uo = ta.uo(self.df['High'], self.df['Low'], self.df['Close'])
        self.results['ULTIMATE_OSCILLATOR'] = uo
        return uo

    # ==================== VOLATILITY INDICATORS ====================

    def bollinger_bands(self, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        bb_df = ta.bbands(self.df['Close'], length=period, std=std)
        if bb_df is not None and not bb_df.empty:
            result = {
                'BB_UPPER': bb_df[f'BBU_{period}_{std}'],
                'BB_MIDDLE': bb_df[f'BBM_{period}_{std}'],
                'BB_LOWER': bb_df[f'BBL_{period}_{std}'],
                'BB_WIDTH': bb_df[f'BBB_{period}_{std}'],
                'BB_PERCENT': bb_df[f'BBP_{period}_{std}']
            }
            self.results.update(result)
            return result
        return {}

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range"""
        atr = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=period)
        self.results[f'ATR_{period}'] = atr
        return atr

    def keltner_channels(self, period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """Keltner Channels"""
        kc_df = ta.kc(
            self.df['High'],
            self.df['Low'],
            self.df['Close'],
            length=period,
            scalar=multiplier
        )
        if kc_df is not None and not kc_df.empty:
            result = {
                'KC_UPPER': kc_df[f'KCUe_{period}_{multiplier}'],
                'KC_MIDDLE': kc_df[f'KCBe_{period}_{multiplier}'],
                'KC_LOWER': kc_df[f'KCLe_{period}_{multiplier}']
            }
            self.results.update(result)
            return result
        return {}

    def donchian_channels(self, period: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channels"""
        dc_df = ta.donchian(self.df['High'], self.df['Low'], lower_length=period, upper_length=period)
        if dc_df is not None and not dc_df.empty:
            result = {
                'DC_UPPER': dc_df[f'DCU_{period}_{period}'],
                'DC_MIDDLE': dc_df[f'DCM_{period}_{period}'],
                'DC_LOWER': dc_df[f'DCL_{period}_{period}']
            }
            self.results.update(result)
            return result
        return {}

    def historical_volatility(self, period: int = 20) -> pd.Series:
        """Historical Volatility (annualized)"""
        log_returns = np.log(self.df['Close'] / self.df['Close'].shift(1))
        hv = log_returns.rolling(window=period).std() * np.sqrt(252) * 100
        self.results[f'HV_{period}'] = hv
        return hv

    # ==================== VOLUME INDICATORS ====================

    def obv(self) -> pd.Series:
        """On Balance Volume"""
        obv = ta.obv(self.df['Close'], self.df['Volume'])
        self.results['OBV'] = obv
        return obv

    def ad(self) -> pd.Series:
        """Accumulation/Distribution"""
        ad = ta.ad(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])
        self.results['AD'] = ad
        return ad

    def cmf(self, period: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        cmf = ta.cmf(
            self.df['High'],
            self.df['Low'],
            self.df['Close'],
            self.df['Volume'],
            length=period
        )
        self.results[f'CMF_{period}'] = cmf
        return cmf

    def vwma(self, period: int = 20) -> pd.Series:
        """Volume Weighted Moving Average"""
        vwma = ta.vwma(self.df['Close'], self.df['Volume'], length=period)
        self.results[f'VWMA_{period}'] = vwma
        return vwma

    def volume_sma(self, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        vol_sma = self.df['Volume'].rolling(window=period).mean()
        self.results[f'VOLUME_SMA_{period}'] = vol_sma
        return vol_sma

    def pvt(self) -> pd.Series:
        """Price Volume Trend"""
        pvt = ta.pvt(self.df['Close'], self.df['Volume'])
        self.results['PVT'] = pvt
        return pvt

    # ==================== SUPPORT/RESISTANCE ====================

    def pivot_points(self) -> Dict[str, float]:
        """Calculate Pivot Points (Standard)"""
        if len(self.df) < 1:
            return {}

        high = self.df['High'].iloc[-1]
        low = self.df['Low'].iloc[-1]
        close = self.df['Close'].iloc[-1]

        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        result = {
            'PIVOT': pivot,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
        self.results.update(result)
        return result

    def swing_highs_lows(self, window: int = 5) -> Dict[str, pd.Series]:
        """Identify Swing Highs and Lows"""
        swing_high = pd.Series(False, index=self.df.index)
        swing_low = pd.Series(False, index=self.df.index)

        for i in range(window, len(self.df) - window):
            # Swing High: highest in window
            if self.df['High'].iloc[i] == self.df['High'].iloc[i-window:i+window+1].max():
                swing_high.iloc[i] = True

            # Swing Low: lowest in window
            if self.df['Low'].iloc[i] == self.df['Low'].iloc[i-window:i+window+1].min():
                swing_low.iloc[i] = True

        result = {
            'SWING_HIGH': swing_high,
            'SWING_LOW': swing_low
        }
        self.results.update(result)
        return result

    # ==================== PATTERN RECOGNITION ====================

    def candlestick_patterns(self) -> Dict[str, pd.Series]:
        """Detect common candlestick patterns"""
        patterns = {}

        # Doji
        body = abs(self.df['Close'] - self.df['Open'])
        range_size = self.df['High'] - self.df['Low']
        patterns['DOJI'] = body < (range_size * 0.1)

        # Hammer
        lower_shadow = self.df[['Open', 'Close']].min(axis=1) - self.df['Low']
        upper_shadow = self.df['High'] - self.df[['Open', 'Close']].max(axis=1)
        patterns['HAMMER'] = (lower_shadow > 2 * body) & (upper_shadow < body)

        # Shooting Star
        patterns['SHOOTING_STAR'] = (upper_shadow > 2 * body) & (lower_shadow < body)

        # Engulfing patterns
        bullish_engulfing = (
            (self.df['Close'].shift(1) < self.df['Open'].shift(1)) &  # Previous bearish
            (self.df['Close'] > self.df['Open']) &  # Current bullish
            (self.df['Open'] < self.df['Close'].shift(1)) &  # Opens below previous close
            (self.df['Close'] > self.df['Open'].shift(1))  # Closes above previous open
        )
        patterns['BULLISH_ENGULFING'] = bullish_engulfing

        bearish_engulfing = (
            (self.df['Close'].shift(1) > self.df['Open'].shift(1)) &  # Previous bullish
            (self.df['Close'] < self.df['Open']) &  # Current bearish
            (self.df['Open'] > self.df['Close'].shift(1)) &  # Opens above previous close
            (self.df['Close'] < self.df['Open'].shift(1))  # Closes below previous open
        )
        patterns['BEARISH_ENGULFING'] = bearish_engulfing

        self.results.update(patterns)
        return patterns

    # ==================== ALL INDICATORS ====================

    def calculate_all(self, custom_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculate all indicators and return complete dataframe

        Args:
            custom_config: Optional dict to customize periods/parameters
        """
        config = {
            'sma_periods': [20, 50, 200],
            'ema_periods': [9, 20, 50],
            'rsi_period': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb_period': 20,
            'atr_period': 14,
            'stoch': {'k': 14, 'd': 3},
            'adx_period': 14,
            'cci_period': 20,
            'mfi_period': 14,
            'volume_sma_period': 20,
        }

        if custom_config:
            config.update(custom_config)

        # Trend indicators
        for period in config['sma_periods']:
            self.sma(period)

        for period in config['ema_periods']:
            self.ema(period)

        self.vwap()
        self.supertrend()
        self.adx(config['adx_period'])

        # Momentum indicators
        self.rsi(config['rsi_period'])
        self.macd(**config['macd'])
        self.stochastic(**config['stoch'])
        self.cci(config['cci_period'])
        self.williams_r()
        self.roc()
        self.mfi(config['mfi_period'])

        # Volatility indicators
        self.bollinger_bands(config['bb_period'])
        self.atr(config['atr_period'])
        self.keltner_channels()
        self.historical_volatility()

        # Volume indicators
        self.obv()
        self.ad()
        self.cmf()
        self.volume_sma(config['volume_sma_period'])

        # Support/Resistance
        self.pivot_points()
        self.swing_highs_lows()

        # Patterns
        self.candlestick_patterns()

        # Combine all results
        result_df = self.df.copy()
        for key, value in self.results.items():
            if isinstance(value, pd.Series):
                result_df[key] = value
            elif isinstance(value, (int, float)):
                result_df[key] = value

        return result_df

    def get_latest_signals(self) -> Dict[str, Any]:
        """Get latest indicator values and generate signals"""
        if self.results == {}:
            self.calculate_all()

        latest = {}

        # Get latest values for key indicators
        for key, value in self.results.items():
            if isinstance(value, pd.Series) and len(value) > 0:
                latest[key] = value.iloc[-1]
            elif isinstance(value, (int, float)):
                latest[key] = value

        # Generate simple signals
        signals = {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volatility': 'NORMAL',
            'volume': 'NORMAL'
        }

        # Trend signal
        if 'SMA_50' in latest and 'SMA_200' in latest:
            if latest['SMA_50'] > latest['SMA_200']:
                signals['trend'] = 'BULLISH'
            else:
                signals['trend'] = 'BEARISH'

        # Momentum signal
        if 'RSI_14' in latest:
            if latest['RSI_14'] > 70:
                signals['momentum'] = 'OVERBOUGHT'
            elif latest['RSI_14'] < 30:
                signals['momentum'] = 'OVERSOLD'
            elif latest['RSI_14'] > 50:
                signals['momentum'] = 'BULLISH'
            else:
                signals['momentum'] = 'BEARISH'

        # Volatility signal
        if 'ATR_14' in latest and len(self.df) > 20:
            atr_sma = self.df['Close'].iloc[-20:].std()
            if latest['ATR_14'] > atr_sma * 1.5:
                signals['volatility'] = 'HIGH'
            elif latest['ATR_14'] < atr_sma * 0.5:
                signals['volatility'] = 'LOW'

        # Volume signal
        if 'VOLUME_SMA_20' in latest and len(self.df) > 0:
            current_vol = self.df['Volume'].iloc[-1]
            if current_vol > latest['VOLUME_SMA_20'] * 1.5:
                signals['volume'] = 'HIGH'
            elif current_vol < latest['VOLUME_SMA_20'] * 0.5:
                signals['volume'] = 'LOW'

        return {
            'indicators': latest,
            'signals': signals
        }


# Convenience function
def calculate_indicators(df: pd.DataFrame, indicators: Optional[list] = None) -> pd.DataFrame:
    """
    Quick function to calculate specific indicators or all indicators

    Args:
        df: OHLCV DataFrame
        indicators: List of indicator names to calculate. If None, calculates all.

    Returns:
        DataFrame with indicators added
    """
    ti = TechnicalIndicators(df)

    if indicators is None:
        return ti.calculate_all()

    # Calculate specific indicators
    result_df = df.copy()

    for indicator in indicators:
        indicator = indicator.upper()

        if 'RSI' in indicator:
            period = int(indicator.split('_')[1]) if '_' in indicator else 14
            result_df[f'RSI_{period}'] = ti.rsi(period)

        elif 'MACD' in indicator:
            macd_result = ti.macd()
            for key, val in macd_result.items():
                result_df[key] = val

        elif 'BB' in indicator or 'BOLLINGER' in indicator:
            bb_result = ti.bollinger_bands()
            for key, val in bb_result.items():
                result_df[key] = val

        elif 'ATR' in indicator:
            period = int(indicator.split('_')[1]) if '_' in indicator else 14
            result_df[f'ATR_{period}'] = ti.atr(period)

        elif 'VWAP' in indicator:
            result_df['VWAP'] = ti.vwap()

        elif 'OBV' in indicator:
            result_df['OBV'] = ti.obv()

        # Add more as needed

    return result_df
