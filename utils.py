#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the stock analysis system.

This module provides common utility functions used across the codebase
to reduce duplication and improve maintainability.
"""
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, TypeVar
import numpy as np
import pandas as pd

from types_definitions import NumericValue, OptionalFloat
from config import Config
from exceptions import CacheError


# ============================== Type Variables ============================== #

T = TypeVar("T")


# ============================== Time Utilities ============================== #


def now_ist() -> datetime:
    """
    Get current datetime in Indian Standard Time (IST, UTC+5:30).

    Returns:
        Timezone-aware datetime in IST
    """
    ist_offset = timedelta(hours=Config.market.IST_OFFSET_HOURS)
    ist_tz = timezone(ist_offset)
    return datetime.now(timezone.utc).astimezone(ist_tz)


def format_timestamp(
    dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format a datetime object as a string.

    Args:
        dt: Datetime to format (defaults to current IST time)
        fmt: Format string

    Returns:
        Formatted datetime string
    """
    if dt is None:
        dt = now_ist()
    return dt.strftime(fmt)


def is_market_hours() -> bool:
    """
    Check if current time is within market trading hours.

    Returns:
        True if within market hours, False otherwise
    """
    now = now_ist()
    market_open = now.replace(
        hour=Config.market.MARKET_OPEN_HOUR,
        minute=Config.market.MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    market_close = now.replace(
        hour=Config.market.MARKET_CLOSE_HOUR,
        minute=Config.market.MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )
    return market_open <= now <= market_close


# ============================== Numeric Utilities ============================== #


def safe_float(x: Any, default: float = np.nan) -> float:
    """
    Safely convert a value to float, handling edge cases.

    Args:
        x: Value to convert (can be int, float, numpy type, pandas Series, etc.)
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        # Handle pandas Series
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            x = x.iloc[0]

        # Handle numpy types
        if isinstance(x, (np.integer, np.floating)):
            return float(x)

        # Handle None and NaN
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default

        return float(x)
    except (ValueError, TypeError, AttributeError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, handling edge cases.

    Args:
        x: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    try:
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            x = x.iloc[0]

        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default

        return int(x)
    except (ValueError, TypeError, AttributeError):
        return default


def safe_round(x: NumericValue, decimals: int = 2) -> Union[float, Any]:
    """
    Safely round a numeric value.

    Args:
        x: Value to round
        decimals: Number of decimal places

    Returns:
        Rounded value or original if rounding fails
    """
    try:
        return round(float(x), decimals)
    except (ValueError, TypeError):
        return x


def percentage_change(current: NumericValue, previous: NumericValue) -> OptionalFloat:
    """
    Calculate percentage change between two values.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Percentage change, or None if calculation fails
    """
    try:
        curr = float(current)
        prev = float(previous)
        if prev == 0:
            return None
        return ((curr / prev) - 1.0) * 100.0
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.

    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


# ============================== String Utilities ============================== #


def md5_hash(s: str) -> str:
    """
    Generate MD5 hash of a string.

    Args:
        s: Input string

    Returns:
        Hexadecimal MD5 hash
    """
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def clean_symbol(symbol: str) -> str:
    """
    Remove Yahoo Finance suffix from symbol.

    Args:
        symbol: Symbol with suffix (e.g., "RELIANCE.NS")

    Returns:
        Clean symbol (e.g., "RELIANCE")
    """
    return symbol.replace(Config.symbols.YF_SUFFIX, "")


def add_yf_suffix(symbol: str) -> str:
    """
    Add Yahoo Finance suffix to symbol if not present.

    Args:
        symbol: Symbol without suffix

    Returns:
        Symbol with suffix
    """
    if not symbol.endswith(Config.symbols.YF_SUFFIX):
        return symbol + Config.symbols.YF_SUFFIX
    return symbol


# ============================== Cache Utilities ============================== #


def load_cache(path: str) -> Dict[str, Any]:
    """
    Load cache data from JSON file.

    Args:
        path: Path to cache file

    Returns:
        Dictionary of cached data

    Raises:
        CacheError: If cache loading fails critically
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise CacheError(
            operation="read", cache_file=path, reason=f"Invalid JSON: {str(e)}"
        )
    except Exception as e:
        # Don't fail on cache errors, just return empty
        return {}


def save_cache(path: str, data: Dict[str, Any]) -> None:
    """
    Save cache data to JSON file.

    Args:
        path: Path to cache file
        data: Dictionary to cache

    Raises:
        CacheError: If cache saving fails
    """
    try:
        # Write to temporary file first
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Atomic replace
        os.replace(tmp_path, path)
    except Exception as e:
        raise CacheError(operation="write", cache_file=path, reason=str(e))


def clear_cache(path: Optional[str] = None) -> None:
    """
    Clear cache file(s).

    Args:
        path: Specific cache file path, or None to clear all caches
    """
    if path:
        if os.path.exists(path):
            os.remove(path)
    else:
        # Clear all cache files
        cache_dir = Config.paths.CACHE_DIR
        for filename in os.listdir(cache_dir):
            if filename.endswith(".json"):
                os.remove(os.path.join(cache_dir, filename))


# ============================== DataFrame Utilities ============================== #


def row_scalar(row: pd.Series, col: str, default: Any = np.nan) -> Any:
    """
    Extract scalar value from a DataFrame row, handling Series edge cases.

    Args:
        row: DataFrame row
        col: Column name
        default: Default value if column missing or value is Series

    Returns:
        Scalar value or default
    """
    if col not in row.index:
        return default

    value = row[col]

    # Handle accidental nested Series
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        return safe_float(value.iloc[0], default)

    return safe_float(value, default)


def ensure_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Check if DataFrame has all required columns.

    Args:
        df: DataFrame to check
        required_columns: List of required column names

    Returns:
        True if all columns present, False otherwise
    """
    return all(col in df.columns for col in required_columns)


def drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where all values are NaN.

    Args:
        df: DataFrame to clean

    Returns:
        Cleaned DataFrame
    """
    return df.dropna(how="all")


# ============================== List Utilities ============================== #


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deduplicate_list(lst: List[T], key: Optional[callable] = None) -> List[T]:
    """
    Remove duplicates from list while preserving order.

    Args:
        lst: List to deduplicate
        key: Optional function to extract comparison key

    Returns:
        Deduplicated list
    """
    if key is None:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        seen = set()
        result = []
        for item in lst:
            k = key(item)
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result


# ============================== Validation Utilities ============================== #


def is_valid_symbol(symbol: str) -> bool:
    """
    Check if a symbol is valid.

    Args:
        symbol: Stock symbol to validate

    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Remove suffix for validation
    clean = clean_symbol(symbol)

    # Basic checks: not empty, alphanumeric with some special chars
    return bool(
        clean and len(clean) <= 20 and clean.replace("-", "").replace("&", "").isalnum()
    )


def is_valid_price(price: Any) -> bool:
    """
    Check if a price value is valid.

    Args:
        price: Price value to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        p = float(price)
        return p > 0 and np.isfinite(p)
    except (ValueError, TypeError):
        return False


def is_valid_percentage(pct: Any) -> bool:
    """
    Check if a percentage value is valid.

    Args:
        pct: Percentage value to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        p = float(pct)
        return -100 <= p <= 1000 and np.isfinite(p)
    except (ValueError, TypeError):
        return False


# ============================== File Utilities ============================== #


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_file_age_hours(path: str) -> Optional[float]:
    """
    Get the age of a file in hours.

    Args:
        path: File path

    Returns:
        Age in hours, or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None

    mtime = os.path.getmtime(path)
    age_seconds = datetime.now().timestamp() - mtime
    return age_seconds / 3600


# ============================== Export ============================== #

__all__ = [
    # Time
    "now_ist",
    "format_timestamp",
    "is_market_hours",
    # Numeric
    "safe_float",
    "safe_int",
    "safe_round",
    "percentage_change",
    "clamp",
    # String
    "md5_hash",
    "truncate_string",
    "clean_symbol",
    "add_yf_suffix",
    # Cache
    "load_cache",
    "save_cache",
    "clear_cache",
    # DataFrame
    "row_scalar",
    "ensure_columns",
    "drop_all_nan_rows",
    # List
    "chunk_list",
    "deduplicate_list",
    # Validation
    "is_valid_symbol",
    "is_valid_price",
    "is_valid_percentage",
    # File
    "ensure_dir",
    "get_file_age_hours",
]
