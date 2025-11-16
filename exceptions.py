#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom exceptions for the stock analysis system.

This module defines a hierarchy of custom exceptions to provide
more precise error handling and better debugging information.
"""
from typing import Optional, Any, Dict


# ============================== Base Exceptions ============================== #


class StockAnalysisError(Exception):
    """Base exception for all stock analysis errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            details: Additional context or debugging information
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of the exception."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================== Data Errors ============================== #


class DataError(StockAnalysisError):
    """Base class for data-related errors."""

    pass


class DataFetchError(DataError):
    """Raised when data cannot be fetched from a source."""

    def __init__(
        self,
        source: str,
        symbol: Optional[str] = None,
        reason: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize the exception.

        Args:
            source: Data source name (e.g., "yfinance", "NewsAPI")
            symbol: Stock symbol if applicable
            reason: Reason for failure
            original_exception: Original exception if wrapping
        """
        message = f"Failed to fetch data from {source}"
        if symbol:
            message += f" for symbol {symbol}"
        if reason:
            message += f": {reason}"

        details = {"source": source}
        if symbol:
            details["symbol"] = symbol
        if original_exception:
            details["original_error"] = str(original_exception)

        super().__init__(message, details)
        self.source = source
        self.symbol = symbol
        self.original_exception = original_exception


class DataValidationError(DataError):
    """Raised when data fails validation checks."""

    def __init__(
        self, field: str, value: Any, expected: str, symbol: Optional[str] = None
    ):
        """
        Initialize the exception.

        Args:
            field: Name of the field that failed validation
            value: The invalid value
            expected: Description of expected value/format
            symbol: Stock symbol if applicable
        """
        message = (
            f"Validation failed for field '{field}': expected {expected}, got {value}"
        )
        if symbol:
            message = f"{symbol}: {message}"

        details = {"field": field, "value": value, "expected": expected}
        if symbol:
            details["symbol"] = symbol

        super().__init__(message, details)


class InsufficientDataError(DataError):
    """Raised when there is not enough data for analysis."""

    def __init__(
        self, symbol: str, required: int, available: int, data_type: str = "data points"
    ):
        """
        Initialize the exception.

        Args:
            symbol: Stock symbol
            required: Required number of data points
            available: Available number of data points
            data_type: Type of data (e.g., "bars", "news articles")
        """
        message = (
            f"Insufficient {data_type} for {symbol}: "
            f"required {required}, got {available}"
        )
        details = {
            "symbol": symbol,
            "required": required,
            "available": available,
            "data_type": data_type,
        }
        super().__init__(message, details)


# ============================== API Errors ============================== #


class APIError(StockAnalysisError):
    """Base class for API-related errors."""

    pass


class APIKeyMissingError(APIError):
    """Raised when a required API key is not configured."""

    def __init__(self, api_name: str, env_var: str):
        """
        Initialize the exception.

        Args:
            api_name: Name of the API service
            env_var: Environment variable name for the API key
        """
        message = (
            f"API key for {api_name} is missing. "
            f"Please set the {env_var} environment variable."
        )
        details = {"api_name": api_name, "env_var": env_var}
        super().__init__(message, details)


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, api_name: str, retry_after: Optional[int] = None):
        """
        Initialize the exception.

        Args:
            api_name: Name of the API service
            retry_after: Seconds to wait before retrying
        """
        message = f"Rate limit exceeded for {api_name}"
        if retry_after:
            message += f". Retry after {retry_after} seconds."

        details = {"api_name": api_name}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(message, details)


class APIResponseError(APIError):
    """Raised when API returns an unexpected response."""

    def __init__(
        self, api_name: str, status_code: int, response_text: Optional[str] = None
    ):
        """
        Initialize the exception.

        Args:
            api_name: Name of the API service
            status_code: HTTP status code
            response_text: Response body text
        """
        message = f"{api_name} returned status code {status_code}"
        details = {"api_name": api_name, "status_code": status_code}
        if response_text:
            details["response"] = response_text[:200]  # Limit length

        super().__init__(message, details)


# ============================== Analysis Errors ============================== #


class AnalysisError(StockAnalysisError):
    """Base class for analysis-related errors."""

    pass


class IndicatorCalculationError(AnalysisError):
    """Raised when technical indicator calculation fails."""

    def __init__(self, indicator_name: str, symbol: str, reason: Optional[str] = None):
        """
        Initialize the exception.

        Args:
            indicator_name: Name of the indicator
            symbol: Stock symbol
            reason: Reason for failure
        """
        message = f"Failed to calculate {indicator_name} for {symbol}"
        if reason:
            message += f": {reason}"

        details = {"indicator": indicator_name, "symbol": symbol}
        super().__init__(message, details)


class SignalGenerationError(AnalysisError):
    """Raised when signal generation fails."""

    def __init__(self, symbol: str, reason: str):
        """
        Initialize the exception.

        Args:
            symbol: Stock symbol
            reason: Reason for failure
        """
        message = f"Failed to generate signal for {symbol}: {reason}"
        details = {"symbol": symbol, "reason": reason}
        super().__init__(message, details)


# ============================== Sentiment Errors ============================== #


class SentimentError(StockAnalysisError):
    """Base class for sentiment analysis errors."""

    pass


class SentimentModelError(SentimentError):
    """Raised when sentiment model fails to load or execute."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize the exception.

        Args:
            model_name: Name of the sentiment model
            reason: Reason for failure
            original_exception: Original exception if wrapping
        """
        message = f"Sentiment model '{model_name}' error: {reason}"
        details = {"model": model_name}
        if original_exception:
            details["original_error"] = str(original_exception)

        super().__init__(message, details)
        self.original_exception = original_exception


class NewsNotFoundError(SentimentError):
    """Raised when no news articles are found."""

    def __init__(self, query: str, days: int):
        """
        Initialize the exception.

        Args:
            query: Search query used
            days: Number of days searched
        """
        message = f"No news found for query '{query}' in the last {days} days"
        details = {"query": query, "days": days}
        super().__init__(message, details)


# ============================== Configuration Errors ============================== #


class ConfigurationError(StockAnalysisError):
    """Base class for configuration errors."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, issue: str):
        """
        Initialize the exception.

        Args:
            config_key: Configuration key that is invalid
            issue: Description of the issue
        """
        message = f"Invalid configuration for '{config_key}': {issue}"
        details = {"config_key": config_key, "issue": issue}
        super().__init__(message, details)


class MissingDependencyError(ConfigurationError):
    """Raised when a required dependency is not installed."""

    def __init__(
        self, dependency: str, feature: str, install_command: Optional[str] = None
    ):
        """
        Initialize the exception.

        Args:
            dependency: Name of the missing dependency
            feature: Feature that requires the dependency
            install_command: Command to install the dependency
        """
        message = f"Missing dependency '{dependency}' required for {feature}."
        if install_command:
            message += f" Install with: {install_command}"

        details = {"dependency": dependency, "feature": feature}
        if install_command:
            details["install_command"] = install_command

        super().__init__(message, details)


# ============================== File I/O Errors ============================== #


class FileOperationError(StockAnalysisError):
    """Base class for file operation errors."""

    pass


class CacheError(FileOperationError):
    """Raised when cache operations fail."""

    def __init__(self, operation: str, cache_file: str, reason: Optional[str] = None):
        """
        Initialize the exception.

        Args:
            operation: Operation that failed (e.g., "read", "write")
            cache_file: Path to cache file
            reason: Reason for failure
        """
        message = f"Cache {operation} failed for {cache_file}"
        if reason:
            message += f": {reason}"

        details = {"operation": operation, "cache_file": cache_file}
        super().__init__(message, details)


class ReportGenerationError(FileOperationError):
    """Raised when report generation fails."""

    def __init__(self, report_type: str, output_path: str, reason: str):
        """
        Initialize the exception.

        Args:
            report_type: Type of report (e.g., "CSV", "Markdown")
            output_path: Path where report was to be saved
            reason: Reason for failure
        """
        message = f"Failed to generate {report_type} report at {output_path}: {reason}"
        details = {
            "report_type": report_type,
            "output_path": output_path,
            "reason": reason,
        }
        super().__init__(message, details)


# ============================== Timeout Errors ============================== #


class TimeoutError(StockAnalysisError):
    """Raised when an operation times out."""

    def __init__(
        self, operation: str, timeout_seconds: float, context: Optional[str] = None
    ):
        """
        Initialize the exception.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout value in seconds
            context: Additional context
        """
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"
        if context:
            message += f": {context}"

        details = {"operation": operation, "timeout": timeout_seconds}
        super().__init__(message, details)


# ============================== Export ============================== #

__all__ = [
    # Base
    "StockAnalysisError",
    # Data
    "DataError",
    "DataFetchError",
    "DataValidationError",
    "InsufficientDataError",
    # API
    "APIError",
    "APIKeyMissingError",
    "APIRateLimitError",
    "APIResponseError",
    # Analysis
    "AnalysisError",
    "IndicatorCalculationError",
    "SignalGenerationError",
    # Sentiment
    "SentimentError",
    "SentimentModelError",
    "NewsNotFoundError",
    # Configuration
    "ConfigurationError",
    "InvalidConfigurationError",
    "MissingDependencyError",
    # File I/O
    "FileOperationError",
    "CacheError",
    "ReportGenerationError",
    # Timeout
    "TimeoutError",
]
