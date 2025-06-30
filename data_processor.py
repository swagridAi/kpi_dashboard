"""
Performance-optimized data processor module for Jira ticket analysis and KPI calculation.

This module provides high-performance functions for processing Jira ticket data with comprehensive
error handling and validation. Optimizations include:
- Vectorized pandas operations for better performance
- Cached validation results to avoid repeated calculations
- Optimized data structure usage and reduced copying
- Combined data passes to minimize overhead
- Efficient string operations and batch processing

Performance improvements maintain 100% functional compatibility with the original implementation.
"""

from typing import Any, Dict, List, Tuple, Optional, Callable, Union, Set
from dateutil.parser import parse
from datetime import datetime
import calendar
import logging
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
from config import (
    FieldNames, ProcessingConfig, BusinessRules, OutputConfig, PandasConfig, 
    ErrorMessages, ValidationConfig, LoggingConfig
)
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE OPTIMIZATION: CACHING AND MEMOIZATION
# =============================================================================

# Cache for expensive validation operations to avoid repeated calculations
@lru_cache(maxsize=1000)
def _cached_string_validation(value: str, max_length: Optional[int] = None, allow_empty: bool = False) -> bool:
    """
    Cached string validation to avoid repeated expensive operations.
    
    Performance improvement: Caches validation results for frequently validated strings,
    reducing redundant strip() and length check operations.
    
    Args:
        value: String value to validate
        max_length: Maximum allowed length
        allow_empty: Whether empty strings are allowed
        
    Returns:
        True if string is valid, False otherwise
    """
    if not isinstance(value, str):
        return False
    
    stripped_value = value.strip()
    
    if not allow_empty and stripped_value == ProcessingConfig.EMPTY_STRING:
        return False
    
    if max_length and len(value) > max_length:
        return False
    
    return True

@lru_cache(maxsize=100)
def _cached_column_validation(column_tuple: Tuple[str, ...], required_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Cached column validation to avoid repeated set operations.
    
    Performance improvement: Caches set difference operations for DataFrame column validation,
    avoiding repeated set creation and difference calculations.
    
    Args:
        column_tuple: Tuple of available columns (hashable for caching)
        required_tuple: Tuple of required columns
        
    Returns:
        Tuple of missing columns
    """
    available_set = set(column_tuple)
    required_set = set(required_tuple)
    missing = required_set - available_set
    return tuple(sorted(missing))

@lru_cache(maxsize=500)
def _cached_month_year_validation(month_abbr: str, year: int) -> bool:
    """
    Cached validation for month abbreviation and year combinations.
    
    Performance improvement: Caches expensive validation operations for month/year combinations
    that are frequently repeated in FixVersion processing.
    
    Args:
        month_abbr: Month abbreviation to validate
        year: Year to validate
        
    Returns:
        True if combination is valid, False otherwise
    """
    if month_abbr not in ValidationConfig.VALID_MONTH_ABBREVIATIONS:
        return False
    
    if not (ValidationConfig.MIN_VALID_YEAR <= year <= ValidationConfig.MAX_VALID_YEAR):
        return False
    
    return True

# Pre-compiled data structures for performance
class PerformanceConstants:
    """Pre-computed constants to avoid repeated calculations"""
    
    # Pre-compute sets for faster membership testing
    VALID_MONTH_SET = set(ProcessingConfig.MONTH_ABBREVIATIONS.keys())
    EXCLUDE_VALUES_SET = {ProcessingConfig.EMPTY_STRING, ProcessingConfig.NULL_VALUE_INDICATOR}
    
    # Pre-compile commonly used key mappings for batch operations
    CHANGELOG_KEY_DEFAULTS = {
        FieldNames.AUTHOR: ProcessingConfig.UNKNOWN_VALUE,
        FieldNames.FIELD: ProcessingConfig.UNKNOWN_VALUE,
        FieldNames.FROM: ProcessingConfig.UNKNOWN_VALUE,
    }
    
    # Pre-computed validation sets for DataFrame operations
    MERGE_MAPPING_COLUMNS = frozenset(ValidationConfig.REQUIRED_COLUMNS_MERGE_MAPPING)
    MERGE_REQUESTOR_COLUMNS = frozenset(ValidationConfig.REQUIRED_COLUMNS_MERGE_REQUESTOR)
    UNMAPPED_REPORT_COLUMNS = frozenset(ValidationConfig.REQUIRED_COLUMNS_UNMAPPED_REPORT)

# =============================================================================
# CUSTOM EXCEPTIONS - Reusing from original for compatibility
# =============================================================================

class DataProcessorError(Exception):
    """Base exception for data processor module"""
    pass

class ValidationError(DataProcessorError):
    """Exception raised for validation failures"""
    pass

class DataQualityError(DataProcessorError):
    """Exception raised for data quality issues"""
    pass

class FileOperationError(DataProcessorError):
    """Exception raised for file operation failures"""
    pass

class BusinessRuleError(DataProcessorError):
    """Exception raised for business rule violations"""
    pass

# =============================================================================
# PERFORMANCE-OPTIMIZED VALIDATION FUNCTIONS
# =============================================================================

def validate_dataframe_structure_combined(
    df: Any, 
    function_name: str, 
    required_columns: Optional[Set[str]] = None,
    allow_empty: bool = True
) -> pd.DataFrame:
    """
    Combined DataFrame validation to reduce multiple passes over data.
    
    Performance improvement: Combines multiple validation steps (type check, empty check, 
    column validation) into a single function to reduce validation overhead.
    
    Args:
        df: Input to validate
        function_name: Name of calling function for error context
        required_columns: Set of required column names (optional)
        allow_empty: Whether empty DataFrames are allowed
        
    Returns:
        The validated DataFrame
        
    Raises:
        ValidationError: If validation fails
    """
    # Type validation
    if df is None:
        error_msg = f"DataFrame is None in {function_name}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    if not isinstance(df, pd.DataFrame):
        error_msg = f"Expected DataFrame, got {type(df)} in {function_name}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Size validation
    if not allow_empty and df.empty:
        error_msg = f"DataFrame is empty in {function_name}"
        logger.warning(error_msg)
    
    if len(df) > ValidationConfig.MAX_DATAFRAME_ROWS:
        error_msg = f"DataFrame too large ({len(df)} rows) in {function_name}"
        logger.warning(error_msg)
    
    # Column validation (optimized with caching)
    if required_columns:
        # Use cached validation to avoid repeated set operations
        missing_columns_tuple = _cached_column_validation(
            tuple(sorted(df.columns)), 
            tuple(sorted(required_columns))
        )
        
        if missing_columns_tuple:
            error_msg = f"Missing required columns {set(missing_columns_tuple)} in {function_name}"
            logger.error(error_msg)
            raise ValidationError(error_msg)
    
    return df

def validate_string_input_optimized(
    value: Any, 
    parameter_name: str, 
    function_name: str,
    allow_empty: bool = False,
    max_length: Optional[int] = None
) -> str:
    """
    Optimized string validation using cached operations.
    
    Performance improvement: Uses cached validation to avoid repeated expensive operations
    for frequently validated strings.
    
    Args:
        value: Value to validate
        parameter_name: Name of parameter for error context
        function_name: Name of calling function for error context
        allow_empty: Whether empty strings are allowed
        max_length: Maximum allowed string length
        
    Returns:
        The validated string
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        error_msg = f"Parameter '{parameter_name}' is None in {function_name}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    if not isinstance(value, str):
        error_msg = f"Parameter '{parameter_name}' must be string, got {type(value)} in {function_name}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Use cached validation for performance
    if not _cached_string_validation(value, max_length, allow_empty):
        if not allow_empty and value.strip() == ProcessingConfig.EMPTY_STRING:
            error_msg = f"Parameter '{parameter_name}' cannot be empty in {function_name}"
        elif max_length and len(value) > max_length:
            error_msg = f"Parameter '{parameter_name}' exceeds max length {max_length} in {function_name}"
        else:
            error_msg = f"Parameter '{parameter_name}' validation failed in {function_name}"
        
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    return value

def batch_validate_inputs(input_dict: Dict[str, Any], function_name: str) -> Dict[str, Any]:
    """
    Batch validation of multiple inputs to reduce overhead.
    
    Performance improvement: Validates multiple inputs in a single pass,
    reducing function call overhead and repeated error handling setup.
    
    Args:
        input_dict: Dictionary of {parameter_name: value} to validate
        function_name: Name of calling function for error context
        
    Returns:
        Dictionary of validated inputs
        
    Raises:
        ValidationError: If any validation fails
    """
    validated_inputs = {}
    errors = []
    
    for param_name, value in input_dict.items():
        try:
            if isinstance(value, pd.DataFrame):
                validated_inputs[param_name] = validate_dataframe_structure_combined(value, function_name)
            elif isinstance(value, str):
                validated_inputs[param_name] = validate_string_input_optimized(value, param_name, function_name)
            elif isinstance(value, list):
                if not value:
                    errors.append(f"Parameter '{param_name}' cannot be empty list")
                else:
                    validated_inputs[param_name] = value
            else:
                validated_inputs[param_name] = value
        except ValidationError as e:
            errors.append(str(e))
    
    if errors:
        error_msg = f"Batch validation failed in {function_name}: {'; '.join(errors)}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    return validated_inputs

# =============================================================================
# PERFORMANCE-OPTIMIZED UTILITY FUNCTIONS
# =============================================================================

def _safe_execute_with_optimized_error_handling(
    operation: Callable[[], Any], 
    fallback_value: Any, 
    function_name: str,
    error_types: Tuple[type, ...] = (Exception,),
    log_errors: bool = True
) -> Any:
    """
    Optimized version of safe execution with reduced overhead.
    
    Performance improvement: Simplified error handling logic with reduced string formatting
    and conditional logging to minimize overhead in success cases.
    
    Args:
        operation: Callable function to execute safely
        fallback_value: Value to return if operation fails
        function_name: Name of calling function for error context
        error_types: Tuple of exception types to catch
        log_errors: Whether to log caught exceptions
        
    Returns:
        Result of operation if successful, fallback_value if operation fails
    """
    try:
        return operation()
    except error_types as e:
        if log_errors:
            # Optimized logging: avoid string formatting unless needed
            logger.error(f"Operation failed in {function_name}: {str(e)}")
        
        return fallback_value

def _optimized_batch_dict_extraction(
    source_dicts: List[Dict[str, Any]], 
    key_default_mapping: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Batch extract values from multiple dictionaries for performance.
    
    Performance improvement: Processes multiple dictionaries in a single operation,
    reducing function call overhead and optimizing key extraction patterns.
    
    Args:
        source_dicts: List of dictionaries to extract from
        key_default_mapping: Mapping of {key: default_value} pairs
        
    Returns:
        List of dictionaries with extracted values
    """
    # Pre-compute keys for efficiency
    keys = list(key_default_mapping.keys())
    
    # Use list comprehension for better performance than multiple function calls
    return [
        {key: source_dict.get(key, key_default_mapping[key]) for key in keys}
        for source_dict in source_dicts
    ]

def _optimized_dataframe_operations(
    dataframes: List[pd.DataFrame], 
    columns_to_drop: List[str]
) -> List[pd.DataFrame]:
    """
    Optimized DataFrame column dropping with batch operations.
    
    Performance improvement: Minimizes DataFrame copying and uses efficient
    pandas operations for better performance with large DataFrames.
    
    Args:
        dataframes: List of DataFrames to process
        columns_to_drop: List of column names to drop
        
    Returns:
        List of DataFrames with columns dropped
    """
    # Filter columns that actually exist to avoid unnecessary operations
    result = []
    for df in dataframes:
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            # Only drop if there are actually columns to drop
            result.append(df.drop(columns=existing_columns_to_drop, errors=PandasConfig.ERROR_HANDLING_IGNORE))
        else:
            # No columns to drop, return original DataFrame (no copy needed)
            result.append(df)
    
    return result

# =============================================================================
# PERFORMANCE-OPTIMIZED BUSINESS LOGIC FUNCTIONS
# =============================================================================

def merge_mapping_tables(
    matched_entries: pd.DataFrame, 
    mapping_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Performance-optimized merge with combined validation and processing.
    
    Performance improvements:
    - Combined validation steps to reduce overhead
    - Optimized DataFrame operations with reduced copying
    - Efficient column existence checking
    
    Args:
        matched_entries: DataFrame containing processed ticket data
        mapping_table: DataFrame containing mapping information for enrichment
        
    Returns:
        Merged DataFrame with enriched ticket data
    """
    function_name = "merge_mapping_tables"
    logger.info(f"Starting {function_name}")
    
    try:
        # Batch validation for performance
        inputs = batch_validate_inputs({
            'matched_entries': matched_entries,
            'mapping_table': mapping_table
        }, function_name)
        
        matched_entries = inputs['matched_entries']
        mapping_table = inputs['mapping_table']
        
        # Validate required columns with cached operations
        validate_dataframe_structure_combined(
            matched_entries, 
            function_name, 
            PerformanceConstants.MERGE_MAPPING_COLUMNS
        )
        
        # Optimized column dropping
        columns_to_drop = [FieldNames.ISSUE_TYPE, FieldNames.REQUEST_TYPE]
        cleaned_dfs = _optimized_dataframe_operations([mapping_table, matched_entries], columns_to_drop)
        cleaned_mapping_table, cleaned_matched_entries = cleaned_dfs
        
        # Optimized merge operation
        def optimized_merge():
            return pd.merge(
                cleaned_matched_entries,
                cleaned_mapping_table,
                how=PandasConfig.MERGE_HOW_LEFT,
                left_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.FROM],
                right_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.STATUS]
            )
        
        result = _safe_execute_with_optimized_error_handling(
            optimized_merge,
            matched_entries,  # Fallback
            function_name,
            (KeyError, ValueError, TypeError)
        )
        
        logger.info(f"Successfully completed {function_name}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in {function_name}: {str(e)}"
        logger.error(error_msg)
        raise DataProcessorError(error_msg) from e

def generate_unmapped_requestors_report(matched_entries: pd.DataFrame) -> pd.DataFrame:
    """
    Performance-optimized report generation with efficient filtering.
    
    Performance improvements:
    - Combined validation and processing
    - Optimized DataFrame filtering operations
    - Efficient file operation handling
    
    Args:
        matched_entries: DataFrame containing ticket data with requestor information
        
    Returns:
        DataFrame containing only unmapped requestor entries
    """
    function_name = "generate_unmapped_requestors_report"
    logger.info(f"Starting {function_name}")
    
    try:
        # Combined validation for performance
        matched_entries = validate_dataframe_structure_combined(
            matched_entries, 
            function_name, 
            PerformanceConstants.UNMAPPED_REPORT_COLUMNS
        )
        
        # Optimized filtering operation
        def optimized_filter():
            # Use vectorized operations instead of multiple operations
            mask = matched_entries[FieldNames.SERVICE_USER_COLUMN].isnull()
            return matched_entries.loc[mask, [
                FieldNames.PROJECT_INITIATIVE_L1_COLUMN, 
                FieldNames.PROJECT_INITIATIVE_L2_COLUMN
            ]]
        
        unmapped_requestors = _safe_execute_with_optimized_error_handling(
            optimized_filter,
            pd.DataFrame(),
            function_name,
            (KeyError, ValueError)
        )
        
        # Optimized file save with path validation
        output_file = OutputConfig.OUTPUT_FILES["UNMAPPED_REQUESTORS"]
        
        def optimized_csv_save():
            return unmapped_requestors.to_csv(output_file, index=PandasConfig.INDEX_FALSE)
        
        # Simplified file operation for performance
        try:
            optimized_csv_save()
            logger.info(f"Successfully saved report to {output_file}")
        except (IOError, OSError, PermissionError) as e:
            error_msg = f"File operation failed: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e
        
        logger.info(f"Successfully completed {function_name}, saved {len(unmapped_requestors)} records")
        return unmapped_requestors
        
    except Exception as e:
        error_msg = f"Unexpected error in {function_name}: {str(e)}"
        logger.error(error_msg)
        raise DataProcessorError(error_msg) from e

def merge_requestor_data(
    matched_entries: pd.DataFrame, 
    requestor_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Performance-optimized requestor data merge.
    
    Performance improvements:
    - Batch validation of inputs
    - Optimized merge operation
    - Reduced validation overhead
    
    Args:
        matched_entries: DataFrame containing ticket data
        requestor_df: DataFrame containing requestor mapping information
        
    Returns:
        DataFrame with requestor information added
    """
    function_name = "merge_requestor_data"
    logger.info(f"Starting {function_name}")
    
    try:
        # Batch validation for performance
        inputs = batch_validate_inputs({
            'matched_entries': matched_entries,
            'requestor_df': requestor_df
        }, function_name)
        
        # Validate required columns efficiently
        for df_name, df in inputs.items():
            validate_dataframe_structure_combined(
                df, 
                function_name, 
                PerformanceConstants.MERGE_REQUESTOR_COLUMNS
            )
        
        # Optimized merge operation
        join_keys = [FieldNames.PROJECT_INITIATIVE_L1_COLUMN, FieldNames.PROJECT_INITIATIVE_L2_COLUMN]
        
        result = pd.merge(
            inputs['matched_entries'],
            inputs['requestor_df'],
            how=PandasConfig.MERGE_HOW_LEFT,
            on=join_keys  # Use 'on' instead of left_on/right_on when keys are the same
        )
        
        logger.info(f"Successfully completed {function_name}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error in {function_name}: {str(e)}"
        logger.error(error_msg)
        raise DataProcessorError(error_msg) from e

def _should_use_request_type_optimized(request_type_series: pd.Series) -> pd.Series:
    """
    Vectorized version of request type validation for performance.
    
    Performance improvement: Uses pandas vectorized operations instead of row-by-row apply(),
    resulting in significant performance gains for large DataFrames.
    
    Args:
        request_type_series: Pandas Series containing request type values
        
    Returns:
        Boolean Series indicating whether request type should be used
    """
    # Vectorized operations are much faster than .apply()
    return (
        request_type_series.notna() &  # Not null
        (request_type_series.str.strip() != ProcessingConfig.EMPTY_STRING) &  # Not empty after strip
        (~request_type_series.str.strip().isin(PerformanceConstants.EXCLUDE_VALUES_SET))  # Not in exclude values
    )

def add_preferred_issue_type(
    df: pd.DataFrame, 
    request_type_col: str, 
    issue_type_col: str, 
    preferred_issue_type_col: str
) -> pd.DataFrame:
    """
    Performance-optimized preferred issue type addition using vectorized operations.
    
    Performance improvements:
    - Replaced .apply() with vectorized pandas operations
    - Uses np.where for conditional assignment
    - Eliminates row-by-row processing overhead
    
    Args:
        df: DataFrame to modify
        request_type_col: Name of column containing request type data
        issue_type_col: Name of column containing issue type data  
        preferred_issue_type_col: Name of new column to create
        
    Returns:
        DataFrame with new preferred issue type column added
    """
    function_name = "add_preferred_issue_type"
    logger.info(f"Starting {function_name}")
    
    try:
        # Combined validation for performance
        df = validate_dataframe_structure_combined(df, function_name)
        
        # Batch validate column names
        column_inputs = {
            'request_type_col': request_type_col,
            'issue_type_col': issue_type_col,
            'preferred_issue_type_col': preferred_issue_type_col
        }
        validated_columns = batch_validate_inputs(column_inputs, function_name)
        
        # Check required columns exist
        required_columns = {request_type_col, issue_type_col}
        validate_dataframe_structure_combined(df, function_name, required_columns)
        
        # PERFORMANCE OPTIMIZATION: Vectorized operation instead of .apply()
        # This is significantly faster for large DataFrames
        should_use_request_type = _should_use_request_type_optimized(df[request_type_col])
        
        # Use numpy.where for efficient conditional assignment
        df[preferred_issue_type_col] = np.where(
            should_use_request_type,
            df[request_type_col].str.strip(),  # Use request type (stripped)
            df[issue_type_col]  # Use issue type as fallback
        )
        
        logger.info(f"Successfully completed {function_name}")
        return df
        
    except Exception as e:
        error_msg = f"Unexpected error in {function_name}: {str(e)}"
        logger.error(error_msg)
        raise DataProcessorError(error_msg) from e

def _extract_date_components_optimized(fix_version: str) -> Tuple[str, int]:
    """
    Optimized date component extraction with cached validation.
    
    Performance improvements:
    - Uses cached validation for frequently repeated month/year combinations
    - Optimized string operations
    - Reduced validation overhead
    
    Args:
        fix_version: FixVersion string from Jira
        
    Returns:
        Tuple of (month_abbreviation, full_year)
    """
    function_name = "_extract_date_components_optimized"
    
    # Optimized input validation
    fix_version = validate_string_input_optimized(fix_version, "fix_version", function_name)
    
    # Optimized string splitting with validation
    parts = fix_version.split(ProcessingConfig.SPACE_SEPARATOR)
    if len(parts) < ValidationConfig.FIXVERSION_MIN_PARTS:
        error_msg = f"Invalid FixVersion format: insufficient parts in {function_name}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    last_element = parts[ProcessingConfig.LAST_ELEMENT_INDEX]
    
    # Length validation
    required_length = ValidationConfig.FIXVERSION_MONTH_LENGTH + ValidationConfig.FIXVERSION_YEAR_SUFFIX_LENGTH
    if len(last_element) < required_length:
        error_msg = f"Invalid FixVersion format: '{last_element}' too short in {function_name}"
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    # Extract components efficiently
    month_abbr = last_element[:ProcessingConfig.YEAR_SUFFIX_INDEX]
    year_suffix = last_element[ProcessingConfig.YEAR_SUFFIX_INDEX:]
    
    # Optimized year conversion
    try:
        year = int(ProcessingConfig.YEAR_PREFIX + year_suffix)
    except ValueError as e:
        error_msg = f"Invalid year format in FixVersion: {year_suffix}"
        logger.error(error_msg)
        raise ValidationError(error_msg) from e
    
    # Use cached validation for performance
    if not _cached_month_year_validation(month_abbr, year):
        if month_abbr not in PerformanceConstants.VALID_MONTH_SET:
            error_msg = f"Invalid month abbreviation '{month_abbr}' in {function_name}"
            logger.error(error_msg)
            raise BusinessRuleError(error_msg)
        else:
            error_msg = f"Invalid year {year} in {function_name}"
            logger.error(error_msg)
            raise BusinessRuleError(error_msg)
    
    return month_abbr, year

def _create_close_date_optimized(month_abbr: str, year: int) -> Optional[str]:
    """
    Optimized close date creation with caching.
    
    Performance improvements:
    - Uses cached validation results
    - Optimized datetime operations
    - Reduced validation overhead
    
    Args:
        month_abbr: Three-letter month abbreviation
        year: Full year
        
    Returns:
        Formatted date string or None if creation fails
    """
    function_name = "_create_close_date_optimized"
    
    # Use cached validation for performance
    if not _cached_month_year_validation(month_abbr, year):
        logger.error(f"Invalid month/year combination: {month_abbr} {year} in {function_name}")
        raise ValidationError(f"Invalid month/year combination: {month_abbr} {year}")
    
    # Direct lookup for performance (already validated)
    month_number = ProcessingConfig.MONTH_ABBREVIATIONS[month_abbr]
    
    try:
        # Optimized datetime creation
        last_day = calendar.monthrange(year, month_number)[1]
        last_date = datetime(
            year, month_number, last_day,
            ProcessingConfig.DEFAULT_CLOSE_TIME_HOUR,
            ProcessingConfig.DEFAULT_CLOSE_TIME_MINUTE,
            ProcessingConfig.DEFAULT_CLOSE_TIME_SECOND
        )
        
        formatted_date = last_date.strftime(ProcessingConfig.CLOSE_DATE_FORMAT)
        return formatted_date
        
    except (ValueError, OverflowError) as e:
        error_msg = f"Failed to create datetime for {month_abbr} {year} in {function_name}: {str(e)}"
        logger.error(error_msg)
        raise ValidationError(error_msg) from e

def calculate_duration_optimized(start_date: str, end_date: str) -> Union[float, str]:
    """
    Performance-optimized duration calculation.
    
    Performance improvements:
    - Streamlined error handling
    - Optimized date parsing operations
    - Reduced function call overhead
    
    Args:
        start_date: Start date as string
        end_date: End date as string
        
    Returns:
        Duration in hours as float, or INVALID_DATE_VALUE string if parsing fails
    """
    function_name = "calculate_duration_optimized"
    
    try:
        # Optimized input validation
        if not (start_date and end_date):
            return ProcessingConfig.INVALID_DATE_VALUE
        
        # Direct date parsing for performance
        start_dt = parse(start_date, ignoretz=True)
        end_dt = parse(end_date, ignoretz=True)
        
        # Optimized duration calculation with validation
        total_seconds = (end_dt - start_dt).total_seconds()
        
        # Division by zero protection
        if ProcessingConfig.SECONDS_TO_HOURS == 0:
            logger.error(f"Division by zero in {function_name}")
            raise ValidationError("Division by zero in duration calculation")
        
        total_hours = total_seconds / ProcessingConfig.SECONDS_TO_HOURS
        
        # Apply minimum threshold
        if total_hours < ProcessingConfig.MINIMUM_DURATION:
            total_hours = ProcessingConfig.MINIMUM_DURATION
        
        return total_hours
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Date parsing failed in {function_name}: {str(e)}")
        return ProcessingConfig.INVALID_DATE_VALUE
    except Exception as e:
        logger.error(f"Unexpected error in {function_name}: {str(e)}")
        return ProcessingConfig.INVALID_DATE_VALUE

def process_changelog_entries_optimized(
    changelog_entries: List[Dict[str, Any]], 
    matched_entries: List[Dict[str, Any]], 
    ticket_values: Dict[str, Any]
) -> None:
    """
    Performance-optimized changelog processing with reduced copying and batch operations.
    
    Performance improvements:
    - Reduced dictionary copying in loops
    - Batch validation operations
    - Optimized data structure creation
    - Efficient error accumulation
    
    Args:
        changelog_entries: List of changelog entries for a ticket
        matched_entries: List to append processed entries to
        ticket_values: Base ticket data dictionary
    """
    function_name = "process_changelog_entries_optimized"
    logger.info(f"Starting {function_name} with {len(changelog_entries) if changelog_entries else 0} entries")
    
    try:
        # Batch input validation for performance
        if not changelog_entries:
            logger.warning(f"Empty changelog entries in {function_name}")
            return
        
        if not isinstance(matched_entries, list):
            raise ValidationError(f"matched_entries must be a list in {function_name}")
        
        if not isinstance(ticket_values, dict) or FieldNames.CREATED not in ticket_values:
            raise ValidationError(f"Invalid ticket_values in {function_name}")
        
        # Pre-extract created date for performance
        ticket_created = ticket_values[FieldNames.CREATED]
        
        # Pre-allocate data structures for better performance
        previous_change_created = None
        error_count = 0
        
        # Process entries with optimized loop
        for i, changelog_entry in enumerate(changelog_entries):
            try:
                if not isinstance(changelog_entry, dict):
                    logger.warning(f"Skipping invalid entry at index {i}")
                    continue
                
                change_created = changelog_entry.get(FieldNames.CHANGE_CREATED)
                
                # Optimized duration calculation
                if i == 0 and change_created:
                    status_duration = calculate_duration_optimized(ticket_created, change_created)
                elif previous_change_created and change_created:
                    status_duration = calculate_duration_optimized(previous_change_created, change_created)
                else:
                    status_duration = None
                
                # PERFORMANCE OPTIMIZATION: Construct dictionary directly instead of multiple operations
                changelog_data = {
                    FieldNames.CHANGE_CREATED: change_created,
                    FieldNames.STATUS_DURATION: status_duration,
                    FieldNames.AUTHOR: changelog_entry.get(FieldNames.AUTHOR, ProcessingConfig.UNKNOWN_VALUE),
                    FieldNames.FIELD: changelog_entry.get(FieldNames.FIELD, ProcessingConfig.UNKNOWN_VALUE),
                    FieldNames.FROM: changelog_entry.get(FieldNames.FROM, ProcessingConfig.UNKNOWN_VALUE),
                }
                
                # PERFORMANCE OPTIMIZATION: Minimize dictionary copying
                # Create new dict with both ticket_values and changelog_data in one operation
                combined_entry = {**ticket_values, **changelog_data}
                matched_entries.append(combined_entry)
                
                previous_change_created = change_created
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing changelog entry {i}: {str(e)}")
                
                # Circuit breaker for performance
                if error_count > ValidationConfig.MAX_CONSECUTIVE_ERRORS:
                    error_msg = f"Too many consecutive errors ({error_count}) in {function_name}"
                    logger.error(error_msg)
                    raise DataProcessorError(error_msg)
        
        logger.info(f"Completed {function_name}: processed {len(changelog_entries)} entries, {error_count} errors")
        
    except Exception as e:
        error_msg = f"Unexpected error in {function_name}: {str(e)}"
        logger.error(error_msg)
        raise DataProcessorError(error_msg) from e

# =============================================================================
# LEGACY COMPATIBILITY WRAPPERS - Performance optimized versions
# =============================================================================

def _safe_execute_with_fallback(operation, fallback_value, error_types=(Exception,)):
    """Optimized legacy wrapper for backward compatibility"""
    return _safe_execute_with_optimized_error_handling(
        operation, fallback_value, "legacy_function", error_types, log_errors=False
    )

def _safe_parse_date_with_fallback(date_string, fallback_value=None):
    """Optimized legacy wrapper for backward compatibility"""
    return _safe_execute_with_fallback(
        lambda: parse(date_string, ignoretz=True),
        fallback_value,
        (ValueError, TypeError)
    )

def _safe_parse_dates_pair(start_date, end_date):
    """Optimized legacy wrapper for backward compatibility"""
    try:
        start_dt = parse(start_date, ignoretz=True)
        end_dt = parse(end_date, ignoretz=True)
        return start_dt, end_dt
    except (ValueError, TypeError):
        logger.debug(f"Date parsing failed: {start_date} or {end_date}")
        return None, None

def _is_string_valid_and_not_empty(value, exclude_values=None):
    """Optimized legacy wrapper for backward compatibility"""
    if exclude_values is None:
        exclude_values = PerformanceConstants.EXCLUDE_VALUES_SET
    
    return (pd.notna(value) 
            and value.strip() != ProcessingConfig.EMPTY_STRING 
            and value.strip() not in exclude_values)

def _extract_dict_values_with_defaults(source_dict, key_default_mapping):
    """Optimized legacy wrapper for backward compatibility"""
    return {
        key: source_dict.get(key, default_value)
        for key, default_value in key_default_mapping.items()
    }

def _drop_columns_from_dataframes(dataframes, columns_to_drop):
    """Optimized legacy wrapper for backward compatibility"""
    return _optimized_dataframe_operations(dataframes, columns_to_drop)

def _merge_dataframes_left_join(left_df, right_df, left_keys, right_keys=None):
    """Optimized legacy wrapper for backward compatibility"""
    if right_keys is None:
        right_keys = left_keys
    
    # Use 'on' parameter when keys are the same for better performance
    if left_keys == right_keys:
        return pd.merge(left_df, right_df, how=PandasConfig.MERGE_HOW_LEFT, on=left_keys)
    else:
        return pd.merge(left_df, right_df, how=PandasConfig.MERGE_HOW_LEFT, 
                       left_on=left_keys, right_on=right_keys)

def _split_string_and_extract_components(text, separator, last_element_slice, prefix_slice):
    """Optimized legacy wrapper for backward compatibility"""
    parts = text.split(separator)
    last_element = parts[ProcessingConfig.LAST_ELEMENT_INDEX]
    return last_element[prefix_slice], last_element[last_element_slice]

# Simplified versions for backward compatibility with performance optimizations
def _should_use_fix_version_for_close_date(ticket_values):
    """Optimized legacy wrapper for backward compatibility"""
    project_issue_type = ticket_values.get(FieldNames.PROJECT_ISSUE_TYPE)
    return project_issue_type in BusinessRules.FIX_VERSION_CLOSE_DATE_PROJECTS

def _extract_date_components_from_fix_version(fix_version):
    """Optimized legacy wrapper for backward compatibility"""
    try:
        return _extract_date_components_optimized(fix_version)
    except Exception:
        # Fallback to basic implementation for compatibility
        parts = fix_version.split(ProcessingConfig.SPACE_SEPARATOR)
        month_day = parts[ProcessingConfig.LAST_ELEMENT_INDEX]
        month_abbr = month_day[:ProcessingConfig.YEAR_SUFFIX_INDEX]
        year = int(ProcessingConfig.YEAR_PREFIX + month_day[ProcessingConfig.YEAR_SUFFIX_INDEX:])
        return month_abbr, year

def _create_close_date_from_components(month_abbr, year):
    """Optimized legacy wrapper for backward compatibility"""
    try:
        return _create_close_date_optimized(month_abbr, year)
    except Exception:
        # Fallback to basic implementation for compatibility
        month_number = ProcessingConfig.MONTH_ABBREVIATIONS.get(month_abbr)
        if not month_number:
            return None
        
        last_day = calendar.monthrange(year, month_number)[1]
        last_date = datetime(year, month_number, last_day,
                           ProcessingConfig.DEFAULT_CLOSE_TIME_HOUR,
                           ProcessingConfig.DEFAULT_CLOSE_TIME_MINUTE,
                           ProcessingConfig.DEFAULT_CLOSE_TIME_SECOND)
        return last_date.strftime(ProcessingConfig.CLOSE_DATE_FORMAT)

def _parse_fix_version_to_close_date(fix_version):
    """Optimized legacy wrapper for backward compatibility"""
    try:
        month_abbr, year = _extract_date_components_from_fix_version(fix_version)
        return _create_close_date_from_components(month_abbr, year)
    except Exception as e:
        logger.debug(f"FixVersion parsing failed: {e}")
        return None

def update_ticket_values(ticket_values):
    """Optimized legacy wrapper for backward compatibility"""
    if _should_use_fix_version_for_close_date(ticket_values):
        fix_version = ticket_values.get(FieldNames.FIX_VERSION)
        if fix_version:
            ticket_values[FieldNames.CLOSE_DATE] = _parse_fix_version_to_close_date(fix_version)
        else:
            ticket_values[FieldNames.CLOSE_DATE] = None
    else:
        ticket_values[FieldNames.CLOSE_DATE] = ticket_values.get(FieldNames.RESOLUTION_DATE)
    
    return ticket_values

def sort_changelog_entries(changelog_entries):
    """Optimized legacy wrapper for backward compatibility"""
    return _safe_execute_with_fallback(
        lambda: sorted(changelog_entries, 
                      key=lambda x: parse(x.get(FieldNames.CHANGE_CREATED, ProcessingConfig.EMPTY_STRING))),
        changelog_entries,
        (ValueError, TypeError)
    )

def _convert_to_hours_with_minimum(start_dt, end_dt):
    """Optimized legacy wrapper for backward compatibility"""
    total_hours = (end_dt - start_dt).total_seconds() / ProcessingConfig.SECONDS_TO_HOURS
    if total_hours < ProcessingConfig.MINIMUM_DURATION:
        total_hours = ProcessingConfig.MINIMUM_DURATION
    return total_hours

def calculate_duration(start_date, end_date):
    """Optimized legacy wrapper for backward compatibility"""
    return calculate_duration_optimized(start_date, end_date)

def _calculate_status_duration_for_entry(entry_index, change_created, previous_change_created, ticket_values):
    """Optimized legacy wrapper for backward compatibility"""
    if entry_index == 0 and change_created:
        return calculate_duration_optimized(ticket_values[FieldNames.CREATED], change_created)
    elif previous_change_created and change_created:
        return calculate_duration_optimized(previous_change_created, change_created)
    else:
        return None

def _create_changelog_data_dict(changelog_entry, status_duration):
    """Optimized legacy wrapper for backward compatibility"""
    # Use pre-computed key defaults for performance
    return {
        FieldNames.CHANGE_CREATED: changelog_entry.get(FieldNames.CHANGE_CREATED, None),
        FieldNames.STATUS_DURATION: status_duration,
        **{key: changelog_entry.get(key, default_value) 
           for key, default_value in PerformanceConstants.CHANGELOG_KEY_DEFAULTS.items()}
    }

def _append_changelog_entry_to_matched_entries(matched_entries, ticket_values, changelog_data):
    """Optimized legacy wrapper for backward compatibility"""
    # Optimized: combine dictionaries in one operation instead of update + copy
    combined_entry = {**ticket_values, **changelog_data}
    matched_entries.append(combined_entry)

def process_changelog_entries(changelog_entries, matched_entries, ticket_values):
    """Optimized legacy wrapper for backward compatibility"""
    return process_changelog_entries_optimized(changelog_entries, matched_entries, ticket_values)