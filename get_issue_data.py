# Standard library imports (alphabetical order)
import json
import logging
from pathlib import Path

# Third-party imports (alphabetical order)
import pandas as pd

# Type hint imports (organized by usage frequency)
from typing import (
    Any,
    Callable, 
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# Local application imports - Configuration (alphabetical order)
from config import (
    BusinessRules,
    DefaultValues,
    FieldNames,
    JiraFields,
    LoggingConfig,
    OutputConfig,
)

# Local application imports - Business rules (single import)
from business_rules import filter_matched_entries

# Local application imports - Data processing (alphabetical order)
from data_processor import (
    add_preferred_issue_type,
    generate_unmapped_requestors_report,
    merge_mapping_tables,
    merge_requestor_data,
    process_changelog_entries,
    sort_changelog_entries,
    update_ticket_values,
)

# Local application imports - JIRA parsing (specific imports only)
from jira_parser import (
    get_fields,
    process_components,
)

# Local application imports - KPI calculations (alphabetical order)  
from kpi_calculator import (
    get_issue_rate,
    get_time_status_per_month,
    standard_throughput_calculation,
)

# Local application imports - Output formatting (specific import only)
from output_formatter import get_requestor_data_clean

# Local application imports - Power BI formatting (alphabetical order)
from powerbi_formatter import (
    assemble_final_dataframes_dict,
    create_insights_matrix,
    create_kpi_result_views,
    create_slo_category_view,
    create_slo_service_view,
    enhance_recent_results,
    process_all_results_for_power,
)

# Type aliases for better code readability and domain clarity
JiraEntryDict = Dict[str, Any]  # Raw JIRA API response entry
ChangelogEntryDict = Dict[str, Any]  # Processed changelog entry
TicketValuesDict = Dict[str, Any]  # Complete ticket values dictionary
FieldMappingDict = Dict[str, Tuple[str, Any]]  # Field mapping configuration
ValidationFunction = Callable[[Any], bool]  # Function that validates data and returns bool
DataFrameTransformer = Callable[..., pd.DataFrame]  # Function that transforms DataFrames
OutputDictionary = Dict[str, pd.DataFrame]  # Final output dictionary for Excel export

# Generic type variable for DataFrame operations
DataFrameType = TypeVar('DataFrameType', bound=pd.DataFrame)

# Initialize logging configuration
logging.basicConfig(level=logging.INFO)
logging.info(LoggingConfig.PREFERRED_ISSUE_TYPE_LOG_MESSAGE)

# ========================================
# GENERIC UTILITY FUNCTIONS (DRY UTILITIES)
# ========================================

def safely_access_nested_field(
    source_obj: Optional[Dict[str, Any]], 
    field_path: List[str], 
    default_value: Any = None,
    intermediate_default: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Generic utility for safely accessing nested dictionary fields with comprehensive error handling.
    
    This function eliminates the need for repeated nested .get() calls throughout the codebase
    and provides consistent error handling for missing or malformed data structures.
    
    Args:
        source_obj: Source dictionary to access fields from. Can be None for safe handling.
        field_path: Ordered list of field names to traverse (e.g., ['fields', 'customfield_123', 'value'])
        default_value: Value to return if any field in the path is missing or if traversal fails
        intermediate_default: Default value to use for intermediate missing fields (typically {} for dicts)
        
    Returns:
        Any: The value found at the end of the field path, or default_value if traversal fails
        
    Raises:
        None: This function is designed to never raise exceptions - all errors result in default_value
        
    Examples:
        >>> entry = {'fields': {'customfield_123': {'value': 'test'}}}
        >>> safely_access_nested_field(entry, ['fields', 'customfield_123', 'value'], 'default')
        'test'
        
        >>> safely_access_nested_field(entry, ['fields', 'missing_field'], 'default')
        'default'
        
        >>> safely_access_nested_field(None, ['any', 'path'], 'safe_default')
        'safe_default'
        
    Business Context:
        JIRA API responses have inconsistent structure depending on field configuration,
        user permissions, and data completeness. This utility provides defensive programming
        against malformed or incomplete API responses.
    """
    # Early return for null source objects - common in JIRA API responses
    if not source_obj:
        return default_value
    
    current: Any = source_obj
    
    # Traverse each field in the path sequentially with comprehensive error checking
    for i, field in enumerate(field_path):
        # Defensive programming: ensure current level is a dictionary
        if not isinstance(current, dict):
            return default_value
            
        # Check if field exists at current level
        if field not in current:
            return default_value
        
        current = current[field]
        
        # Handle intermediate None values with appropriate defaults
        # This prevents AttributeError on subsequent field access
        if i < len(field_path) - 1 and current is None and intermediate_default is not None:
            return default_value
    
    # Return actual value or default if final value is None
    return current if current is not None else default_value


def safely_access_single_field(
    source_obj: Optional[Dict[str, Any]], 
    field_name: str, 
    default_value: Any = None
) -> Any:
    """
    Generic utility for safely accessing single dictionary field with type validation.
    
    This function provides a standardized way to access dictionary fields with proper
    type checking and default value handling. It's more robust than dict.get() because
    it handles cases where the source object itself is not a dictionary.
    
    Args:
        source_obj: Source dictionary to access field from. Can be None for safe handling.
        field_name: Name of the field to access
        default_value: Value to return if field not found or source_obj is invalid
        
    Returns:
        Any: Field value from source_obj[field_name] or default_value if not accessible
        
    Raises:
        None: This function never raises exceptions - invalid access returns default_value
        
    Examples:
        >>> data = {'name': 'John', 'age': 30}
        >>> safely_access_single_field(data, 'name', 'Unknown')
        'John'
        
        >>> safely_access_single_field(data, 'missing', 'Unknown')  
        'Unknown'
        
        >>> safely_access_single_field(None, 'any_field', 'Safe')
        'Safe'
        
        >>> safely_access_single_field("not_a_dict", 'field', 'Safe')
        'Safe'
    """
    # Type validation: ensure source is actually a dictionary
    if not isinstance(source_obj, dict):
        return default_value
        
    # Use dict.get() for safe field access with default
    return source_obj.get(field_name, default_value)


def build_field_mapping_dict(
    source_obj: Optional[Dict[str, Any]], 
    field_mappings: FieldMappingDict
) -> Dict[str, Any]:
    """
    Generic utility for building dictionaries using configurable field mappings with defaults.
    
    This function eliminates code duplication when building dictionaries by mapping
    source fields to target fields with appropriate default values. It provides
    a declarative way to specify field transformations.
    
    Args:
        source_obj: Source dictionary to extract fields from. Can be None for safe handling.
        field_mappings: Dictionary mapping target_field_name -> (source_field_name, default_value)
                       Format: {target: (source, default), ...}
        
    Returns:
        Dict[str, Any]: New dictionary with mapped fields and default values applied
        
    Raises:
        None: This function handles all error cases gracefully by using defaults
        
    Examples:
        >>> source = {'jira_field_1': 'value1', 'jira_field_2': 'value2'}
        >>> mappings = {
        ...     'clean_field_1': ('jira_field_1', 'default1'),
        ...     'clean_field_2': ('jira_field_2', 'default2'),
        ...     'missing_field': ('nonexistent', 'default3')
        ... }
        >>> build_field_mapping_dict(source, mappings)
        {'clean_field_1': 'value1', 'clean_field_2': 'value2', 'missing_field': 'default3'}
        
    Business Context:
        JIRA API responses often need to be transformed from JIRA-specific field names
        to business-friendly field names for internal processing. This utility provides
        a consistent way to perform these transformations with proper error handling.
    """
    # Handle null source objects gracefully
    if not source_obj:
        return {}
    
    result: Dict[str, Any] = {}
    
    # Process each field mapping with individual error handling
    for target_field, (source_field, default_value) in field_mappings.items():
        # Use safe field access to get value or default
        result[target_field] = safely_access_single_field(source_obj, source_field, default_value)
    
    return result


def apply_transformation_to_multiple_dataframes(
    dataframes: List[pd.DataFrame],
    transformation_func: DataFrameTransformer,
    *args: Any,
    **kwargs: Any
) -> List[pd.DataFrame]:
    """
    Generic utility for applying the same transformation function to multiple DataFrames efficiently.
    
    This function eliminates code duplication when the same transformation needs to be
    applied to multiple DataFrames with identical parameters. It's particularly useful
    for batch operations like adding columns, applying filters, or data cleansing.
    
    Args:
        dataframes: List of DataFrames to transform. All must be valid pandas DataFrames.
        transformation_func: Function that takes a DataFrame and returns a transformed DataFrame.
                           Must be callable and return a DataFrame.
        *args: Additional positional arguments to pass to transformation_func
        **kwargs: Additional keyword arguments to pass to transformation_func
        
    Returns:
        List[pd.DataFrame]: List of transformed DataFrames in the same order as input
        
    Raises:
        TypeError: If transformation_func is not callable
        ValueError: If any DataFrame transformation fails
        
    Examples:
        >>> df1 = pd.DataFrame({'a': [1, 2]})
        >>> df2 = pd.DataFrame({'a': [3, 4]}) 
        >>> def add_column(df, col_name, value):
        ...     df[col_name] = value
        ...     return df
        >>> result = apply_transformation_to_multiple_dataframes(
        ...     [df1, df2], add_column, 'new_col', 'test_value'
        ... )
        >>> # Both DataFrames now have 'new_col' with 'test_value'
        
    Performance Notes:
        - Transformations are applied sequentially, not in parallel
        - Each DataFrame is processed independently
        - Memory usage scales with number and size of DataFrames
    """
    # Validate that transformation function is callable
    if not callable(transformation_func):
        raise TypeError(f"transformation_func must be callable, got {type(transformation_func)}")
    
    # Apply transformation to each DataFrame with consistent parameters
    transformed_dataframes: List[pd.DataFrame] = []
    for df in dataframes:
        try:
            transformed_df: pd.DataFrame = transformation_func(df, *args, **kwargs)
            transformed_dataframes.append(transformed_df)
        except Exception as e:
            raise ValueError(f"Failed to transform DataFrame: {str(e)}") from e
    
    return transformed_dataframes


def validate_data_structure(
    data: Any, 
    validators: List[ValidationFunction]
) -> bool:
    """
    Generic utility for applying multiple validation checks to data structures.
    
    This function provides a composable way to validate data by applying a series
    of validation functions. All validators must pass for the data to be considered valid.
    It eliminates duplication of validation logic across the codebase.
    
    Args:
        data: Data of any type to be validated
        validators: List of functions that take data and return True if valid, False otherwise.
                   Each function should handle its own error cases gracefully.
        
    Returns:
        bool: True if ALL validators return True, False if ANY validator returns False
        
    Raises:
        None: This function catches all exceptions from validators and treats them as validation failures
        
    Examples:
        >>> data = {'key': 'value'}
        >>> validators = [
        ...     lambda x: x is not None,
        ...     lambda x: isinstance(x, dict),
        ...     lambda x: 'key' in x
        ... ]
        >>> validate_data_structure(data, validators)
        True
        
        >>> validate_data_structure(None, validators)
        False
        
        >>> validate_data_structure("not_a_dict", validators)  
        False
        
    Business Context:
        JIRA API responses require multiple layers of validation due to:
        - Optional fields that may be missing
        - Different data types based on configuration  
        - Malformed responses due to network issues
        This utility provides consistent validation patterns.
    """
    try:
        # Apply all validators using short-circuit evaluation for efficiency
        # If any validator returns False, immediately return False
        return all(validator(data) for validator in validators)
    except Exception:
        # If any validator raises an exception, treat as validation failure
        # This provides defensive programming against malformed validators
        return False


def create_dataframe_column_from_operation(
    df: pd.DataFrame,
    new_column: str,
    source_columns: List[str],
    operation: Literal['concat', 'slice', 'astype_slice'],
    operation_params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generic utility for creating DataFrame columns using common transformation operations.
    
    This function standardizes common DataFrame column creation patterns to eliminate
    code duplication. It supports the most frequently used column operations in the
    JIRA data processing pipeline.
    
    Args:
        df: Source DataFrame to add column to. Must be a valid pandas DataFrame.
        new_column: Name of the new column to create
        source_columns: List of existing column names to use as sources for the operation
        operation: Type of operation to perform. Supported operations:
                  - 'concat': Concatenate multiple columns with separator
                  - 'slice': Slice string column (requires start/end params)  
                  - 'astype_slice': Convert to string then slice (requires start/end params)
        operation_params: Dictionary of parameters for the operation:
                         - For 'concat': {'separator': str} (default: ' - ')
                         - For 'slice'/'astype_slice': {'start': int, 'end': int} (default: start=0, end=None)
        
    Returns:
        pd.DataFrame: DataFrame with new column added. Original DataFrame is modified in place.
        
    Raises:
        KeyError: If any source_columns don't exist in the DataFrame
        ValueError: If operation is not supported or required parameters are missing
        
    Examples:
        >>> df = pd.DataFrame({'service': ['A', 'B'], 'kpi': ['Lead', 'Cycle']})
        >>> # Concatenate columns
        >>> result = create_dataframe_column_from_operation(
        ...     df, 'service_kpi', ['service', 'kpi'], 'concat', {'separator': '_'}
        ... )
        >>> # result['service_kpi'] contains ['A_Lead', 'B_Cycle']
        
        >>> df = pd.DataFrame({'date': ['2024-01-15', '2024-02-20']})
        >>> # Extract year-month from date
        >>> result = create_dataframe_column_from_operation(
        ...     df, 'year_month', ['date'], 'astype_slice', {'start': 0, 'end': 7}
        ... )
        >>> # result['year_month'] contains ['2024-01', '2024-02']
        
    Business Context:
        Common DataFrame operations in JIRA processing include:
        - Creating composite keys (service + KPI type)
        - Extracting date components for monthly grouping
        - Formatting display strings for reports
    """
    # Initialize operation parameters with defaults
    if operation_params is None:
        operation_params = {}
    
    # Validate that all source columns exist in DataFrame
    missing_columns: List[str] = [col for col in source_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Source columns not found in DataFrame: {missing_columns}")
    
    # Perform operation based on type
    if operation == 'concat':
        # Concatenate multiple columns with configurable separator
        separator: str = operation_params.get('separator', ' - ')
        
        if len(source_columns) < 2:
            raise ValueError("concat operation requires at least 2 source columns")
        
        # Start with first column and concatenate others
        result_series: pd.Series = df[source_columns[0]].astype(str)
        for col in source_columns[1:]:
            result_series = result_series + separator + df[col].astype(str)
        
        df[new_column] = result_series
    
    elif operation in ['slice', 'astype_slice']:
        # String slicing operations with optional type conversion
        start_idx: int = operation_params.get('start', 0)
        end_idx: Optional[int] = operation_params.get('end', None)
        
        if len(source_columns) != 1:
            raise ValueError(f"{operation} operation requires exactly 1 source column")
        
        source_column: str = source_columns[0]
        
        if operation == 'astype_slice':
            # Convert to string first, then slice
            df[new_column] = df[source_column].astype(str).str[start_idx:end_idx]
        else:
            # Direct string slicing (assumes column is already string type)
            df[new_column] = df[source_column].str[start_idx:end_idx]
    
    else:
        # Invalid operation type
        raise ValueError(f"Unsupported operation: {operation}. Supported: 'concat', 'slice', 'astype_slice'")
    
    return df


def safely_return_with_fallback(
    validation_func: Callable[[], bool],
    success_func: Callable[[], Any],
    fallback_value: Any
) -> Any:
    """
    Generic utility for safe execution with validation and fallback handling.
    
    This function implements a common pattern: validate preconditions, execute main logic
    if validation passes, or return a fallback value if validation fails. It provides
    consistent error handling across the codebase.
    
    Args:
        validation_func: Function that returns True if main logic should execute, False otherwise.
                        Should not take parameters and should handle its own exceptions.
        success_func: Function to execute if validation passes. Should not take parameters.
                     Return value will be returned from this function.
        fallback_value: Value to return if validation fails or success_func raises an exception
        
    Returns:
        Any: Result of success_func() if validation passes, otherwise fallback_value
        
    Raises:
        None: This function catches all exceptions and returns fallback_value for safety
        
    Examples:
        >>> def validate_positive(): return 5 > 0
        >>> def calculate(): return 5 * 10
        >>> safely_return_with_fallback(validate_positive, calculate, -1)
        50
        
        >>> def validate_negative(): return 5 < 0  
        >>> safely_return_with_fallback(validate_negative, calculate, -1)
        -1
        
        >>> def validate_error(): raise ValueError("test")
        >>> safely_return_with_fallback(validate_error, calculate, -1)
        -1
        
    Business Context:
        JIRA data processing involves many operations that can fail due to:
        - Missing or malformed data
        - Network issues
        - Unexpected API response formats
        This utility provides consistent error handling patterns.
    """
    try:
        # Execute validation function with exception handling
        if not validation_func():
            return fallback_value
        
        # Execute main logic if validation passes
        return success_func()
        
    except Exception:
        # Any exception during validation or execution results in fallback
        # This provides maximum safety for data processing pipelines
        return fallback_value


# ========================================
# REFACTORED VALIDATION FUNCTIONS (USING UTILITIES)
# ========================================

def is_valid_changelog_entry(changelog: Optional[Dict[str, Any]]) -> bool:
    """
    Validate that changelog entry contains required fields for processing.
    
    Checks that the changelog is a valid dictionary containing the 'histories' field
    which is required for extracting status change information from JIRA entries.
    
    Args:
        changelog: JIRA changelog dictionary from API response, can be None
        
    Returns:
        bool: True if changelog is valid and contains histories, False otherwise
        
    Examples:
        >>> valid_changelog = {'histories': [{'created': '2024-01-01', 'items': []}]}
        >>> is_valid_changelog_entry(valid_changelog)
        True
        
        >>> is_valid_changelog_entry(None)
        False
        
        >>> is_valid_changelog_entry({'other_field': 'value'})
        False
        
    Business Context:
        JIRA issues may not have changelog data if:
        - Issue was never updated after creation
        - User lacks permissions to view change history  
        - API response is incomplete due to network issues
    """
    return validate_data_structure(
        changelog,
        [
            lambda x: x is not None,
            lambda x: isinstance(x, dict),
            lambda x: JiraFields.HISTORIES in x
        ]
    )


def is_status_change_item(item: Optional[Dict[str, Any]]) -> bool:
    """
    Validate that changelog item represents a status field change.
    
    JIRA changelog items can represent changes to any field (status, assignee, labels, etc.).
    This function filters to only status changes which are needed for KPI calculations.
    
    Args:
        item: Individual changelog item dictionary from JIRA histories, can be None
        
    Returns:
        bool: True if item represents a status field change, False otherwise
        
    Examples:
        >>> status_item = {'field': 'status', 'fromString': 'Open', 'toString': 'In Progress'}
        >>> is_status_change_item(status_item)
        True
        
        >>> assignee_item = {'field': 'assignee', 'fromString': 'John', 'toString': 'Jane'}
        >>> is_status_change_item(assignee_item)
        False
        
    Business Context:
        KPI calculations (lead time, cycle time) depend only on status transitions.
        Other field changes (assignee, labels, comments) are not relevant for timing metrics.
    """
    field_value: str = safely_access_single_field(item, JiraFields.FIELD_NAME, DefaultValues.UNKNOWN)
    return field_value == JiraFields.STATUS_FIELD


def has_valid_custom_field(custom_field: Any) -> bool:
    """
    Validate that custom field value is a dictionary suitable for processing.
    
    JIRA custom fields can have various data types (string, number, dict, array)
    depending on field configuration. This function validates that the field
    is a dictionary which is required for accessing subfields.
    
    Args:
        custom_field: Custom field value from JIRA API, any type
        
    Returns:
        bool: True if custom_field is a non-null dictionary, False otherwise
        
    Examples:
        >>> valid_field = {'requestType': {'name': 'Bug Report'}}
        >>> has_valid_custom_field(valid_field)
        True
        
        >>> has_valid_custom_field("string_value")
        False
        
        >>> has_valid_custom_field(None)
        False
        
    Business Context:
        Custom fields in JIRA can be configured as:
        - Simple text/number fields (not dictionaries)
        - Complex objects with subfields (dictionaries)
        - Arrays of values
        Processing logic needs to handle all these cases safely.
    """
    return validate_data_structure(
        custom_field,
        [
            lambda x: x is not None,
            lambda x: isinstance(x, dict)
        ]
    )


def is_entry_processable(entry: Optional[JiraEntryDict]) -> bool:
    """
    Validate that JIRA entry is suitable for processing.
    
    Performs basic validation to ensure the entry is a non-null dictionary
    before attempting to extract fields from it. This prevents downstream
    processing errors from malformed API responses.
    
    Args:
        entry: JIRA entry dictionary from API response, can be None
        
    Returns:
        bool: True if entry can be safely processed, False otherwise
        
    Examples:
        >>> valid_entry = {'key': 'PROJ-123', 'fields': {...}}
        >>> is_entry_processable(valid_entry)
        True
        
        >>> is_entry_processable(None)
        False
        
        >>> is_entry_processable("not_a_dict")
        False
        
    Business Context:
        JIRA API can return:
        - Valid entries with full field data
        - Null entries due to permissions or deletion
        - Malformed entries due to network issues
        Early validation prevents processing pipeline failures.
    """
    return validate_data_structure(
        entry,
        [
            lambda x: x is not None,
            lambda x: isinstance(x, dict)
        ]
    )


# ========================================
# REFACTORED DATA EXTRACTION FUNCTIONS (USING UTILITIES)
# ========================================

def extract_author_display_name(history: Dict[str, Any]) -> str:
    """
    Extract human-readable author name from changelog history entry.
    
    JIRA changelog history entries contain author information in nested structure.
    This function safely extracts the display name for audit trail purposes.
    
    Args:
        history: Single changelog history entry from JIRA API
        
    Returns:
        str: Author display name or 'Unknown' if not found
        
    Examples:
        >>> history = {
        ...     'author': {'displayName': 'John Doe', 'emailAddress': 'john@example.com'},
        ...     'created': '2024-01-01T10:00:00.000+0000'
        ... }
        >>> extract_author_display_name(history)
        'John Doe'
        
        >>> extract_author_display_name({'author': {}})
        'Unknown'
        
    Business Context:
        Author information is used for:
        - Audit trails in reporting
        - Identifying who made status changes  
        - Performance analysis by team member
        Display name is preferred over internal user IDs for readability.
    """
    return safely_access_nested_field(
        history,
        [JiraFields.AUTHOR_FIELD, JiraFields.DISPLAY_NAME],
        DefaultValues.UNKNOWN,
        {}
    )


def extract_change_timestamp(history: Dict[str, Any]) -> str:
    """
    Extract timestamp when the change occurred from changelog history entry.
    
    Timestamps are critical for calculating KPI metrics like lead time and cycle time.
    This function safely extracts the ISO timestamp from changelog entries.
    
    Args:
        history: Single changelog history entry from JIRA API
        
    Returns:
        str: ISO timestamp string or 'Unknown' if not found
        
    Examples:
        >>> history = {
        ...     'created': '2024-01-01T10:00:00.000+0000',
        ...     'author': {...}
        ... }
        >>> extract_change_timestamp(history)
        '2024-01-01T10:00:00.000+0000'
        
        >>> extract_change_timestamp({})
        'Unknown'
        
    Business Context:
        Timestamps are used for:
        - Calculating duration between status changes
        - Ordering changelog entries chronologically
        - Creating time-series reports and trends
        Accuracy is critical for KPI calculations.
    """
    return safely_access_single_field(history, JiraFields.CREATED, DefaultValues.UNKNOWN)


def build_changelog_entry_dict(
    history: Dict[str, Any], 
    item: Dict[str, Any]
) -> ChangelogEntryDict:
    """
    Build complete changelog entry dictionary from history and item data.
    
    Combines author/timestamp information from history with field change details
    from item to create a complete changelog entry suitable for processing.
    
    Args:
        history: Changelog history containing author and timestamp information
        item: Changelog item containing specific field change details
        
    Returns:
        ChangelogEntryDict: Complete changelog entry with all required fields
        
    Examples:
        >>> history = {
        ...     'author': {'displayName': 'John Doe'},
        ...     'created': '2024-01-01T10:00:00.000+0000'
        ... }
        >>> item = {
        ...     'field': 'status',
        ...     'fromString': 'Open', 
        ...     'toString': 'In Progress'
        ... }
        >>> result = build_changelog_entry_dict(history, item)
        >>> result['Author']
        'John Doe'
        >>> result['From']
        'Open'
        
    Business Context:
        Changelog entries are the foundation for KPI calculations.
        They provide the complete audit trail of status changes needed
        to calculate lead time, cycle time, and other timing metrics.
    """
    # Define field mappings for changelog item fields
    # This declarative approach makes field mapping clear and maintainable
    item_field_mappings: FieldMappingDict = {
        FieldNames.FIELD: (JiraFields.FIELD_NAME, DefaultValues.UNKNOWN),
        FieldNames.FROM: (JiraFields.FROM_STRING, DefaultValues.UNKNOWN),
        FieldNames.TO: (JiraFields.TO_STRING, DefaultValues.UNKNOWN)
    }
    
    # Build base dictionary from item using utility function
    changelog_dict: Dict[str, Any] = build_field_mapping_dict(item, item_field_mappings)
    
    # Add history-specific fields using extraction functions
    changelog_dict[FieldNames.AUTHOR] = extract_author_display_name(history)
    changelog_dict[FieldNames.CHANGE_CREATED] = extract_change_timestamp(history)
    
    return changelog_dict


def extract_changelog_entries_from_history(
    changelog: Optional[Dict[str, Any]]
) -> List[ChangelogEntryDict]:
    """
    Extract all relevant status change entries from JIRA changelog with comprehensive error handling.
    
    Processes the complete changelog structure to find all status changes,
    which are the only changelog items relevant for KPI calculations.
    Uses safe execution patterns to handle malformed or incomplete data.
    
    Args:
        changelog: Complete JIRA changelog dictionary from API response, can be None
        
    Returns:
        List[ChangelogEntryDict]: List of processed status change entries, empty if no valid changes
        
    Examples:
        >>> changelog = {
        ...     'histories': [
        ...         {
        ...             'author': {'displayName': 'John'},
        ...             'created': '2024-01-01T10:00:00.000+0000',
        ...             'items': [
        ...                 {'field': 'status', 'fromString': 'Open', 'toString': 'In Progress'},
        ...                 {'field': 'assignee', 'fromString': 'Jane', 'toString': 'John'}
        ...             ]
        ...         }
        ...     ]
        ... }
        >>> entries = extract_changelog_entries_from_history(changelog)
        >>> len(entries)  # Only status change extracted
        1
        >>> entries[0]['From']
        'Open'
        
    Business Context:
        Changelog extraction is a critical step because:
        - KPI calculations depend entirely on status transition timing
        - Malformed changelogs would break the entire processing pipeline
        - JIRA APIs can return incomplete data due to permissions or errors
        Robust error handling ensures processing continues even with partial data.
    """
    return safely_return_with_fallback(
        lambda: is_valid_changelog_entry(changelog),
        lambda: [
            build_changelog_entry_dict(history, item)
            for history in changelog[JiraFields.HISTORIES]  # type: ignore[index]  # validated by is_valid_changelog_entry
            for item in history.get(JiraFields.ITEMS, [])
            if is_status_change_item(item)
        ],
        []  # Return empty list if changelog is invalid or processing fails
    )


def extract_custom_field_safely(
    entry: Optional[JiraEntryDict], 
    custom_field_id: str
) -> Dict[str, Any]:
    """
    Safely extract custom field from JIRA entry with comprehensive validation.
    
    Custom fields in JIRA require special handling because they:
    - May not exist if not configured for the issue type
    - May have different data types based on field configuration
    - May be null if not set by users
    This function provides defensive extraction with proper error handling.
    
    Args:
        entry: JIRA entry dictionary from API response, can be None
        custom_field_id: JIRA custom field identifier (e.g., 'customfield_23641')
        
    Returns:
        Dict[str, Any]: Custom field dictionary or empty dict if not valid/accessible
        
    Examples:
        >>> entry = {
        ...     'fields': {
        ...         'customfield_123': {'requestType': {'name': 'Bug Report'}}
        ...     }
        ... }
        >>> result = extract_custom_field_safely(entry, 'customfield_123')
        >>> result['requestType']['name']
        'Bug Report'
        
        >>> extract_custom_field_safely(None, 'any_field')
        {}
        
        >>> entry_with_string_field = {'fields': {'customfield_123': 'string_value'}}
        >>> extract_custom_field_safely(entry_with_string_field, 'customfield_123')
        {}  # String fields return empty dict as they're not processable as dicts
        
    Business Context:
        Custom fields store business-critical information like:
        - Service desk request types
        - Project initiative hierarchies  
        - Business process metadata
        Safe extraction prevents pipeline failures from field configuration changes.
    """
    return safely_return_with_fallback(
        lambda: is_entry_processable(entry),
        lambda: _extract_validated_custom_field(entry, custom_field_id),  # type: ignore[arg-type]  # validated by is_entry_processable
        {}
    )


def _extract_validated_custom_field(entry: JiraEntryDict, custom_field_id: str) -> Dict[str, Any]:
    """
    Internal helper to extract custom field after entry validation.
    
    Args:
        entry: Validated JIRA entry dictionary
        custom_field_id: JIRA custom field identifier
        
    Returns:
        Dict[str, Any]: Custom field dictionary or empty dict if not valid
    """
    # Extract the custom field using safe nested access
    custom_field: Any = safely_access_nested_field(
        entry,
        [JiraFields.FIELDS, custom_field_id],
        {},
        {}
    )
    
    # Validate that the custom field is a dictionary before returning
    # Non-dictionary custom fields (strings, numbers) are not processable
    return custom_field if has_valid_custom_field(custom_field) else {}


def extract_request_type_from_entry(entry: Optional[JiraEntryDict]) -> Dict[str, Any]:
    """
    Extract service desk request type information from JIRA entry.
    
    Service desk request types provide additional categorization beyond standard
    issue types. They contain business-relevant information about how the request
    was submitted and what type of service is being requested.
    
    Args:
        entry: JIRA entry dictionary from API response, can be None
        
    Returns:
        Dict[str, Any]: Request type dictionary or empty dict if not found/accessible
        
    Examples:
        >>> entry = {
        ...     'fields': {
        ...         'customfield_23641': {
        ...             'requestType': {
        ...                 'name': 'Bug Report',
        ...                 'description': 'Report a software bug'
        ...             }
        ...         }
        ...     }
        ... }
        >>> result = extract_request_type_from_entry(entry)
        >>> result['name']
        'Bug Report'
        
    Business Context:
        Request types are used for:
        - Categorizing service desk requests
        - Applying different SLA rules
        - Routing to appropriate teams
        - Reporting on service delivery metrics
    """
    # Extract the request type custom field safely
    custom_field: Dict[str, Any] = extract_custom_field_safely(entry, JiraFields.REQUEST_TYPE_CUSTOM_FIELD)
    
    # Extract the request type subfield, returning empty dict if not found
    return safely_access_single_field(custom_field, JiraFields.REQUEST_TYPE_SUBFIELD, {})


def extract_request_type_details(request_type: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract name and description from request type dictionary.
    
    Request types contain both a name (for categorization) and description  
    (for user guidance). Both are optional fields that may be missing.
    
    Args:
        request_type: Request type dictionary from JIRA custom field
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (request_type_name, request_type_description)
        
    Examples:
        >>> request_type = {
        ...     'name': 'Bug Report',
        ...     'description': 'Report a software bug that needs fixing'
        ... }
        >>> name, desc = extract_request_type_details(request_type)
        >>> name
        'Bug Report'
        >>> desc
        'Report a software bug that needs fixing'
        
        >>> extract_request_type_details({})
        (None, None)
        
    Business Context:
        Request type details are used for:
        - User-friendly display in reports
        - Understanding the purpose of each request
        - Categorizing requests for analysis
        - Providing context for service delivery metrics
    """
    name: Optional[str] = safely_access_single_field(request_type, JiraFields.NAME_SUBFIELD, None)
    description: Optional[str] = safely_access_single_field(request_type, JiraFields.DESCRIPTION_SUBFIELD, None)
    return name, description


def extract_components_from_entry(entry: Optional[JiraEntryDict]) -> Any:
    """
    Extract components field from JIRA entry for business rule filtering.
    
    Components in JIRA represent functional areas or teams responsible for
    different parts of a system. They're used in business rules to filter
    which issues should be included in KPI calculations.
    
    Args:
        entry: JIRA entry dictionary from API response, can be None
        
    Returns:
        Any: Components field value (typically a list) or None if not found
        
    Examples:
        >>> entry = {
        ...     'fields': {
        ...         'components': [
        ...             {'name': 'Frontend', 'description': 'UI components'},
        ...             {'name': 'Backend', 'description': 'API services'}
        ...         ]
        ...     }
        ... }
        >>> components = extract_components_from_entry(entry)
        >>> len(components)
        2
        >>> components[0]['name']
        'Frontend'
        
    Business Context:
        Components are used for:
        - Filtering issues by functional area
        - Routing to appropriate teams
        - Applying different business rules per component
        - Granular reporting and analysis
        Business rules may exclude certain components from KPI calculations.
    """
    return safely_access_nested_field(entry, [JiraFields.FIELDS, JiraFields.COMPONENTS], None, {})


# ========================================
# REFACTORED TICKET PROCESSING FUNCTIONS (USING UTILITIES)
# ========================================

def build_request_type_update_dict(
    request_type_name: Optional[str], 
    request_type_description: Optional[str]
) -> TicketValuesDict:
    """
    Build dictionary for updating ticket values with request type information.
    
    Creates a standardized dictionary structure for adding request type information
    to ticket values. This separates the data structure creation from the business
    logic of extracting request type information.
    
    Args:
        request_type_name: Service desk request type name, can be None
        request_type_description: Service desk request type description, can be None
        
    Returns:
        TicketValuesDict: Dictionary with request type fields for ticket update
        
    Examples:
        >>> update_dict = build_request_type_update_dict('Bug Report', 'Report software bugs')
        >>> update_dict[FieldNames.REQUEST_TYPE]
        'Bug Report'
        >>> update_dict[FieldNames.REQUEST_DESCRIPTION]
        'Report software bugs'
        
        >>> build_request_type_update_dict(None, None)
        {'RequestType': None, 'RequestDescription': None}
        
    Business Context:
        Request type information is used throughout the processing pipeline for:
        - Issue categorization and routing
        - Applying appropriate business rules
        - Preferred issue type calculations
        - Service delivery reporting
    """
    return {
        FieldNames.REQUEST_TYPE: request_type_name,
        FieldNames.REQUEST_DESCRIPTION: request_type_description
    }


def build_ticket_values_from_entry(entry: Optional[JiraEntryDict]) -> TicketValuesDict:
    """
    Build complete ticket values dictionary from JIRA entry with comprehensive error handling.
    
    This is the main function for converting a raw JIRA API entry into a processed
    ticket values dictionary suitable for KPI calculations. It orchestrates all
    the data extraction and enrichment steps.
    
    Args:
        entry: JIRA entry dictionary from API response, can be None
        
    Returns:
        TicketValuesDict: Complete ticket values dictionary or empty dict if processing fails
        
    Raises:
        None: All errors are handled gracefully by returning empty dict
        
    Examples:
        >>> entry = {
        ...     'key': 'PROJ-123',
        ...     'fields': {
        ...         'created': '2024-01-01T10:00:00.000+0000',
        ...         'resolutiondate': '2024-01-02T10:00:00.000+0000',
        ...         # ... other fields
        ...     }
        ... }
        >>> ticket_values = build_ticket_values_from_entry(entry)
        >>> ticket_values['Key']  # Available if get_fields() succeeds
        'PROJ-123'
        
    Business Context:
        Ticket values are the foundation for all downstream processing:
        - KPI calculations depend on complete ticket data
        - Business rules filter based on ticket properties
        - Reporting uses ticket values for categorization
        Robust error handling ensures processing continues even with incomplete data.
    """
    return safely_return_with_fallback(
        lambda: True,  # Always attempt processing - get_fields() handles its own validation
        lambda: _build_complete_ticket_values(entry),
        {}  # Return empty dict if any step fails
    )


def _build_complete_ticket_values(entry: Optional[JiraEntryDict]) -> TicketValuesDict:
    """
    Internal helper to build complete ticket values from validated entry.
    
    This function performs all the steps needed to build a complete ticket values
    dictionary, including extraction, enrichment, and business rule application.
    
    Args:
        entry: JIRA entry dictionary, may be None
        
    Returns:
        TicketValuesDict: Complete ticket values or empty dict if any step fails
    """
    # Step 1: Extract base ticket values using jira_parser.get_fields()
    # This handles the core JIRA field extraction and validation
    ticket_values: TicketValuesDict = get_fields(entry)
    if not ticket_values:
        return {}
    
    # Step 2: Extract and add request type information
    # Service desk request types provide additional categorization
    request_type: Dict[str, Any] = extract_request_type_from_entry(entry)
    request_type_name, request_type_description = extract_request_type_details(request_type)
    request_type_update: TicketValuesDict = build_request_type_update_dict(
        request_type_name, 
        request_type_description
    )
    ticket_values.update(request_type_update)
    
    # Step 3: Extract and add component information  
    # Components are used for business rule filtering
    components: Any = extract_components_from_entry(entry)
    component_dict: Dict[str, Any] = process_components(components)
    ticket_values.update(component_dict)
    
    # Step 4: Apply business logic for close date calculation
    # Different issue types use different logic for determining close dates
    # (some use resolution date, others use fix version dates)
    return update_ticket_values(ticket_values)


def process_single_jira_entry(
    entry: Optional[JiraEntryDict], 
    matched_entries: List[TicketValuesDict]
) -> None:
    """
    Process a single JIRA entry and add all resulting changelog entries to matched_entries list.
    
    This function orchestrates the complete processing of a single JIRA entry,
    from extracting ticket values to processing changelog entries and adding
    them to the accumulator list for DataFrame conversion.
    
    Args:
        entry: JIRA entry dictionary from API response, can be None
        matched_entries: List to append processed entries to (MODIFIED IN PLACE)
        
    Returns:
        None: Function modifies matched_entries list in place
        
    Side Effects:
        - Modifies matched_entries list by appending processed changelog entries
        - Each entry can result in multiple changelog entries being added
        
    Examples:
        >>> entry = {'key': 'PROJ-123', 'fields': {...}, 'changelog': {...}}
        >>> matched_entries = []
        >>> process_single_jira_entry(entry, matched_entries)
        >>> len(matched_entries)  # Number of status changes found
        3
        
    Business Context:
        Single JIRA entries can generate multiple processed entries because:
        - Each status change becomes a separate row for KPI calculations
        - Timing metrics require individual status transition records
        - Changelog processing creates audit trail for analysis
        
    Performance Notes:
        - Function processes entries sequentially for memory efficiency
        - Changelog sorting is performed for chronological accuracy
        - Early return on invalid entries minimizes processing overhead
    """
    # Step 1: Build complete ticket values from JIRA entry
    # Early return if ticket values cannot be extracted (invalid/incomplete entry)
    ticket_values: TicketValuesDict = build_ticket_values_from_entry(entry)
    if not ticket_values:
        return  # Skip invalid entries to prevent downstream processing errors
    
    # Step 2: Extract changelog data for status transition analysis
    # Changelog is optional - issues without changes will have empty changelog
    changelog: Optional[Dict[str, Any]] = safely_access_single_field(entry, JiraFields.CHANGELOG, None)
    changelog_entries: List[ChangelogEntryDict] = extract_changelog_entries_from_history(changelog)
    
    # Step 3: Sort changelog entries chronologically for accurate duration calculations
    # Chronological order is critical for calculating time between status changes
    sorted_changelog_entries: List[ChangelogEntryDict] = sort_changelog_entries(changelog_entries)
    
    # Step 4: Process changelog entries and add to matched_entries
    # This creates individual records for each status change with calculated durations
    # The matched_entries list is modified in place for memory efficiency
    process_changelog_entries(sorted_changelog_entries, matched_entries, ticket_values)


# ========================================
# REFACTORED DATA PROCESSING PIPELINE FUNCTIONS (USING UTILITIES)
# ========================================

def convert_jira_entries_to_dataframe(json_data: List[JiraEntryDict]) -> pd.DataFrame:
    """
    Convert raw JIRA entries to processed DataFrame with comprehensive error handling.
    
    This function is the first step in the data processing pipeline. It takes raw
    JIRA API response data and converts it into a structured DataFrame suitable
    for analysis and KPI calculations.
    
    Args:
        json_data: List of raw JIRA entry dictionaries from API response
        
    Returns:
        pd.DataFrame: Processed entries as DataFrame with all changelog entries expanded
        
    Examples:
        >>> jira_data = [
        ...     {'key': 'PROJ-123', 'fields': {...}, 'changelog': {...}},
        ...     {'key': 'PROJ-124', 'fields': {...}, 'changelog': {...}}
        ... ]
        >>> df = convert_jira_entries_to_dataframe(jira_data)
        >>> df.columns
        ['Key', 'Author', 'ChangeCreated', 'StatusDuration', ...]
        
    Business Context:
        DataFrame conversion is necessary because:
        - Pandas provides efficient data manipulation operations
        - KPI calculations require aggregation and grouping operations
        - Downstream functions expect structured DataFrame format
        - DataFrame allows for consistent data validation and cleansing
        
    Performance Notes:
        - Processes entries sequentially to control memory usage
        - Each JIRA entry can generate multiple DataFrame rows (one per status change)
        - Final DataFrame creation is deferred until all entries are processed
    """
    # Initialize accumulator list for all processed changelog entries
    # Using list instead of DataFrame for memory efficiency during processing
    matched_entries: List[TicketValuesDict] = []
    
    # Process each JIRA entry individually with error isolation
    # If one entry fails, processing continues with remaining entries
    for entry in json_data:
        try:
            process_single_jira_entry(entry, matched_entries)
        except Exception as e:
            # Log error but continue processing - don't let one bad entry break the pipeline
            logging.warning(f"Failed to process JIRA entry {entry.get('key', 'unknown')}: {str(e)}")
            continue
    
    # Convert accumulated entries to DataFrame
    # pandas.DataFrame() handles empty list gracefully by creating empty DataFrame
    return pd.DataFrame(matched_entries)


def enhance_entries_with_preferred_types(
    matched_entries: pd.DataFrame, 
    mapping_table: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add preferred issue type columns to both matched entries and mapping table using DRY utility.
    
    The preferred issue type calculation determines which issue type to use for
    categorization when both JIRA issue type and service desk request type are available.
    This is a business rule that prioritizes request types over issue types.
    
    Args:
        matched_entries: Processed JIRA entries DataFrame
        mapping_table: Mapping table DataFrame for business rule application
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Enhanced matched_entries and mapping_table
        
    Business Context:
        Preferred issue type logic:
        - If service desk request type is available and valid, use it
        - Otherwise, fall back to standard JIRA issue type
        - This provides more granular categorization for service desk issues
        - Both DataFrames need the same logic for consistent mapping
        
    Performance Notes:
        - Uses utility function to avoid code duplication
        - Single transformation function call per DataFrame
        - Maintains referential integrity between matched entries and mapping
    """
    # Use DRY utility to apply same transformation to multiple DataFrames
    # This eliminates code duplication and ensures consistent logic
    enhanced_dataframes: List[pd.DataFrame] = apply_transformation_to_multiple_dataframes(
        [matched_entries, mapping_table],
        add_preferred_issue_type,
        FieldNames.REQUEST_TYPE,      # Source field 1: Service desk request type
        FieldNames.ISSUE_TYPE,        # Source field 2: JIRA issue type  
        FieldNames.PREFERRED_ISSUE_TYPE  # Target field: Calculated preferred type
    )
    
    # Return enhanced DataFrames in original order
    return enhanced_dataframes[0], enhanced_dataframes[1]


def merge_requestor_and_mapping_data(
    matched_entries: pd.DataFrame, 
    requestor_df: pd.DataFrame, 
    mapping_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge requestor data and mapping table with matched entries, including intermediate file generation.
    
    This function performs multiple merge operations and generates intermediate outputs
    for audit and debugging purposes. It's a critical step that enriches the base
    JIRA data with business context.
    
    Args:
        matched_entries: Enhanced matched entries DataFrame with preferred issue types
        requestor_df: Requestor mapping DataFrame (maps project initiatives to service users)
        mapping_table: Enhanced mapping table DataFrame with business rules
        
    Returns:
        pd.DataFrame: Fully merged DataFrame ready for business rule application
        
    Side Effects:
        - Saves matched_entries.csv file for debugging/audit purposes
        - Generates unmapped_requestors.csv report for data quality monitoring
        
    Business Context:
        Merge operations add:
        - Service user information from project initiative mappings
        - Business categorization rules from mapping table
        - Data quality reports for unmapped requestors
        These enrichments are essential for accurate KPI calculations and reporting.
        
    Data Quality:
        - Unmapped requestors report identifies data quality issues
        - CSV outputs provide audit trail for troubleshooting
        - Left joins preserve all original data even with missing mappings
    """
    # Step 1: Merge requestor data to add service user information
    # This maps project initiatives to responsible service users/teams
    entries_with_requestors: pd.DataFrame = merge_requestor_data(matched_entries, requestor_df)
    
    # Step 2: Generate data quality report for unmapped requestors
    # This identifies project initiatives that don't have service user mappings
    # Important for data quality monitoring and business process improvement
    unmapped_requestors: pd.DataFrame = generate_unmapped_requestors_report(entries_with_requestors)
    
    # Step 3: Save intermediate results for audit and debugging
    # CSV files provide visibility into processing pipeline for troubleshooting
    entries_with_requestors.to_csv(OutputConfig.OUTPUT_FILES["MATCHED_ENTRIES"], index=False)
    
    # Step 4: Merge mapping table to apply business categorization rules  
    # This adds service categories and other business metadata needed for KPIs
    fully_merged_entries: pd.DataFrame = merge_mapping_tables(entries_with_requestors, mapping_table)
    
    return fully_merged_entries


def prepare_kpi_targets_with_service_mapping(kpi_targets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SERVICE_KPI composite key column to KPI targets DataFrame using DRY utility.
    
    The SERVICE_KPI composite key is used throughout the system for joining
    KPI targets with calculated results. This function creates this key using
    a standardized column creation utility.
    
    Args:
        kpi_targets_df: KPI targets DataFrame with Service and KPI columns
        
    Returns:
        pd.DataFrame: KPI targets with SERVICE_KPI composite key column added
        
    Examples:
        >>> targets = pd.DataFrame({
        ...     'Service': ['ServiceA', 'ServiceB'], 
        ...     'KPI': ['Lead', 'Cycle']
        ... })
        >>> result = prepare_kpi_targets_with_service_mapping(targets)
        >>> result['Service KPI'].tolist()
        ['ServiceA - Lead', 'ServiceB - Cycle']
        
    Business Context:
        The SERVICE_KPI composite key is used for:
        - Joining KPI targets with calculated metrics
        - Ensuring consistent identification across all processing steps
        - Supporting granular KPI analysis at service+metric level
        Standard separator (' - ') provides readable composite keys.
    """
    return create_dataframe_column_from_operation(
        kpi_targets_df,
        FieldNames.SERVICE_KPI,              # New composite key column
        [FieldNames.SERVICE, 'KPI'],         # Source columns to combine
        'concat',                            # Concatenation operation
        {'separator': ' - '}                 # Business-standard separator
    )


def apply_business_rules_and_formatting(
    matched_entries: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply business rules filtering and add resolution date formatting using DRY utilities.
    
    This function applies critical business rules that determine which issues
    are included in KPI calculations, and adds formatting needed for monthly
    aggregation and reporting.
    
    Args:
        matched_entries: Fully merged DataFrame with all enrichments applied
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (filtered_entries, services_categories_mapping)
        
    Business Rules Applied:
        - Component filtering: Only specific components included per project-issuetype
        - Date formatting: Resolution dates formatted for monthly grouping (YYYY-MM)
        
    Examples:
        >>> entries = pd.DataFrame({
        ...     'Service': ['A', 'B'], 
        ...     'Category': ['Cat1', 'Cat2'],
        ...     'close_date': ['2024-01-15T10:00:00', '2024-02-20T15:30:00']
        ... })
        >>> filtered, categories = apply_business_rules_and_formatting(entries)
        >>> filtered['ResolutionDate_yyyy_mm'].tolist()
        ['2024-01', '2024-02']
        
    Business Context:
        Business rules ensure:
        - Only relevant issues are included in KPIs (component filtering)
        - Consistent monthly aggregation (date formatting)
        - Service-to-category mapping available for hierarchical reporting
    """
    # Step 1: Apply component filtering business rules
    # These rules exclude specific components from KPI calculations based on project-issuetype
    # Filtering rules are defined in BusinessRules.PROJECT_COMPONENT_FILTERS
    filtered_entries: pd.DataFrame = filter_matched_entries(
        matched_entries, 
        BusinessRules.PROJECT_COMPONENT_FILTERS
    )
    
    # Step 2: Add formatted resolution date column for monthly grouping
    # Extracts YYYY-MM from full datetime for monthly KPI aggregation
    # Uses DRY utility for consistent column creation patterns
    filtered_entries = create_dataframe_column_from_operation(
        filtered_entries,
        FieldNames.RESOLUTION_DATE_YYYY_MM,   # Target column for monthly grouping
        [FieldNames.CLOSE_DATE],              # Source: calculated close date
        'astype_slice',                       # Convert to string and slice
        {'start': 0, 'end': 7}               # Extract first 7 characters (YYYY-MM)
    )
    
    # Step 3: Extract service to category mapping for hierarchical reporting
    # This mapping is used for category-level SLO views and executive reporting
    # Drop duplicates to create clean lookup table
    services_categories: pd.DataFrame = filtered_entries[
        [FieldNames.SERVICE, FieldNames.CATEGORY]
    ].drop_duplicates()
    
    return filtered_entries, services_categories


def calculate_all_kpi_metrics(
    filtered_entries: pd.DataFrame, 
    kpi_targets_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate all KPI metrics from filtered entries using specialized calculation functions.
    
    This function orchestrates all KPI calculations by calling specialized functions
    for different types of metrics. Each calculation type requires different
    aggregation logic and business rules.
    
    Args:
        filtered_entries: Business rule filtered DataFrame with all required fields
        kpi_targets_df: KPI targets DataFrame with composite keys for joining
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            (throughput_results, time_results, issue_rate_results)
            
    KPI Types Calculated:
        1. Throughput: Tickets processed per time period (count-based metrics)
        2. Time: Lead time, cycle time, response time (duration-based metrics)  
        3. Issue Rate: Percentage of tickets with linked issues (quality metrics)
        
    Business Context:
        Different KPI types require different calculation approaches:
        - Throughput: Count unique tickets per service per month
        - Time: Calculate average durations between status changes
        - Issue Rate: Calculate percentage of tickets with complications
        All calculations respect service-level targets and business rules.
    """
    # Calculate throughput metrics (tickets processed per time period)
    # Includes ticket counts, hours logged, and processing rates
    throughput_results: pd.DataFrame = standard_throughput_calculation(filtered_entries, kpi_targets_df)
    
    # Calculate time-based metrics (lead time, cycle time, response time, etc.)
    # Includes status transition durations and overall resolution times
    time_results: pd.DataFrame = get_time_status_per_month(filtered_entries, kpi_targets_df)
    
    # Calculate issue rate metrics (percentage of tickets with linked issues)
    # Used as proxy for complexity/quality metrics
    issue_rate_results: pd.DataFrame = get_issue_rate(filtered_entries)
    
    return throughput_results, time_results, issue_rate_results


def create_all_powerbi_views(
    all_results_kpi: pd.DataFrame,
    recent_results_kpi: pd.DataFrame,
    kpi_insights: pd.DataFrame,
    services_categories: pd.DataFrame,
    category_definitions_df: pd.DataFrame,
    kpi_definitions_df: pd.DataFrame,
    kpi_targets_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create all Power BI views and enhancements for comprehensive business intelligence reporting.
    
    This function creates multiple specialized views of the KPI data optimized for
    different Power BI visualization types and business reporting needs.
    
    Args:
        all_results_kpi: Complete historical KPI results for trending analysis
        recent_results_kpi: Recent period KPI results for current state reporting
        kpi_insights: KPI insights with calculated averages and changes
        services_categories: Service to category mapping for hierarchical views
        category_definitions_df: Category definitions for tooltip/documentation
        kpi_definitions_df: KPI definitions for tooltip/documentation  
        kpi_targets_df: KPI targets for comparison and matrix views
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (processed_all_results, enhanced_recent_results, enhanced_insights, 
             matrix_view, category_slo_view, service_slo_view)
             
    Power BI Views Created:
        1. Processed All Results: Historical data with date ordering for time series
        2. Enhanced Recent Results: Current state with definition tooltips
        3. Enhanced Insights: Trend analysis with change indicators
        4. Matrix View: Tabular format with targets, changes, and averages
        5. Category SLO: Executive-level SLO percentages by business category
        6. Service SLO: Operational-level SLO percentages by service
        
    Business Context:
        Different stakeholders need different views:
        - Executives: Category-level SLOs and high-level trends
        - Operations: Service-level SLOs and detailed metrics
        - Analysts: Historical trends and detailed breakdowns
        Each view is optimized for specific Power BI visualization types.
    """
    # Step 1: Process historical results for trending analysis
    # Remove KPI Type column (not needed for Power BI) and add date ordering
    all_results_kpi = all_results_kpi.drop(columns=[FieldNames.KPI_TYPE], errors='ignore')
    processed_all_results: pd.DataFrame = process_all_results_for_power(all_results_kpi)
    
    # Step 2: Enhance recent results with definition tooltips
    # Adds human-readable descriptions for categories and KPI types
    enhanced_recent_results: pd.DataFrame = enhance_recent_results(
        recent_results_kpi, 
        category_definitions_df, 
        kpi_definitions_df
    )
    
    # Step 3: Create SLO percentage views for executive and operational reporting
    # Category view: Aggregated by business category for executive dashboards
    category_slo_view: pd.DataFrame = create_slo_category_view(processed_all_results, services_categories)
    
    # Service view: Detailed by service for operational teams
    service_slo_view: pd.DataFrame = create_slo_service_view(processed_all_results)
    
    # Step 4: Create insights matrix with change analysis
    # Combines KPI insights with targets and change indicators for comprehensive analysis
    enhanced_insights: pd.DataFrame
    matrix_view: pd.DataFrame
    enhanced_insights, matrix_view = create_insights_matrix(kpi_insights, kpi_targets_df)
    
    return (
        processed_all_results,    # Historical trends with date ordering
        enhanced_recent_results,  # Current state with definitions
        enhanced_insights,        # Trend analysis with change metrics
        matrix_view,             # Tabular format with targets and changes
        category_slo_view,       # Executive SLO percentages
        service_slo_view         # Operational SLO percentages
    )


def generate_final_output_dictionary(
    matched_entries: pd.DataFrame,
    throughput_results: pd.DataFrame,
    time_results: pd.DataFrame,
    issue_rate_results: pd.DataFrame,
    all_results_kpi: pd.DataFrame,
    recent_results_kpi: pd.DataFrame,
    kpi_insights: pd.DataFrame,
    matrix_view: pd.DataFrame,
    category_slo_view: pd.DataFrame,
    service_slo_view: pd.DataFrame,
    requestor_data_clean: pd.DataFrame,
    requestor_data_clean_grouped: pd.DataFrame
) -> OutputDictionary:
    """
    Generate final output dictionary with all processed DataFrames for Excel export.
    
    This function assembles all processed DataFrames into a structured dictionary
    that will be used to create Excel workbook with multiple sheets. The dictionary
    maintains consistent ordering for reliable output generation.
    
    Args:
        matched_entries: Raw processed JIRA entries with all enrichments
        throughput_results: Calculated throughput KPI metrics
        time_results: Calculated time-based KPI metrics
        issue_rate_results: Calculated issue rate metrics
        all_results_kpi: Complete historical KPI data for Power BI
        recent_results_kpi: Recent period KPI data with definitions
        kpi_insights: KPI insights with change analysis
        matrix_view: Matrix table format for Power BI
        category_slo_view: Category-level SLO percentages
        service_slo_view: Service-level SLO percentages
        requestor_data_clean: Clean requestor analysis data
        requestor_data_clean_grouped: Grouped requestor analysis
        
    Returns:
        OutputDictionary: Dictionary mapping sheet names to DataFrames for Excel export
        
    Excel Sheet Organization:
        - Raw Data: Matched entries and calculation inputs
        - KPI Calculations: Throughput, time, and issue rate results
        - Power BI Views: Formatted data for business intelligence
        - Analysis: SLO views and requestor analysis
        
    Business Context:
        Excel output serves multiple purposes:
        - Data validation and audit trails
        - Manual analysis and reporting
        - Power BI data source (some sheets)
        - Business user access to processed data
        Consistent sheet ordering improves user experience.
    """
    # Use assembly function to create standardized output structure
    # This function handles the complex parameter mapping and ensures consistent sheet naming
    return assemble_final_dataframes_dict(
        matched_entries, throughput_results, time_results, issue_rate_results,
        all_results_kpi, recent_results_kpi, kpi_insights, matrix_view,
        category_slo_view, service_slo_view, requestor_data_clean, requestor_data_clean_grouped
    )


# ========================================
# MAIN ORCHESTRATION FUNCTIONS (COMPREHENSIVE DOCUMENTATION)
# ========================================

def process_jira_data_pipeline(
    json_data: List[JiraEntryDict], 
    mapping_table: pd.DataFrame, 
    kpi_targets_df: pd.DataFrame, 
    category_definitions_df: pd.DataFrame, 
    kpi_definitions_df: pd.DataFrame, 
    requestor_df: pd.DataFrame, 
    output_xlsx_path: str
) -> OutputDictionary:
    """
    Main pipeline for processing JIRA data through complete KPI calculation workflow.
    
    This is the primary orchestration function that coordinates all data processing
    steps from raw JIRA API data to final business intelligence outputs. It implements
    a robust pipeline with comprehensive error handling and audit trails.
    
    Args:
        json_data: Raw JIRA API response data as list of entry dictionaries
        mapping_table: Issue type mapping table for business categorization rules
        kpi_targets_df: KPI targets and definitions for performance measurement
        category_definitions_df: Service category definitions for tooltips and documentation
        kpi_definitions_df: KPI type definitions for tooltips and documentation
        requestor_df: Requestor mapping table (project initiatives to service users)
        output_xlsx_path: Output Excel file path (used for audit trail documentation)
        
    Returns:
        OutputDictionary: Complete processed data dictionary ready for Excel export
        
    Raises:
        ValueError: If any critical processing step fails (data validation errors)
        FileNotFoundError: If required input files are missing
        PermissionError: If output file cannot be written
        
    Pipeline Steps:
        1. Data Conversion: Raw JIRA → Structured DataFrame
        2. Enhancement: Add preferred issue types and business categorization
        3. Enrichment: Merge requestor data and mapping rules
        4. Preparation: Prepare KPI targets with composite keys
        5. Filtering: Apply business rules and date formatting
        6. Calculation: Calculate all KPI metrics (throughput, time, issue rate)
        7. Views Creation: Generate Power BI optimized views
        8. Analysis: Create SLO views and requestor analysis
        9. Assembly: Package all results for export
        
    Business Context:
        This pipeline transforms raw JIRA data into business intelligence outputs:
        - Executive dashboards (category SLO views)
        - Operational metrics (service-level KPIs)
        - Trend analysis (historical KPI data)
        - Data quality reports (unmapped requestors)
        - Audit trails (raw processed data)
        
    Performance Considerations:
        - Sequential processing for memory efficiency
        - Error isolation between processing steps
        - Incremental data accumulation to handle large datasets
        - Comprehensive logging for troubleshooting
        
    Data Quality:
        - Validation at each step with graceful error handling
        - Audit trails through intermediate file generation
        - Data quality reports for business process improvement
        - Comprehensive error logging for operational monitoring
        
    Examples:
        >>> # Load raw JIRA data and mapping tables
        >>> jira_data = [{'key': 'PROJ-123', 'fields': {...}}, ...]
        >>> mapping_df = pd.read_excel('mapping.xlsx')
        >>> kpi_targets = pd.read_excel('targets.xlsx')
        >>> 
        >>> # Process through complete pipeline
        >>> results = process_jira_data_pipeline(
        ...     jira_data, mapping_df, kpi_targets, 
        ...     categories_df, kpi_defs_df, requestors_df, 'output.xlsx'
        ... )
        >>> 
        >>> # Results contain all processed DataFrames
        >>> results['All KPI Results'].shape
        (1500, 12)  # Historical KPI data
        >>> results['Category SLO Met Percent'].head()
        # Executive SLO view ready for dashboards
    """
    # Step 1: Convert raw JIRA data to structured DataFrame
    # This step handles malformed entries gracefully and creates audit trail
    logging.info("Step 1: Converting JIRA entries to DataFrame")
    matched_entries: pd.DataFrame = convert_jira_entries_to_dataframe(json_data)
    logging.info(f"Converted {len(json_data)} JIRA entries to {len(matched_entries)} processed entries")
    
    # Step 2: Enhance with preferred issue types for consistent categorization
    # Applies business logic to determine best issue type for each entry
    logging.info("Step 2: Enhancing entries with preferred issue types")
    matched_entries, mapping_table = enhance_entries_with_preferred_types(matched_entries, mapping_table)
    
    # Step 3: Merge requestor and mapping data for business context
    # Adds service ownership and business categorization rules
    logging.info("Step 3: Merging requestor and mapping data")
    matched_entries = merge_requestor_and_mapping_data(matched_entries, requestor_df, mapping_table)
    
    # Step 4: Prepare KPI targets with composite keys for joining
    # Creates SERVICE_KPI composite keys needed for all subsequent operations
    logging.info("Step 4: Preparing KPI targets with service mapping")
    kpi_targets_df = prepare_kpi_targets_with_service_mapping(kpi_targets_df)
    
    # Step 5: Apply business rules and formatting for KPI calculations
    # Filters data and adds monthly aggregation columns
    logging.info("Step 5: Applying business rules and formatting")
    matched_entries, services_categories = apply_business_rules_and_formatting(matched_entries)
    logging.info(f"Applied business rules, retained {len(matched_entries)} entries")
    
    # Step 6: Calculate all KPI metrics using specialized functions
    # Performs throughput, time, and issue rate calculations
    logging.info("Step 6: Calculating all KPI metrics")
    throughput_results, time_results, issue_rate_results = calculate_all_kpi_metrics(
        matched_entries, 
        kpi_targets_df
    )
    
    # Step 7: Create KPI result views for Power BI consumption
    # Combines and formats KPI results for business intelligence
    logging.info("Step 7: Creating KPI result views")
    all_results_kpi, recent_results_kpi, kpi_insights = create_kpi_result_views(
        throughput_results, 
        time_results
    )
    
    # Step 8: Create all Power BI views with enhancements
    # Generates specialized views for different stakeholder needs
    logging.info("Step 8: Creating Power BI views")
    (all_results_kpi, recent_results_kpi, kpi_insights, matrix_view, 
     category_slo_view, service_slo_view) = create_all_powerbi_views(
        all_results_kpi, recent_results_kpi, kpi_insights, services_categories,
        category_definitions_df, kpi_definitions_df, kpi_targets_df
    )
    
    # Step 9: Generate requestor analysis for service ownership insights
    # Creates reports on request patterns by service ownership
    logging.info("Step 9: Generating requestor analysis")
    requestor_data_clean, requestor_data_clean_grouped = get_requestor_data_clean(matched_entries)
    
    # Step 10: Assemble final output dictionary for Excel export
    # Packages all results into structured format for output generation
    logging.info("Step 10: Assembling final output dictionary")
    final_output: OutputDictionary = generate_final_output_dictionary(
        matched_entries, throughput_results, time_results, issue_rate_results,
        all_results_kpi, recent_results_kpi, kpi_insights, matrix_view,
        category_slo_view, service_slo_view, requestor_data_clean, requestor_data_clean_grouped
    )
    
    logging.info(f"Pipeline completed successfully. Generated {len(final_output)} output sheets.")
    return final_output


def get_issue_data(
    input_json: Union[str, Path], 
    mapping_table: pd.DataFrame, 
    kpi_targets_df: pd.DataFrame, 
    category_definitions_df: pd.DataFrame, 
    kpi_definitions_df: pd.DataFrame, 
    requestor_df: pd.DataFrame, 
    output_xlsx_path: Union[str, Path]
) -> OutputDictionary:
    """
    Load JSON file and process through complete JIRA data pipeline with comprehensive error handling.
    
    This is the main entry point for the JIRA data processing system. It handles file I/O
    and orchestrates the complete data processing pipeline from raw JSON input to
    business intelligence outputs.
    
    Args:
        input_json: Path to input JSON file containing JIRA API response data
        mapping_table: Issue type mapping table for business categorization rules
        kpi_targets_df: KPI targets and definitions for performance measurement
        category_definitions_df: Service category definitions for tooltips
        kpi_definitions_df: KPI type definitions for tooltips
        requestor_df: Requestor mapping table (project initiatives to service users)
        output_xlsx_path: Output Excel file path for results
        
    Returns:
        OutputDictionary: Complete processed data dictionary ready for Excel export
        
    Raises:
        FileNotFoundError: If input JSON file doesn't exist
        PermissionError: If input file cannot be read due to permissions
        json.JSONDecodeError: If input file contains invalid JSON
        ValueError: If JSON structure is invalid for processing
        MemoryError: If input file is too large for available memory
        
    File Format Requirements:
        - Input JSON must be valid JSON array of JIRA issue objects
        - Each issue object must follow JIRA REST API response format
        - Required fields: key, fields, changelog (changelog can be empty)
        - File encoding must be UTF-8
        
    Examples:
        >>> # Process JIRA data from file
        >>> results = get_issue_data(
        ...     'jira_export.json',
        ...     mapping_df,
        ...     targets_df, 
        ...     categories_df,
        ...     kpi_defs_df,
        ...     requestors_df,
        ...     'kpi_results.xlsx'
        ... )
        >>> 
        >>> # Access specific results
        >>> executive_view = results['Category SLO Met Percent']
        >>> operational_view = results['Service SLO Met Percent']  
        >>> audit_trail = results['Matched Entries']
        
    Business Context:
        This function serves as the main interface for:
        - Scheduled data processing jobs
        - Manual analysis and reporting
        - Data pipeline orchestration
        - Business intelligence data preparation
        
    Performance Notes:
        - Memory usage scales with input file size
        - Processing time depends on number of issues and changelog complexity
        - Large files (>100MB) may require increased memory allocation
        - Consider chunked processing for very large datasets
        
    Data Security:
        - Input files should be validated before processing
        - Consider data sanitization for sensitive information
        - Output files may contain business-sensitive KPI data
        - Implement appropriate access controls for file handling
    """
    # Validate input file exists and is readable
    input_path: Path = Path(input_json) if isinstance(input_json, str) else input_json
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    
    # Load and validate JSON data with comprehensive error handling
    try:
        logging.info(f"Loading JSON data from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as json_file:
            data: List[JiraEntryDict] = json.load(json_file)
        
        # Validate JSON structure
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain an array of JIRA entries, got: {type(data)}")
        
        if len(data) == 0:
            logging.warning("JSON file contains no entries")
            
        logging.info(f"Successfully loaded {len(data)} JIRA entries from JSON file")
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in file {input_path}: {str(e)}", e.doc, e.pos) from e
    except MemoryError as e:
        raise MemoryError(f"File {input_path} is too large for available memory") from e
    except PermissionError as e:
        raise PermissionError(f"Cannot read file {input_path}: insufficient permissions") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"File {input_path} is not valid UTF-8: {str(e)}") from e
    
    # Process data through complete pipeline with error context
    try:
        logging.info("Starting JIRA data processing pipeline")
        result: OutputDictionary = process_jira_data_pipeline(
            data, 
            mapping_table, 
            kpi_targets_df, 
            category_definitions_df, 
            kpi_definitions_df, 
            requestor_df, 
            str(output_xlsx_path)  # Convert Path to string for downstream compatibility
        )
        
        logging.info("JIRA data processing pipeline completed successfully")
        return result
        
    except Exception as e:
        logging.error(f"Pipeline processing failed for file {input_path}: {str(e)}")
        raise ValueError(f"Failed to process JIRA data from {input_path}: {str(e)}") from e