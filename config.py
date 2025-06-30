"""
Config module for Jira KPI Processing System

This module contains all constants, configuration parameters and mappings used throughout the Jira data processing
And calculation system
"""

from dataclasses import dataclass
from typing import Dict, List, Final

# ========
# FIELD NAMES - Organised into logical groups
# ========

@dataclass(frozen=True)
class FieldNames:
    """Core field names used across the system"""
    ISSUE_TYPE = "IssueType"
    REQUEST_TYPE = "RequestType"
    PREFERRED_ISSUE_TYPE = "PreferredIssueType"
    PROJECT = "Project"
    STATUS = "Status"
    FROM = "From"
    
    # KPI and service fields
    SERVICE_KPI = "Service KPI"
    DATES = "dates"
    KPI_MET = "KPI Met"
    TARGET = "Target"
    
    # Definition fields
    CATEGORY_DEFINITION = "Category Definition"
    KPI_DEFINITION = "KPI Definition"
    
    # Date and time columns
    RESOLUTION_DATE_YYYY_MM = "ResolutionDate_yyyy_mm"
    TOTAL_STATUS_DURATION_COLUMN = "TotalStatusDuration"
    TIME_TYPE_COLUMN = "Time Type"
    INCLUSIVE_DURATION_COLUMN = "InclusiveDuration"
    
    # Project initiative columns
    PROJECT_INITIATIVE_L1_COLUMN = "project_initiative_l1"
    PROJECT_INITIATIVE_L2_COLUMN = "project_initiative_l2"
    
    # File-specific columns
    SERVICE_USER_COLUMN = "Service user"

    # Ticket data fields (new additions for jira_parser)
    KEY = "Key"
    RESOLUTION_DATE = "ResolutionDate"
    PROJECT_ISSUE_TYPE = "project-issuetype"
    CREATED = "Created"
    HAS_ISSUES = "HasIssues"
    CCF_DATE = "CCFDate"
    TICKET_DURATION = "TicketDuration"
    FIX_VERSION = "FixVersion"
    TIME_SPENT_SECONDS = "TimespentSeconds"
    COMPONENT_NAME = "ComponentName"
    COMPONENT_DESCRIPTION = "ComponentDescription"
    PARENT_VALUE = "parent_value"
    CHILD_VALUE = "child_value"

    # PowerBi fields
    AVERAGE_KPI_VALUE = "Average KPI Value"        # Eliminated string literal duplication
    CHANGE_IN_KPI_VALUE = "Change in KPI Value"    # PowerBI column consistency  
    KPI_MET_PERCENTAGE = "KPI_Met_Percentage"      # SLO calculation standardization
    CHANGE_IMPACT = "change_impact"                # Impact assessment columns
    CHANGE_DIRECTION = "change_direction"          # Arrow indicator columns
    STAT = "Stat"                                  # Matrix view structure
    ARROW = "Arrow"                               # Visual indicator columns
    VALUE = "Value"                               # Generic value columns
    LINE_ORDER = "line_order"                     # Matrix ordering
    DATE_ORDER = "DateOrder"                      # PowerBI date sorting
    RESOLUTION_DATE_STRING = "ResolutionDate_string" # Formatted display dates
    DEFINITION = "Definition"                      # Lookup table standard
    SERVICE = "Service"                           # Core business entity
    CATEGORY = "Category"                         # Service grouping
    COMPONENT_NAME = "ComponentName"              # Jira component field
    KPI_VALUE = "KPI Value"                       # Calculated metrics
    KPI_TYPE = "KPI Type"                         # Metric categorization

    # Data processor
    CLOSE_DATE = "close_date"                    # Used 5+ times, calculated field
    CHANGE_CREATED = "ChangeCreated"             # Used 6+ times, changelog field
    STATUS_DURATION = "StatusDuration"           # Used 5+ times, calculated metric
    AUTHOR = "Author"                            # Used 3+ times, changelog field
    FIELD = "Field"                              # Used 3+ times, changelog field

@dataclass(frozen=True)
class JiraFields:
    """Jira-specific field mappings"""
    # Main field containers
    FIELDS = "fields"
    
    # Core Jira fields
    KEY = "key"
    RESOLUTION_DATE = "resolutiondate"
    ISSUE_LINKS = "issuelinks"
    ISSUE_TYPE = "issuetype"
    CREATED = "created"
    FIX_VERSIONS = "fixVersions"
    TIME_SPENT = "timespent"
    
    # Custom fields
    REQUEST_TYPE_CUSTOM_FIELD = "customfield_23641"
    CCF_DATE_CUSTOM_FIELD = "customfield_13454"
    PROJECT_INITIATIVE_CUSTOM_FIELD = "customfield_28846"
    
    # Subfield names
    REQUEST_TYPE_SUBFIELD = "requestType"
    NAME_SUBFIELD = "name"
    VALUE_SUBFIELD = "value"
    CHILD_SUBFIELD = "child"
    DESCRIPTION_SUBFIELD = "description"

@dataclass(frozen=True)
class DataFrameColumns:
    """DataFrame column names used across the system"""
    NAME = "Name"
    DESCRIPTION = "Description"

class DefaultValues:
    """Default values used throughout the system"""
    UNKNOWN = "Unknown"
    EMPTY_STRING = ""
    PROJECT_SEPARATOR = "-"

@dataclass(frozen=True)
class ProcessingConfig:
    """Processing and calculation configuration"""
    
    # Unit conversion factors
    CONVERSION_FACTORS = {
        "hours": 1,
        "days": 1 / 24,
        "weeks": 1 / 168
    }
    
    # Month abbreviations for date parsing
    MONTH_ABBREVIATIONS = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    
    # Date format for calculate close dates
    CLOSE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.000+1000"
    DEFAULT_CLOSE_TIME_HOUR = 14
    DEFAULT_CLOSE_TIME_MINUTE = 54
    DEFAULT_CLOSE_TIME_SECOND = 44
    YEAR_PREFIX = "20"

    # Analysis settings
    HISTORICAL_MONTHS = 7

    # Array indices (new additions for jira_parser)
    FIRST_ELEMENT_INDEX = 0
    PROJECT_KEY_INDEX = 0

    # Time conversion constants
    SECONDS_TO_HOURS = 3600                      # Time conversion business rule
    MINIMUM_DURATION = 0                         # Duration validation threshold
    
    # String processing constants
    EMPTY_STRING = ""                            # Used 8+ times for validation
    NULL_VALUE_INDICATOR = "N/A"                 # Business rule for null detection
    SPACE_SEPARATOR = " "                        # Used 3+ times for splitting
    
    # Default values
    UNKNOWN_VALUE = "Unknown"                    # Used 6+ times as fallback
    INVALID_DATE_VALUE = "Invalid Date"          # Used 4+ times as error indicator
    
    # Array indexing constants
    LAST_ELEMENT_INDEX = -1                      # Used 3+ times for indexing
    YEAR_SUFFIX_INDEX = -2                       # Used 2+ times for slicing


@dataclass(frozen=True)
class ValidationConfig:

    # Year validation range (reasonable business years)
    MIN_VALID_YEAR = 2000                        # Business rule for year validation
    MAX_VALID_YEAR = 2100                        # Business rule for year validation  
    
    # Duration validation (in hours)
    MIN_REASONABLE_DURATION = 0                  # Data quality threshold
    MAX_REASONABLE_DURATION = 8760               # Data quality threshold (1 year)
    
    # DataFrame validation
    MIN_DATAFRAME_ROWS = 0
    MAX_DATAFRAME_ROWS = 1000000              # Performance limit
    
    # Required DataFrame columns for different operations
    REQUIRED_COLUMNS_MERGE_MAPPING = {
        FieldNames.PROJECT, 
        FieldNames.PREFERRED_ISSUE_TYPE, 
        FieldNames.FROM
    }
    
    REQUIRED_COLUMNS_MERGE_REQUESTOR = {
        FieldNames.PROJECT_INITIATIVE_L1_COLUMN,
        FieldNames.PROJECT_INITIATIVE_L2_COLUMN
    }
    
    REQUIRED_COLUMNS_UNMAPPED_REPORT = {
        FieldNames.SERVICE_USER_COLUMN,
        FieldNames.PROJECT_INITIATIVE_L1_COLUMN,
        FieldNames.PROJECT_INITIATIVE_L2_COLUMN
    }

    # FixVersion format validation
    FIXVERSION_MIN_PARTS = 2                     # String validation rule
    FIXVERSION_MONTH_LENGTH = 3                  # String validation rule
    FIXVERSION_YEAR_SUFFIX_LENGTH = 2            # String validation rule

    # File operation validation
    MAX_FILE_SIZE_MB = 100
    ALLOWED_FILE_EXTENSIONS = {'.csv', '.xlsx', '.json'}

    # Business rule validation
    VALID_MONTH_ABBREVIATIONS = set(ProcessingConfig.MONTH_ABBREVIATIONS.keys())
    
    # Error tolerance settings
    MAX_CONSECUTIVE_ERRORS = 10
    ERROR_RATE_THRESHOLD = 0.1  # 10% error rate threshold

@dataclass(frozen=True)
class BusinessRules:
    """Business-specific filtering and processing rules"""
    
    # Component filtering rules for specific project-issuetypes
    PROJECT_COMPONENT_FILTERS = {
        "DQMMBAU-Data Quality Rule": ["DQMMBAU - New DQ Rule", "DQMMBAU - Update DQ Rule"],
        "DQMMBAU-Consumer Validation": ["DQMMBAU - New DQ Rule", "DQMMBAU - Update DQ Rule"],
        "PMCIRIS-PFI Sub-Task": ["Testing - PFI E2E"],
    }
    
    # Project types that use FixVersion for close date calculation
    FIX_VERSION_CLOSE_DATE_PROJECTS = [
        "DQMMBAU-Data Quality Rule",
        "DQMMBAU-Consumer Validation"
    ]
    
    # KPI types to include in throughput calculations
    THROUGHPUT_KPI_TYPES = ["Throughput"]
    
    # KPI types to include in time-based calculations
    TIME_BASED_KPI_TYPES = ["Lead", "Cycle", "Response", "Resolution"]
    
    # Columns to drop during data processing
    COLUMNS_TO_DROP_AFTER_MERGE = ["IssueType", "RequestType"]
    
    # Time type mappings for lead sub-status
    TIME_TYPE_MAPPINGS = {
        "Lead": "Lead sub-status"
    }

@dataclass(frozen=True)
class OutputConfig:
    """Output file and data structure configuration"""
    
    # CSV output file names
    OUTPUT_FILES = {
        "UNMAPPED_REQUESTORS": "unmapped_requestors.csv",
        "MATCHED_ENTRIES": "matched_entries.csv",
        "DATA_BEFORE_MERGE": "data_before_merge.csv",
        "DATA_AFTER_MERGE": "data_after_merge.csv"
    }
    
    # DataFrame names for final output collection
    DATAFRAME_NAMES = {
        "MATCHED_ENTRIES": "Matched Entries",
        "TICKETS_CLOSED": "Tickets Closed",
        "TIME_KPIS": "Time KPIs",
        "ISSUE_RATE": "Issue Rate",
        "ALL_KPI_RESULTS": "All KPI Results",
        "RECENT_KPI_RESULTS": "Recent KPI Results",
        "KPI_INSIGHTS": "KPI Insights",
        "KPI_MATRIX": "KPI Matrix",
        "CATEGORY_SLO": "Category SLO Met Percent",
        "SERVICE_SLO": "Service SLO Met Percent",
        "REQUEST_DATA_CLEAN": "Request Data Clean",
        "REQUEST_DATA_GROUPED": "Request Data Clean Grouped"
    }
    
    # Matrix view configuration
    MATRIX_VIEW_STATS = {
        "TARGET": "Target (red line)",
        "CHANGE": "Change (Month on Month)",
        "AVERAGE": "6-Month Average"
    }
    
    # Matrix view line ordering
    MATRIX_LINE_ORDER = {
        "TARGET": 1,
        "CHANGE": 2,
        "AVERAGE": 3
    }

@dataclass(frozen=True)
class PandasConfig:
    """Pandas-specific configuration parameters"""
    
    # Merge and join parameters
    MERGE_HOW_LEFT = "left"
    ERROR_HANDLING_IGNORE = "ignore"
    INDEX_FALSE = False
    
    # Axis parameters
    AXIS_COLUMNS = 1

@dataclass(frozen=True)
class ErrorMessages:
    """Standardized error messages for logging and debugging"""
    
    # Original error messages
    FIX_VERSION_PROCESSING_ERROR = "Error processing FixVersion"
    DATE_PARSING_ERROR = "Error parsing dates"
    DURATION_DEBUG_MESSAGE = "total_hours is type {type} and value {value}"
    
    # New validation error messages
    INVALID_DATAFRAME_STRUCTURE = "Invalid DataFrame structure"
    MISSING_REQUIRED_COLUMNS = "Missing required columns"
    INVALID_DATA_TYPE = "Invalid data type"
    INVALID_YEAR_RANGE = "Year outside valid range"
    INVALID_MONTH_ABBREVIATION = "Invalid month abbreviation"
    INVALID_DURATION_RANGE = "Duration outside reasonable range"
    DIVISION_BY_ZERO_ERROR = "Division by zero in calculation"
    FILE_OPERATION_ERROR = "File operation failed"
    EMPTY_INPUT_ERROR = "Empty or null input provided"
    STRING_TOO_LONG_ERROR = "String exceeds maximum length"
    CONVERSION_ERROR = "Type conversion failed"
    BUSINESS_RULE_VIOLATION = "Business rule validation failed"
    
    # Error context templates
    VALIDATION_ERROR_TEMPLATE = "Validation failed for {function}: {error}"
    PROCESSING_ERROR_TEMPLATE = "Processing error in {function}: {error}"
    DATA_QUALITY_ERROR_TEMPLATE = "Data quality issue in {function}: {error}"

@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration for error handling and monitoring"""
    
    # Logging levels
    ERROR_LEVEL = "ERROR"
    WARNING_LEVEL = "WARNING"
    INFO_LEVEL = "INFO"
    DEBUG_LEVEL = "DEBUG"
    
    # Log message templates
    VALIDATION_LOG_TEMPLATE = "[{level}] {function}: {message}"
    ERROR_COUNT_TEMPLATE = "[{level}] Error count: {count} in {function}"
    PERFORMANCE_LOG_TEMPLATE = "[{level}] Performance: {function} took {duration}ms"


@dataclass(frozen=True)
class PowerBIConfig:
    """Power BI specific formatting configuration"""
    
    # Change direction arrows
    CHANGE_DIRECTION_ARROWS = {
        "UP": "↑",
        "DOWN": "↓"
    }
    
    # Change impact values
    CHANGE_IMPACT_VALUES = {
        "POSITIVE": "True",
        "NEGATIVE": "False"
    }
    
    # Columns of interest for requestor analysis
    REQUESTOR_ANALYSIS_COLUMNS = [
        "Key",
        "Service user",
        "Category",
        "Service",
        "PreferredIssueType",
        "ResolutionDate_yyyy_mm"
    ]

    DECIMAL_PLACES_ROUNDING: Final[int] = 2       # Consistent decimal formatting
    DATE_ORDER_START: Final[int] = 1              # PowerBI 1-based indexing
    PERCENTAGE_MULTIPLIER: Final[int] = 100       # Decimal to percentage conversion
    SLO_CATEGORY_COLUMNS: Final[List[str]]        # Reusable column sets
    SLO_SERVICE_COLUMNS: Final[List[str]]         # View-specific column filters
    POSITIVE_NON_THROUGHPUT = "Positive"          # Non-throughput impact values