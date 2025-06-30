"""
Config module for Jira KPI Processing System

This module contains all constants, configuration parameters and mappings used throughout the Jira data processing
And calculation system
"""

from dataclasses import dataclass
from typing import Dict, List

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

@dataclass(frozen=True)
class JiraFields:
    """Jira-specific field mappings"""

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
    
    # Analysis settings
    HISTORICAL_MONTHS = 7


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