"""
Config module for Jira KPI Processing System

This module contains all constants, configuration parameters and mappings used throughout the Jira data processing
And calculation system
"""

from dataclasses import dataclass
from typing import import Dict, List

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
    SERVICE_KPI = "Service-KPI"
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
    SERVICE_USER_COLUMN = "Service User"

@dataclass(frozen=True)
class JiraFields:
    """Jira-specific field mappings"""

@dataclass(frozen=True)
class ProcessingConfig:
    """Processing and calculation configuration"""
    
    # Unit conversion factors
    CONVERSION_FACTORS = {
        'hours': 1,
        'days': 1 / 24,
        'weeks': 1 / 168
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