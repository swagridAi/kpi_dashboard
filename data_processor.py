from typing import Any, Dict, List, Tuple
from dateutil.parser import parse
from datetime import datetime
import calendar
from config import FieldNames, ProcessingConfig
import pandas as pd

# Constants for hardcoded values
COLUMNS_TO_DROP = [FieldNames.ISSUE_TYPE, FieldNames.REQUEST_TYPE]
CLOSE_DATE_PROJECT_TYPES = ["DOMMBAU-Data Quality Rule", "DOMMBAU-Consumer Validation"]
CLOSE_DATE_HOUR = 14
CLOSE_DATE_MINUTE = 54
CLOSE_DATE_SECOND = 44
YEAR_PREFIX = "20"
CLOSE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.000+1000"
SECONDS_TO_HOURS = 3600
MINIMUM_DURATION = 0
MONTH_MAPPING = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}
FIELD_NAMES_FROM_KEY = 'FieldNames.FROM'


def merge_mapping_tables(matched_entries, mapping_table):
    # Drop the unnecessary columns from the mapping_table
    columns_to_drop = COLUMNS_TO_DROP  # Columns to drop
    mapping_table = mapping_table.drop(columns=columns_to_drop, errors='ignore')
    matched_entries = matched_entries.drop(columns=columns_to_drop, errors='ignore')
    
    # Perform the merge
    return pd.merge(
        matched_entries,
        mapping_table,  # Select relevant columns
        how='left',
        left_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.FROM],  # Columns in matched_entries
        right_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.STATUS]  # Columns in mapping_table
    )

def generate_unmapped_requestors_report(matched_entries):
    """Generate report of unmapped requestors"""
    unmapped_requestors = matched_entries[matched_entries[FieldNames.SERVICE_USER_COLUMN].isnull()][FieldNames.PROJECT_INITIATIVE_L1_COLUMN, FieldNames.PROJECT_INITIATIVE_L2_COLUMN]
    unmapped_requestors.to_csv('unmapped_requestors.csv', index=False)
    return unmapped_requestors

def merge_requestor_data(matched_entries, requestor_df):
    """Add the requestor column to matched_entries"""
    return matched_entries.merge(
        requestor_df,
        how='left',
        left_on=[FieldNames.PROJECT_INITIATIVE_L1_COLUMN, FieldNames.PROJECT_INITIATIVE_L2_COLUMN],
        right_on=[FieldNames.PROJECT_INITIATIVE_L1_COLUMN, FieldNames.PROJECT_INITIATIVE_L2_COLUMN]
    )

def add_preferred_issue_type(df, request_type_col, issue_type_col, preferred_issue_type_col):
    """
    Add a PREFERRED_ISSUE_TYPE column to the DataFrame based on REQUEST_TYPE and ISSUE_TYPE.
    """
    df[preferred_issue_type_col] = df.apply(
        lambda row: row[request_type_col].strip()
        if pd.notna(row[request_type_col]) and row[request_type_col].strip() != "" and row[request_type_col].strip() != "N/A"
        else row[issue_type_col],
        axis=1
    )
    return df

def _should_use_fix_version_for_close_date(project_issuetype):
    """Check if project-issuetype should use FixVersion for close date calculation."""
    return project_issuetype in CLOSE_DATE_PROJECT_TYPES

def _calculate_close_date_from_fix_version(fix_version):
    """Calculate close date from FixVersion string."""
    try:
        # Extract year and month abbreviation from FixVersion
        parts = fix_version.split(" ")
        month_day = parts[-1]  # Extract the MonthDay part
        month_abbr = month_day[:-2]  # Extract the month abbreviation (e.g., Feb)
        year = int(YEAR_PREFIX + month_day[-2:])  # Extract the year (e.g., 25 -> 2025)
        
        # Map month abbreviations to numbers
        month_number = MONTH_MAPPING.get(month_abbr)
        
        # Get the last day of the month
        last_day = calendar.monthrange(year, month_number)[1]
        last_date = datetime(year, month_number, last_day, CLOSE_DATE_HOUR, CLOSE_DATE_MINUTE, CLOSE_DATE_SECOND)
        
        # Format the date as required
        formatted_date = last_date.strftime(CLOSE_DATE_FORMAT)
        return formatted_date
    except Exception as e:
        print(f"Error processing FixVersion: {e}")
        return None

def update_ticket_values(ticket_values: dict) -> dict:
    """
    Updates the ticket_values dictionary by adding a new key 'close_date'
    based on the value of 'project-issuetype'.
    """
    if _should_use_fix_version_for_close_date(ticket_values.get("project-issuetype")):
        fix_version = ticket_values.get("FixVersion")
        if fix_version:
            ticket_values["close_date"] = _calculate_close_date_from_fix_version(fix_version)
        else:
            ticket_values["close_date"] = None  # Default if FixVersion is missing
    else:
        ticket_values["close_date"] = ticket_values.get("ResolutionDate")
    
    return ticket_values

def sort_changelog_entries(changelog_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort changelog entries by ChangeCreated date.
    """
    try:
        return sorted(changelog_entries, key=lambda x: parse(x.get('ChangeCreated', '')))
    except (ValueError, TypeError):
        return changelog_entries

def calculate_duration(start_date: str, end_date: str) -> Any:
    """
    Calculate the duration in hours between two dates, considering only business days.
    """
    try:
        start_dt = parse(start_date, ignoretz=True)
        end_dt = parse(end_date, ignoretz=True)
        
        total_hours = (end_dt - start_dt).total_seconds() / SECONDS_TO_HOURS
        if total_hours < MINIMUM_DURATION:
            total_hours = MINIMUM_DURATION
        print(f"total_hours is type {type(total_hours)} and value {total_hours}")
        return total_hours
    except (ValueError, TypeError):
        print(f"Error parsing dates: {start_date} or {end_date}")
        return 'Invalid Date'

def _calculate_status_duration_for_entry(i, change_created, previous_change_created, ticket_values):
    """Calculate status duration for a changelog entry."""
    if i == 0 and change_created:
        return calculate_duration(ticket_values["Created"], change_created)
    elif previous_change_created and change_created:
        return calculate_duration(previous_change_created, change_created)
    return None

def _create_changelog_data(changelog_entry, status_duration):
    """Create changelog data dictionary from entry and calculated duration."""
    return {
        'ChangeCreated': changelog_entry.get('ChangeCreated', None),
        'StatusDuration': status_duration,
        'Author': changelog_entry.get('Author', 'Unknown'),
        'Field': changelog_entry.get('Field', 'Unknown'),
        FIELD_NAMES_FROM_KEY: changelog_entry.get(FIELD_NAMES_FROM_KEY, 'Unknown'),
    }

def process_changelog_entries(
    changelog_entries: List[Dict[str, Any]], matched_entries: List[Dict[str, Any]], ticket_values: Dict[str, Any]
) -> None:
    """
    Process changelog entries and calculate status durations.
    """
    previous_change_created = None
    for i, changelog_entry in enumerate(changelog_entries):
        change_created = changelog_entry.get('ChangeCreated', None)
        status_duration = _calculate_status_duration_for_entry(i, change_created, previous_change_created, ticket_values)
        
        changelog_data = _create_changelog_data(changelog_entry, status_duration)
        ticket_values.update(changelog_data)
        matched_entries.append(ticket_values.copy())
        
        previous_change_created = change_created