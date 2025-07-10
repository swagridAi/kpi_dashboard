
from typing import Any, Dict, List, Tuple
from dateutil.parser import parse
from datetime import datetime
import calendar
from config import FieldNames, ProcessingConfig, BusinessRules, DefaultValues
import business_days_calculation as bdc
import pandas as pd

# Constants for hardcoded values
COLUMNS_TO_DROP = [FieldNames.ISSUE_TYPE, FieldNames.REQUEST_TYPE]


def process_changelog_entries(
    changelog_entries: List[Dict[str, Any]], matched_entries: List[Dict[str, Any]], ticket_values: Dict[str, Any]
) -> None:
    """
    Process changelog entries and calculate status durations.
    """
    previous_change_created = None
    for i, changelog_entry in enumerate(changelog_entries):
        change_created = changelog_entry.get(FieldNames.CHANGE_CREATED, None)
        status_duration = _calculate_status_duration_for_entry(i, change_created, previous_change_created, ticket_values)
        
        changelog_data = _create_changelog_data(changelog_entry, status_duration)
        ticket_values.update(changelog_data)
        matched_entries.append(ticket_values.copy())
        
        previous_change_created = change_created


def merge_mapping_tables(matched_entries, mapping_table):
    # Drop the unnecessary columns from the mapping_table
    columns_to_drop = COLUMNS_TO_DROP  # columns to drop
    mapping_table = mapping_table.drop(columns=columns_to_drop, errors='ignore')
    matched_entries = matched_entries.drop(columns=columns_to_drop, errors='ignore')
    
    # Perform the merge
    return pd.merge(
        matched_entries,
        mapping_table,  # select relevant columns
        how='left',
        left_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.FROM],  # columns in matched_entries
        right_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.STATUS]  # columns in mapping_table
    )


def generate_unmapped_requestors_report(matched_entries):
    """Generate report of unmapped requestors"""
    unmapped_requestors = matched_entries[matched_entries[FieldNames.SERVICE_USER_COLUMN].isnull()][FieldNames.
    PROJECT_INITIATIVE_L1_COLUMN, FieldNames.PROJECT_INITIATIVE_L2_COLUMN]]
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
        if pd.notna(row[request_type_col]) and row[request_type_col].strip() != DefaultValues.EMPTY_STRING and row[request_type_col].strip()
        () != ProcessingConfig.NULL_VALUE_INDICATOR
        else row[issue_type_col],
        axis=1
    )
    return df


def _should_use_fix_version_for_close_date(project_issuetype):
    """Check if project-issuetype should use FixVersion for close date calculation."""
    return project_issuetype in BusinessRules.FIX_VERSION_CLOSE_DATE_PROJECTS


def _calculate_close_date_from_fix_version(fix_version):
    """Calculate close date from FixVersion string."""
    try:
        # Extract year and month abbreviation from FixVersion
        parts = fix_version.split("_")
        month_day = parts[-1]  # Extract the MonthDay part
        month_abbr = month_day[:-2]  # Extract the month abbreviation (e.g., Feb)
        year = int(ProcessingConfig.YEAR_PREFIX + month_day[ProcessingConfig.YEAR_SUFFIX_INDEX:])  # Extract the year (e.g., 25 -> 2025)
        
        # Map month abbreviations to numbers
        month_number = ProcessingConfig.MONTH_ABBREVIATIONS.get(month_abbr)
        
        # Get the last day of the month
        last_day = calendar.monthrange(year, month_number)[1]
        last_date = datetime(year, month_number, last_day, ProcessingConfig.DEFAULT_CLOSE_TIME_HOUR, ProcessingConfig.
        DEFAULT_CLOSE_TIME_MINUTE, ProcessingConfig.DEFAULT_CLOSE_TIME_SECOND)
        
        # Format the date as required
        formatted_date = last_date.strftime(ProcessingConfig.CLOSE_DATE_FORMAT)
        return formatted_date
    except Exception as e:
        print(f"Error processing FixVersion: {e}")
        return None


def update_ticket_values(ticket_values: dict) -> dict:
    """
    Updates the ticket_values dictionary by adding a new key 'close_date'
    based on the value of 'project-issuetype'.
    """
    if _should_use_fix_version_for_close_date(ticket_values.get(FieldNames.PROJECT_ISSUE_TYPE)):
        fix_version = ticket_values.get(FieldNames.FIX_VERSION)
        if fix_version:
            ticket_values[FieldNames.CLOSE_DATE] = _calculate_close_date_from_fix_version(fix_version)
        else:
            ticket_values[FieldNames.CLOSE_DATE] = None  # Default if FixVersion is missing
    else:
        ticket_values[FieldNames.CLOSE_DATE] = ticket_values.get(FieldNames.RESOLUTION_DATE)
    
    return ticket_values


def sort_changelog_entries(changelog_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort changelog entries by changeCreated date.
    """
    try:
        return sorted(changelog_entries, key=lambda x: parse(x.get(FieldNames.CHANGE_CREATED, DefaultValues.EMPTY_STRING)))
    except (ValueError, TypeError):
        return changelog_entries


def calculate_duration(start_date: str, end_date: str) -> Any:
    """
    Calculate the duration in hours between two dates, considering only business days.
    """
    
    try:
        start_dt = parse(start_date, ignoretz=True)
        end_dt = parse(end_date, ignoretz=True)
        
        total_hours = (end_dt - start_dt).total_seconds() / ProcessingConfig.SECONDS_TO_HOURS
        if total_hours < ProcessingConfig.MINIMUM_DURATION:
            total_hours = ProcessingConfig.MINIMUM_DURATION
        return total_hours
    except (ValueError, TypeError):
        print(f"Error parsing dates: {start_date} or {end_date}")
        return ProcessingConfig.INVALID_DATE_VALUE

#return bdc.calculate_duration(start_date,end_date) # Testing out this functionality


def _calculate_status_duration_for_entry(i, change_created, previous_change_created, ticket_values):
    """Calculate status duration for a changelog entry."""
    if i == 0 and change_created:
        return calculate_duration(ticket_values[FieldNames.CREATED], change_created)
    elif previous_change_created and change_created:
        return calculate_duration(previous_change_created, change_created)
    return None


def _create_changelog_data(changelog_entry, status_duration):
    """Create changelog data dictionary from entry and calculated duration."""
    return {
        FieldNames.CHANGE_CREATED: changelog_entry.get(FieldNames.CHANGE_CREATED, None),
        FieldNames.STATUS_DURATION: status_duration,
        FieldNames.AUTHOR: changelog_entry.get(FieldNames.AUTHOR, DefaultValues.UNKNOWN),
        FieldNames.FIELD: changelog_entry.get(FieldNames.FIELD, DefaultValues.UNKNOWN),
        FieldNames.FROM: changelog_entry.get(FieldNames.FROM, DefaultValues.UNKNOWN),
    }