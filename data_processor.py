from typing import Any, Dict, List, Tuple
from dateutil.parser import parse
from datetime import datetime
import calendar
from config import FieldNames, ProcessingConfig
import pandas as pd


def merge_mapping_tables(matched_entries, mapping_table):
    # Drop the unnecessary columns from the mapping_table
    columns_to_drop = [FieldNames.ISSUE_TYPE, FieldNames.REQUEST_TYPE]  # Columns to drop
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

def update_ticket_values(ticket_values: dict) -> dict:
    """
    Updates the ticket_values dictionary by adding a new key 'close_date'
    based on the value of 'project-issuetype'.
    """
    if ticket_values.get("project-issuetype") in ["DOMMBAU-Data Quality Rule", "DOMMBAU-Consumer Validation"]:
        fix_version = ticket_values.get("FixVersion")
        if fix_version:
            try:
                # Extract year and month abbreviation from FixVersion
                parts = fix_version.split(" ")
                month_day = parts[-1]  # Extract the MonthDay part
                month_abbr = month_day[:-2]  # Extract the month abbreviation (e.g., Feb)
                year = int("20" + month_day[-2:])  # Extract the year (e.g., 25 -> 2025)
                
                # Map month abbreviations to numbers
                month_mapping = {
                    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                }
                month_number = month_mapping.get(month_abbr)
                
                # Get the last day of the month
                last_day = calendar.monthrange(year, month_number)[1]
                last_date = datetime(year, month_number, last_day, 14, 54, 44)
                
                # Format the date as required
                formatted_date = last_date.strftime("%Y-%m-%dT%H:%M:%S.000+1000")
                ticket_values["close_date"] = formatted_date
            except Exception as e:
                ticket_values["close_date"] = None  # Handle errors gracefully
                print(f"Error processing FixVersion: {e}")
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
        
        total_hours = (end_dt - start_dt).total_seconds() / 3600
        if total_hours < 0:
            total_hours = 0
        print(f"total_hours is type {type(total_hours)} and value {total_hours}")
        return total_hours
    except (ValueError, TypeError):
        print(f"Error parsing dates: {start_date} or {end_date}")
        return 'Invalid Date'

def process_changelog_entries(
    changelog_entries: List[Dict[str, Any]], matched_entries: List[Dict[str, Any]], ticket_values: Dict[str, Any]
) -> None:
    """
    Process changelog entries and calculate status durations.
    """
    previous_change_created = None
    for i, changelog_entry in enumerate(changelog_entries):
        change_created = changelog_entry.get('ChangeCreated', None)
        status_duration = None
        
        if i == 0 and change_created:
            status_duration = calculate_duration(ticket_values["Created"], change_created)
        elif previous_change_created and change_created:
            status_duration = calculate_duration(previous_change_created, change_created)
        
        changelog_data = {
            'ChangeCreated': change_created,
            'StatusDuration': status_duration,
            'Author': changelog_entry.get('Author', 'Unknown'),
            'Field': changelog_entry.get('Field', 'Unknown'),
            'FieldNames.FROM': changelog_entry.get('FieldNames.FROM', 'Unknown'),
        }
        ticket_values.update(changelog_data)
        matched_entries.append(ticket_values.copy())
        
        previous_change_created = change_created