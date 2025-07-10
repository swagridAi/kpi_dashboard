# Standard imports
from typing import Any, Dict, List

# Third-party imports
import pandas as pd
from config import FieldNames, ProcessingConfig, BusinessRules

# Constants
UNIT_HOURS = 'hours'
UNIT_TICKETS = 'tickets'
KPI_TYPE_THROUGHPUT = 'Throughput'
KPI_TYPE_LEAD = 'Lead'
KPI_TYPE_RESOLUTION = 'Resolution'
KPI_SEPARATOR = ' - '
LEAD_SUB_STATUS = 'Lead_sub_status'

UNIT_FIELD = 'Unit'
TOTAL_HOURS_FIELD = 'TotalHours'
TOTAL_UNIQUE_TICKETS_CLOSED_FIELD = 'TotalUniqueTicketsClosed'
KPI_FIELD = 'KPI'
AVERAGE_STATUS_DURATION = 'AverageStatusDuration'
KPI_VALUE = 'KPI Value'
AVERAGE_DURATION = 'AverageDuration'
UNIQUE_KEYS = 'UniqueKeys'
ISSUES_PERCENTAGE = 'issues_percentage'
ISSUES_COUNT = 'issues_count'
TOTAL_TICKETS = 'total_tickets'
KPI_FILTER_VALUES = ['Lead Cycle Response']


def get_issue_rate(df: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate the percentage of tickets with issues grouped by Service and ResolutionDate using pandas.
    
    :param matched_entries: List of dictionaries containing the original data.
    :return: Grouped and calculated data as a list of dictionaries.
    """
    
    # Step 3: Group by FieldNames.SERVICE and RESOLUTION_DATE
    grouped = (
        df.groupby([FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg({
            'total_tickets'=(FieldNames.RESOLUTION_DATE, 'count'),
            'issues_count'=(FieldNames.HAS_ISSUES, 'sum')  # Sum the boolean values for issues count
        })
        .reset_index()
    )
    
    # Step 4: Calculate the percentage of tickets with issues
    grouped[ISSUES_PERCENTAGE] = (grouped[ISSUES_COUNT] / grouped[TOTAL_TICKETS]) * 100
    
    return grouped


def _calculate_preferred_throughput(row):
    if row[UNIT_FIELD] == UNIT_HOURS:
        return row[TOTAL_HOURS_FIELD]
    elif row[UNIT_FIELD] == UNIT_TICKETS:
        return row[TOTAL_UNIQUE_TICKETS_CLOSED_FIELD]
    else:
        return None  # Handle cases where Unit is missing or invalid


def _prepare_throughput_data(ticket_data_df, kpi_targets_df):
    # Step 1: Drop duplicates to ensure unique rows
    ticket_data_df = ticket_data_df[[FieldNames.KEY, FieldNames.SERVICE, FieldNames.TICKET_DURATION, FieldNames.RESOLUTION_DATE_YYYY_MM,
                                   FieldNames.TIME_SPENT_SECONDS]].drop_duplicates()
    
    # Step 2: Group by FieldNames.SERVICE and FieldNames.KPI_TYPE, and calculate count, average duration, and total hours
    grouped = (
        ticket_data_df.groupby([FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg({
            'TotalUniqueTicketsClosed'=(FieldNames.KEY, 'count'),
            'AverageDuration'=(FieldNames.TICKET_DURATION, 'mean'),
            'TotalHours'=(FieldNames.TIME_SPENT_SECONDS, lambda x: x.sum() / ProcessingConfig.SECONDS_TO_HOURS)  # Convert seconds to hours
        })
        .reset_index()
    )
    
    # Step 3: Filter kpi_targets_df so that the column KPI_FIELD only includes these values Lead Cycle Response
    kpi_targets_df = kpi_targets_df[kpi_targets_df[KPI_FIELD].isin(BusinessRules.THROUGHPUT_KPI_TYPES)]
    
    # Step 4: Drop the KPI_FIELD column in kpi_targets_df
    kpi_targets_df = kpi_targets_df.drop(columns=[KPI_FIELD])
    
    return grouped, kpi_targets_df


def _add_throughput_columns(result):
    
    #TODO Add throughput
    # Step 6: Add a new column KPI Value in preferred units
    # Apply the function to calculate KPI Value
    result[KPI_VALUE] = result.apply(_calculate_preferred_throughput, axis=1)
    
    # Step 7: Add a new column KPI met where if the value of KPI Value is less than or equal to Target, then it is True, else False
    result[FieldNames.KPI_MET] = result.apply(lambda row: row[KPI_VALUE] >= row[FieldNames.TARGET] if pd.notna(row[FieldNames.TARGET])
                                            else False, axis=1)
    
    # Step 11: Add a new column called SERVICE_KPI which is a combination of FieldNames.SERVICE and TIME_TYPE_COLUMN
    result[FieldNames.SERVICE_KPI] = result[FieldNames.SERVICE] + KPI_SEPARATOR + KPI_TYPE_THROUGHPUT
    
    result[FieldNames.KPI_TYPE] = KPI_TYPE_THROUGHPUT
    
    return result


def standard_throughput_calculation(ticket_data_df, kpi_targets_df):
    """
    Group the filtered entries by 'Name' and RESOLUTION_DATE, calculate the total number of items in each group,
    and compute the average FieldNames.TICKET_DURATION for each group using pandas.
    """
    
    grouped, kpi_targets_df = _prepare_throughput_data(ticket_data_df, kpi_targets_df)
    
    # Step 5: Merge with KPI targets to add target values
    result = pd.merge(grouped, kpi_targets_df, how='left', on=[FieldNames.SERVICE])
    
    result = _add_throughput_columns(result)
    
    return result


def _prepare_time_status_data(ticket_data_df):
    # Step 0: Rename TIME_TYPE_COLUMN value "Lead" to "Lead_sub_status"
    ticket_data_df[FieldNames.TIME_TYPE_COLUMN] = ticket_data_df[FieldNames.TIME_TYPE_COLUMN].replace(KPI_TYPE_LEAD, LEAD_SUB_STATUS)
    
    # Step 1: Calculate unique keys at the level of Service and ResolutionDate_yyyy_mm
    unique_keys_per_service = (
        ticket_data_df.groupby([FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg({'UniqueKeys'=(FieldNames.KEY, 'nunique')})
        .reset_index()
    )
    
    # Step 2: Group by Service, ResolutionDate_yyyy_mm, and Time Type, and calculate TOTAL_STATUS_DURATION_COLUMN
    grouped = (
        ticket_data_df.groupby([FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TIME_TYPE_COLUMN])
        .agg({**{FieldNames.TOTAL_STATUS_DURATION_COLUMN: (FieldNames.STATUS_DURATION, 'sum')}})  # Aggregate sum of StatusDuration
        .reset_index()
    )
    
    # Step 3: Merge the unique key count into the grouped DataFrame
    grouped = grouped.merge(unique_keys_per_service, on=[FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM], how='left')
    
    # Step 4: Calculate the average StatusDuration using the consistent unique key count
    grouped[AVERAGE_STATUS_DURATION] = grouped[FieldNames.TOTAL_STATUS_DURATION_COLUMN] / grouped[UNIQUE_KEYS]
    
    return grouped


def _calculate_inclusive_durations(grouped):
    # Step 5: Calculate the inclusive duration for each group
    inclusive_durations = (
        grouped.groupby([FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg({**{FieldNames.INCLUSIVE_DURATION_COLUMN: (AVERAGE_STATUS_DURATION, 'sum')}})
        .reset_index()
    )
    
    grouped = grouped.drop(columns=[UNIQUE_KEYS, FieldNames.TOTAL_STATUS_DURATION_COLUMN])
    
    # Step 6: Add the inclusive duration as new rows with TIME_TYPE_COLUMN set to 'Lead'
    inclusive_durations[FieldNames.TIME_TYPE_COLUMN] = KPI_TYPE_LEAD
    inclusive_durations.rename(columns={FieldNames.INCLUSIVE_DURATION_COLUMN: AVERAGE_STATUS_DURATION}, inplace=True)
    
    # Step 7: Append the inclusive durations to the grouped DataFrame
    result = pd.concat([grouped, inclusive_durations], ignore_index=True)
    
    # Step 8: Sort the result by FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM, and TIME_TYPE_COLUMN
    result = result.sort_values(by=[FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TIME_TYPE_COLUMN]).reset_index(
        drop=True)
    
    return result


def _process_ticket_resolution_times(ticket_data_df, result):
    
    # Get total resolution time for tickets
    # Step 1: I want to get the total resolution time it takes for a ticket
    ticket_data_df = ticket_data_df[[FieldNames.KEY, FieldNames.SERVICE, FieldNames.TICKET_DURATION, FieldNames.RESOLUTION_DATE_YYYY_MM]].drop_duplicates()
    
    # Step 2: Group by FieldNames.SERVICE and RESOLUTION_DATE, and calculate count, average duration, and total hours
    ticket_resolution_times = (
        ticket_data_df.groupby([FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg({
            'AverageDuration'=(FieldNames.TICKET_DURATION, 'mean')  })
        .reset_index()
    )
    
    # Rename the AVERAGE_DURATION column to AVERAGE_STATUS_DURATION
    ticket_resolution_times = ticket_resolution_times.rename(columns={'AverageDuration': AVERAGE_STATUS_DURATION})
    
    # Add a new column called Time Type which is filled with the value 'Resolution'
    ticket_resolution_times[FieldNames.TIME_TYPE_COLUMN] = KPI_TYPE_RESOLUTION
    
    # Reorder the columns to have the order:
    ticket_resolution_times = ticket_resolution_times[
        [FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TIME_TYPE_COLUMN, AVERAGE_STATUS_DURATION]
    ]
    
    # Ensure both result and ticket_resolution_times have the same columns
    ticket_resolution_times = ticket_resolution_times[result.columns]
    
    return ticket_resolution_times


def _merge_kpi_targets_and_finalize(result, ticket_resolution_times, kpi_targets_df):
    # Get only KPIs that we are interested in
    # Step 1: Filter kpi_targets_df so that the column KPI_FIELD only includes these values Lead Cycle Response Resolution
    kpi_targets_df = kpi_targets_df[kpi_targets_df[KPI_FIELD].isin(BusinessRules.TIME_BASED_KPI_TYPES)]
    
    # Step 2: Rename KPI_FIELD to TIME_TYPE_COLUMN in kpi_targets_df
    kpi_targets_df.rename(columns={KPI_FIELD: FieldNames.TIME_TYPE_COLUMN}, inplace=True)
    
    # Merge the ticket resolution times with the result DataFrame
    # Step 1: Add ticket_resolution_times to result
    result = pd.concat([result, ticket_resolution_times], ignore_index=True)
    
    # Step 2: Merge with KPI targets to add target values
    result = pd.merge(result, kpi_targets_df, how='left', on=[FieldNames.SERVICE, FieldNames.TIME_TYPE_COLUMN])
    
    # Step 3: Add a new column KPI Value in preferred units
    result[KPI_VALUE] = result.apply(
        lambda row: row[AVERAGE_STATUS_DURATION] * ProcessingConfig.CONVERSION_FACTORS[row[UNIT_FIELD]]
        if pd.notna(row[AVERAGE_STATUS_DURATION]) and row[UNIT_FIELD] in ProcessingConfig.CONVERSION_FACTORS
        else None,
        axis=1
    )
    
    # Step 4: Add a new column KPI met where if the value of KPI Value is less than or equal to Target, then it is True, else False
    result[FieldNames.KPI_MET] = result.apply(lambda row: row[KPI_VALUE] <= row[FieldNames.TARGET] if pd.notna(row[FieldNames.TARGET])
                                            else False, axis=1)
    
    # Step 5: Add a new column called SERVICE_KPI which is a combination of FieldNames.SERVICE and TIME_TYPE_COLUMN
    result[FieldNames.SERVICE_KPI] = result[FieldNames.SERVICE] + KPI_SEPARATOR + result[FieldNames.TIME_TYPE_COLUMN]
    
    # Step 6: Add a new column called FieldNames.KPI_TYPE which is the same as TIME_TYPE_COLUMN
    result[FieldNames.KPI_TYPE] = result[FieldNames.TIME_TYPE_COLUMN]
    
    return result


def get_time_status_per_month(ticket_data_df, kpi_targets_df):
    """
    Group by Service, ResolutionDate_yyyy_mm, and Time Type, calculate the average StatusDuration,
    and add an inclusive duration for each group. The result is sorted by Service, ResolutionDate_yyyy_mm,
    and Time Type.
    
    :param ticket_data_df: DataFrame containing the original data.
    :return: Grouped, aggregated, and sorted data as a DataFrame.
    """
    grouped = _prepare_time_status_data(ticket_data_df)
    result = _calculate_inclusive_durations(grouped)
    ticket_resolution_times = _process_ticket_resolution_times(ticket_data_df, result)
    result = _merge_kpi_targets_and_finalize(result, ticket_resolution_times, kpi_targets_df)
    
    return result