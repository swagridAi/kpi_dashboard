# Standard Imports
from typing import Any, Dict, List

# Third-party imports
import pandas as pd
from config import FieldNames, ProcessingConfig

def standard_throughput_calculation(ticket_data_df, kpi_targets_df):
    """
    Group the filtered entries by 'Name' and 'ResolutionDate', calculate the total number of items in each group,
    and compute the average 'TicketDuration' for each group using pandas.
    """
    
    def calculate_preferred_throughput(row):
        if row['Unit'] == 'hours':
            return row['TotalHours']
        elif row['Unit'] == 'tickets':
            return row['TotalUniqueTicketsClosed']
        else:
            return None  # Handle cases where Unit is missing or invalid
    
    # Step 1: Drop duplicates to ensure unique rows
    ticket_data_df = ticket_data_df[['Key', 'Service', 'TicketDuration', FieldNames.RESOLUTION_DATE_YYYY_MM, 'TimespentSeconds']].drop_duplicates()
    
    # Step 2: Group by 'Service' and 'ResolutionDate', and calculate count, average duration, and total hours
    grouped = (
        ticket_data_df.groupby(['Service', FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg(
            TotalUniqueTicketsClosed=('Key', 'count'),
            AverageDuration=('TicketDuration', 'mean'),
            TotalHours=('TimespentSeconds', lambda x: x.sum() / 3600)  # Convert seconds to hours
        )
        .reset_index()
    )
    
    # Step 3: Filter kpi_targets_df so that the column 'KPI' only includes these values Lead Cycle Response
    kpi_targets_df = kpi_targets_df[kpi_targets_df['KPI'].isin(['Throughput'])]
    
    # Step 4 : Drop the 'KPI' column in kpi_targets_df
    kpi_targets_df = kpi_targets_df.drop(columns=['KPI'])
    
    # Step 5: Merge with KPI targets to add target values
    result = pd.merge(grouped, kpi_targets_df, how='left', on=['Service'])
    
    # Step 6: Add a new column KPI Value in preferred units
    # Apply the function to calculate KPI Value
    result['KPI Value'] = result.apply(calculate_preferred_throughput, axis=1)
    
    # Step 7: Add a new column KPI met where if the value of KPI Value is less than or equal to Target, then it is True, else False
    result[FieldNames.KPI_MET] = result.apply(lambda row: row['KPI Value'] >= row[FieldNames.TARGET] if pd.notna(row[FieldNames.TARGET]) else False, axis=1)
    
    # Step 11: Add a new column called SERVICE_KPI which is a combination of 'Service' and TIME_TYPE_COLUMN
    result[FieldNames.SERVICE_KPI] = result['Service'] + '_Throughput'
    
    result['KPI Type'] = 'Throughput'
    
    return result

def get_time_status_per_month(ticket_data_df, kpi_targets_df):
    """
    Group by Service, ResolutionDate_yyyy_mm, and Time Type, calculate the average StatusDuration,
    and add an inclusive duration for each group. The result is sorted by Service, ResolutionDate_yyyy_mm,
    and Time Type.
    
    :param ticket_data_df: DataFrame containing the original data.
    :return: Grouped, aggregated, and sorted data as a DataFrame.
    """
    
    # Step 0: Rename TIME_TYPE_COLUMN value "Lead" to "Lead_sub_status"
    ticket_data_df[FieldNames.TIME_TYPE_COLUMN] = ticket_data_df[FieldNames.TIME_TYPE_COLUMN].replace('Lead', 'Lead_sub_status')
    
    # Step 1: Calculate unique keys at the level of Service and ResolutionDate_yyyy_mm
    unique_keys_per_service = (
        ticket_data_df.groupby(['Service', FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg(UniqueKeys=('Key', 'nunique'))
        .reset_index()
    )
    
    # Step 2: Group by Service, ResolutionDate_yyyy_mm, and Time Type, and calculate TOTAL_STATUS_DURATION_COLUMN
    grouped = (
        ticket_data_df.groupby(['Service', FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TIME_TYPE_COLUMN])
        .agg(**{FieldNames.TOTAL_STATUS_DURATION_COLUMN: ('StatusDuration', 'sum')})  # Aggregate sum of StatusDuration
        .reset_index()
    )
    
    # Step 3: Merge the unique key count into the grouped DataFrame
    grouped = grouped.merge(unique_keys_per_service, on=['Service', FieldNames.RESOLUTION_DATE_YYYY_MM], how='left')
    
    # Step 4: Calculate the average StatusDuration using the consistent unique key count
    grouped['AverageStatusDuration'] = grouped[FieldNames.TOTAL_STATUS_DURATION_COLUMN] / grouped['UniqueKeys']
    
    # Step 5: Calculate the inclusive duration for each group
    inclusive_durations = (
        grouped.groupby(['Service', FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg(**{FieldNames.INCLUSIVE_DURATION_COLUMN: ('AverageStatusDuration', 'sum')})
        .reset_index()
    )
    
    grouped = grouped.drop(columns=['UniqueKeys', FieldNames.TOTAL_STATUS_DURATION_COLUMN])
    
    # Step 6: Add the inclusive duration as new rows with TIME_TYPE_COLUMN set to 'Lead'
    inclusive_durations[FieldNames.TIME_TYPE_COLUMN] = 'Lead'
    inclusive_durations.rename(columns={FieldNames.INCLUSIVE_DURATION_COLUMN: 'AverageStatusDuration'}, inplace=True)
    
    # Step 7: Append the inclusive durations to the grouped DataFrame
    result = pd.concat([grouped, inclusive_durations], ignore_index=True)
    
    # Step 8: Sort the result by 'Service', FieldNames.RESOLUTION_DATE_YYYY_MM, and TIME_TYPE_COLUMN
    result = result.sort_values(by=['Service', FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TIME_TYPE_COLUMN]).reset_index(drop=True)
    
    # Get total resolution time for tickets
    # Step 1: I want to get the total resolution time it takes for a ticket
    ticket_data_df = ticket_data_df[['Key', 'Service', 'TicketDuration', FieldNames.RESOLUTION_DATE_YYYY_MM]].drop_duplicates()
    
    # Step 2: Group by 'Service' and 'ResolutionDate', and calculate count, average duration, and total hours
    ticket_resolution_times = (
        ticket_data_df.groupby(['Service', FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg(
            AverageDuration=('TicketDuration', 'mean')
        )
        .reset_index()
    )
    
    # Rename the 'AverageDuration' column to 'AverageStatusDuration'
    ticket_resolution_times = ticket_resolution_times.rename(columns={'AverageDuration': 'AverageStatusDuration'})
    
    # Add a new column called Time Type which is filled with the value 'Resolution'
    ticket_resolution_times[FieldNames.TIME_TYPE_COLUMN] = 'Resolution'
    
    # Reorder the columns to have the order:
    ticket_resolution_times = ticket_resolution_times[
        ['Service', FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TIME_TYPE_COLUMN, 'AverageStatusDuration']
    ]
    
    # Ensure both result and ticket_resolution_times have the same columns
    ticket_resolution_times = ticket_resolution_times[result.columns]
    
    # Get only KPIs that we are interested in
    # Step 1: Filter kpi_targets_df so that the column 'KPI' only includes these values Lead Cycle Response
    kpi_targets_df = kpi_targets_df[kpi_targets_df['KPI'].isin(['Lead', 'Cycle', 'Response', 'Resolution'])]
    
    # Step 2 : Rename 'KPI' to TIME_TYPE_COLUMN in kpi_targets_df
    kpi_targets_df.rename(columns={'KPI': FieldNames.TIME_TYPE_COLUMN}, inplace=True)
    
    # Merge the ticket resolution times with the result DataFrame
    # Step 1: Add ticket_resolution_times to result
    result = pd.concat([result, ticket_resolution_times], ignore_index=True)
    
    # Step 2: Merge with KPI targets to add target values
    result = pd.merge(result, kpi_targets_df, how='left', on=['Service', FieldNames.TIME_TYPE_COLUMN])
    
    # Step 3: Add a new column KPI Value in preferred units
    result['KPI Value'] = result.apply(
        lambda row: row['AverageStatusDuration'] * ProcessingConfig.CONVERSION_FACTORS[row['Unit']]
        if pd.notna(row['AverageStatusDuration']) and row['Unit'] in ProcessingConfig.CONVERSION_FACTORS
        else None,
        axis=1
    )
    
    # Step 4: Add a new column KPI met where if the value of KPI Value is less than or equal to Target, then it is True, else False
    result[FieldNames.KPI_MET] = result.apply(lambda row: row['KPI Value'] <= row[FieldNames.TARGET] if pd.notna(row[FieldNames.TARGET]) else False, axis=1)
    
    # Step 5: Add a new column called SERVICE_KPI which is a combination of 'Service' and TIME_TYPE_COLUMN
    result[FieldNames.SERVICE_KPI] = result['Service'] + '_' + result[FieldNames.TIME_TYPE_COLUMN]
    
    # Step 6: Add a new column called 'KPI Type' which is the same as TIME_TYPE_COLUMN
    result['KPI Type'] = result[FieldNames.TIME_TYPE_COLUMN]
    
    return result

def get_issue_rate(df: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate the percentage of tickets with issues grouped by Service and ResolutionDate using pandas.
    
    :param matched_entries: List of dictionaries containing the original data.
    :return: Grouped and calculated data as a list of dictionaries.
    """
    
    # Step 3: Group by 'Service' and 'ResolutionDate'
    grouped = (
        df.groupby(['Service', FieldNames.RESOLUTION_DATE_YYYY_MM])
        .agg(
            total_tickets=('ResolutionDate', 'count'),
            issues_count=('HasIssues', 'sum')  # Sum up the boolean values for issues count
        )
        .reset_index()
    )
    
    # Step 4: Calculate the percentage of tickets with issues
    grouped['issues_percentage'] = (grouped['issues_count'] / grouped['total_tickets']) * 100
    
    return grouped