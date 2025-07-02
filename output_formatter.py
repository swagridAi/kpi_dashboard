# Third-party imports
from config import FieldNames

# Constants
FIRST_DAY_OF_MONTH = 1
APPROXIMATE_DAYS_PER_MONTH = 30
MONTHS_TO_FILTER_FOR_HISTORICAL_DATA = 7
DATE_FORMAT_YEAR_MONTH = '%Y-%m'
DATE_FORMAT_DISPLAY = '%b-%y'
DECIMAL_PLACES = 2
KPI_VALUE_COLUMN = 'KPI Value'
AVERAGE_KPI_VALUE_COLUMN = 'Average KPI Value'
CHANGE_IN_KPI_VALUE_COLUMN = 'Change in KPI Value'
NUMBER_OF_REQUESTS_COLUMN = 'Number of Requests'
TOTAL_REQUESTS_PER_GROUP_COLUMN = 'total_requests_per_group'
PROPORTION_OF_REQUESTS_COLUMN = 'Proportion of Requests'
SERVICE_COLUMN = 'Service'
CATEGORY_COLUMN = 'Category'
KEY_COLUMN = 'Key'
PREFERRED_LIST_TYPE_COLUMN = 'PreferredListType'

def _filter_last_x_months(df, date_column, months):
    """
    Filters the DataFrame to include only rows from the last x completed months.
    
    :param df: The DataFrame containing the data.
    :param date_column: The name of the column containing the dates.
    :param months: The number of past months to filter.
    :return: A filtered DataFrame with only the last x completed months.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    df = df.copy()
    today = datetime.today()
    first_day_of_current_month = datetime(today.year, today.month, FIRST_DAY_OF_MONTH)
    
    first_day_of_x_months_prior = first_day_of_current_month - timedelta(days=months * APPROXIMATE_DAYS_PER_MONTH)
    
    filtered_df = df[
        (df[date_column] >= first_day_of_x_months_prior) &
        (df[date_column] < first_day_of_current_month)
    ]
    
    return filtered_df

def _filter_most_recent_month(df, date_column):
    """
    Filters the DataFrame to include only rows from the most recent month.
    
    :param df: The DataFrame containing the data.
    :param date_column: The name of the column containing the dates.
    :return: A filtered DataFrame with only the most recent month.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    today = datetime.today()
    most_recent_month_start = datetime(today.year, today.month, FIRST_DAY_OF_MONTH)
    
    previous_month_start = most_recent_month_start - timedelta(days=FIRST_DAY_OF_MONTH)
    previous_month_start = datetime(previous_month_start.year, previous_month_start.month, FIRST_DAY_OF_MONTH)
    
    filtered_df = df[(df[date_column] >= previous_month_start) & (df[date_column] < most_recent_month_start)]
    
    return filtered_df

def _calculate_change(group):
    """Calculate the change from the second most recent month to the most recent month"""
    group = group.sort_values(FieldNames.DATES, ascending=True)
    if len(group) >= 2:
        return group.iloc[-1][KPI_VALUE_COLUMN] - group.iloc[-2][KPI_VALUE_COLUMN]
    return None

def _prepare_dataframe_columns(df):
    """Prepare and format DataFrame columns for Power BI"""
    import pandas as pd
    
    df = df[[SERVICE_COLUMN, FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TARGET, KPI_VALUE_COLUMN, FieldNames.KPI_MET, FieldNames.SERVICE_KPI, 'KPI Type']]
    
    df[FieldNames.RESOLUTION_DATE_YYYY_MM] = pd.to_datetime(df[FieldNames.RESOLUTION_DATE_YYYY_MM], format=DATE_FORMAT_YEAR_MONTH, errors='coerce')
    
    df['ResolutionDate_string'] = pd.to_datetime(df[FieldNames.RESOLUTION_DATE_YYYY_MM], format=DATE_FORMAT_YEAR_MONTH).dt.strftime(DATE_FORMAT_DISPLAY)
    
    df[FieldNames.DATES] = pd.to_datetime(df[FieldNames.RESOLUTION_DATE_YYYY_MM])
    
    return df

def _calculate_monthly_insights(df):
    """Calculate average KPI and change insights"""
    import pandas as pd
    
    average_kpi = df.groupby(FieldNames.SERVICE_KPI)[KPI_VALUE_COLUMN].mean().reset_index()
    average_kpi.rename(columns={KPI_VALUE_COLUMN: AVERAGE_KPI_VALUE_COLUMN}, inplace=True)
    
    change_kpi = (
        df.groupby(FieldNames.SERVICE_KPI)
        .apply(_calculate_change)
        .reset_index(name=CHANGE_IN_KPI_VALUE_COLUMN)
    )
    
    monthly_insights = pd.merge(average_kpi, change_kpi, on=FieldNames.SERVICE_KPI)
    
    return monthly_insights

def convert_data_for_power_bi(df):
    """
    Convert the DataFrame to a format suitable for Power BI.
    This function is a placeholder for any specific transformations needed for Power BI compatibility.
    """
    df = _prepare_dataframe_columns(df)
    
    df = _filter_last_x_months(df, FieldNames.DATES, MONTHS_TO_FILTER_FOR_HISTORICAL_DATA)
    
    monthly_insights = _calculate_monthly_insights(df)
    
    most_recent_month_df = _filter_most_recent_month(df, FieldNames.RESOLUTION_DATE_YYYY_MM)
    
    return df, most_recent_month_df, monthly_insights
    

def _get_columns_of_interest():
    """Define the columns of interest for requestor data cleaning"""
    return [
        KEY_COLUMN,
        FieldNames.SERVICE_USER_COLUMN,
        CATEGORY_COLUMN,
        SERVICE_COLUMN,
        PREFERRED_LIST_TYPE_COLUMN,
        FieldNames.RESOLUTION_DATE_YYYY_MM
    ]

def _calculate_request_proportions(grouped_df):
    """Calculate the proportion of requests within each group"""
    total_requests_per_group = grouped_df.groupby(
        [FieldNames.RESOLUTION_DATE_YYYY_MM, CATEGORY_COLUMN, SERVICE_COLUMN]
    )[NUMBER_OF_REQUESTS_COLUMN].transform("sum")
    
    grouped_df[TOTAL_REQUESTS_PER_GROUP_COLUMN] = total_requests_per_group
    grouped_df[PROPORTION_OF_REQUESTS_COLUMN] = grouped_df[NUMBER_OF_REQUESTS_COLUMN] / total_requests_per_group
    
    return grouped_df

def get_requestor_data_clean(df):
    
    columns_of_interest = _get_columns_of_interest()
    
    df_cleaned = df[columns_of_interest].copy()
    
    df_cleaned = df_cleaned.drop_duplicates()
    
    grouped_df = df_cleaned.groupby(
        [FieldNames.RESOLUTION_DATE_YYYY_MM, CATEGORY_COLUMN, SERVICE_COLUMN, FieldNames.SERVICE_USER_COLUMN]
    ).size().reset_index(name=NUMBER_OF_REQUESTS_COLUMN)
    
    grouped_df = _calculate_request_proportions(grouped_df)
    
    return df_cleaned, grouped_df