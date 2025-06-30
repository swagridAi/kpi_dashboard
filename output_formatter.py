# Third-party imports
from config import FieldNames

def convert_data_for_power_bi(df):
    """
    Convert the DataFrame to a format suitable for Power BI.
    This function is a placeholder for any specific transformations needed for Power BI compatibility.
    """
    import pandas as pd
    from datetime import import datetime, timedelta
    
    # Assuming DATES is a column in your DataFrame containing datetime objects
    def filter_last_x_months(df, date_column, months):
        """
        Filters the DataFrame to include only rows from the last x completed months.
        
        :param df: The DataFrame containing the data.
        :param date_column: The name of the column containing the dates.
        :param months: The number of past months to filter.
        :return: A filtered DataFrame with only the last x completed months.
        """
        # Get the current date and calculate the first day of the current month
        df = df.copy()
        today = datetime.today()
        first_day_of_current_month = datetime(today.year, today.month, 1)
        
        # Calculate the first day of the month x months prior
        first_day_of_x_months_prior = first_day_of_current_month - timedelta(days=months * 30)
        
        # Filter the DataFrame to include only rows within the last x completed months
        filtered_df = df[
            (df[date_column] >= first_day_of_x_months_prior) &
            (df[date_column] < first_day_of_current_month)
        ]
        
        return filtered_df
    
    # Assuming 'df' is your DataFrame and 'date_column' contains the dates
    def filter_most_recent_month(df, date_column):
        """
        Filters the DataFrame to include only rows from the most recent month.
        
        :param df: The DataFrame containing the data.
        :param date_column: The name of the column containing the dates.
        :return: A filtered DataFrame with only the most recent month.
        """
        # Ensure the date column is in datetime format
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Get the current date and calculate the start of the most recent month
        today = datetime.today()
        most_recent_month_start = datetime(today.year, today.month, 1)
        
        # Calculate the start of the month before the most recent month
        previous_month_start = most_recent_month_start - timedelta(days=1)
        previous_month_start = datetime(previous_month_start.year, previous_month_start.month, 1)
        
        # Filter the DataFrame for rows in the month before the most recent month
        filtered_df = df[(df[date_column] >= previous_month_start) & (df[date_column] < most_recent_month_start)]
        
        return filtered_df
    
    # I want df to only have these columns: Service, ResolutionDate_yyyy_mm, KPI Value, KPI Met, Service-KPI
    df = df[['Service', FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.TARGET, 'KPI Value', FieldNames.KPI_MET, FieldNames.SERVICE_KPI, 'KPI Type']]
    
    # I want to convert the RESOLUTION_DATE_YYYY_MM column to a datetime format
    df[FieldNames.RESOLUTION_DATE_YYYY_MM] = pd.to_datetime(df[FieldNames.RESOLUTION_DATE_YYYY_MM], format='%Y-%m', errors='coerce')
    
    # Convert the ResolutionDate_yyyy_mm column to the format "Jan'24"
    df['ResolutionDate_string'] = pd.to_datetime(df[FieldNames.RESOLUTION_DATE_YYYY_MM], format='%Y-%m').dt.strftime('%b\'%y')
    
    # Assuming 'df' is your DataFrame and DATES is the column with datetime objects
    df[FieldNames.DATES] = pd.to_datetime(df[FieldNames.RESOLUTION_DATE_YYYY_MM])  # Ensure the dates column is in datetime format
    # Filter the DataFrame to include only the last 7 months meaning it will find the last 6 months of completed data
    df = filter_last_x_months(df, FieldNames.DATES, 7)
    
    # Calculate average KPI Value grouped by Service-KPI
    average_kpi = df.groupby(FieldNames.SERVICE_KPI)['KPI Value'].mean().reset_index()
    average_kpi.rename(columns={'KPI Value': 'Average KPI Value'}, inplace=True)
    
    # Calculate the change from the second most recent month to the most recent month
    def calculate_change(group):
        group = group.sort_values(FieldNames.DATES, ascending=True)
        if len(group) >= 2:
            return group.iloc[-1]['KPI Value'] - group.iloc[-2]['KPI Value']
        return None
    
    change_kpi = (
        df.groupby(FieldNames.SERVICE_KPI)
        .apply(calculate_change)
        .reset_index(name='Change in KPI Value')
    )
    
    # Merge the results
    monthly_insights = pd.merge(average_kpi, change_kpi, on=FieldNames.SERVICE_KPI)
    
    # I want to get a dataframe which is filtered for only the most recent month
    most_recent_month_df = filter_most_recent_month(df, FieldNames.RESOLUTION_DATE_YYYY_MM)
    
    # Return the transformed DataFrame
    return df, most_recent_month_df, monthly_insights


def get_requestor_data_clean(df):

    # Define the columns of interest
    columns_of_interest = [
        'Key',
        FieldNames.SERVICE_USER_COLUMN,
        'Category',
        'Service',
        'PreferredIssueType',
        FieldNames.RESOLUTION_DATE_YYYY_MM
    ]
    
    # Ensure the dataframe only contains the columns of interest
    df_cleaned = df[columns_of_interest].copy()
    
    # Drop duplicate rows based on all columns
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Group by month, category, service, and project initiative
    grouped_df = df_cleaned.groupby(
        ['ResolutionDate_yyyy_mm', 'Category', 'Service', FieldNames.SERVICE_USER_COLUMN]
    ).size().reset_index(name='Number of requests')
    
    # Calculate the total requests per month, category, and service group
    total_requests_per_group = grouped_df.groupby(
        ['ResolutionDate_yyyy_mm', 'Category', 'Service']
    )['Number of requests'].transform('sum')
    
    # Calculate the proportion of requests within each group
    grouped_df['total_requests_per_group'] = total_requests_per_group
    grouped_df['Proportion of requests'] = grouped_df['Number of requests'] / total_requests_per_group
    
    return df_cleaned, grouped_df
