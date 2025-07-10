import pandas as pd
from output_formatter import convert_data_for_power_bi
from config import FieldNames, PowerBiConfig, OutputConfig, DefaultValues
from collections import OrderedDict

# Constants for hardcoded values
MATRIX_STAT_TARGET = "Target (Red line)"
MATRIX_STAT_CHANGE = "Change (Month on Month)"
MATRIX_STAT_AVERAGE = "6-Month Average"
LINE_ORDER_TARGET = 1
LINE_ORDER_CHANGE = 2
LINE_ORDER_AVERAGE = 3
THROUGHPUT_KPI_TYPE = "Throughput"
KPI = "KPI"


def create_kpi_result_views(standard_throughput_results, time_status_per_month):
    
    # Get data into a power BI format
    throughput_results_kpi, recent_throughput_results_kpi, throughput_insights = convert_data_for_power_bi(standard_throughput_results)
    time_results_kpi, recent_time_results_kpi, recent_time_insights = convert_data_for_power_bi(time_status_per_month)
    
    # Concatenate the results for Power BI
    all_results_kpi = pd.concat([throughput_results_kpi, time_results_kpi], ignore_index=True)
    recent_results_kpi = pd.concat([recent_throughput_results_kpi, recent_time_results_kpi], ignore_index=True)
    kpi_insights = pd.concat([throughput_insights, recent_time_insights], ignore_index=True)
    
    return all_results_kpi, recent_results_kpi, kpi_insights

def enhance_recent_results(recent_results_kpi, category_definitions_df, kpi_definitions_df):
    """Add definitions and clean recent results"""
    
    # Drop rows of all_results_kpi which don't have a value for Target as there is no kpi assigned to it
    recent_results_kpi = recent_results_kpi.dropna(subset=[FieldNames.TARGET])
    
    # Left join definitions to the recent results on SERVICE and Rename the 'Definition' column to CATEGORY_DEFINITION in
    recent_results_kpi
    recent_results_kpi = pd.merge(recent_results_kpi, category_definitions_df, on=FieldNames.SERVICE, how="left")
    recent_results_kpi = recent_results_kpi.rename(columns={FieldNames.DEFINITION: FieldNames.CATEGORY_DEFINITION})
    
    # Left join kpi definitions on recent results on FieldNames.KPI_TYPE and Rename the 'Definition' column to KPI_DEFINITION in
    recent_results_kpi
    recent_results_kpi = pd.merge(recent_results_kpi, kpi_definitions_df, on=FieldNames.KPI_TYPE, how="left")
    recent_results_kpi = recent_results_kpi.rename(columns={FieldNames.DEFINITION: FieldNames.KPI_DEFINITION})
    
    return recent_results_kpi


def process_all_results_for_power(all_results_kpi):
    """Process all results for Power BI consumption"""
    
    # Drop rows of all_results_kpi which don't have a value for Target as there is no kpi assigned to it
    all_results_kpi = all_results_kpi.dropna(subset=[FieldNames.TARGET])
    # Sort the DataFrame by the Date column
    all_results_kpi = all_results_kpi.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    all_results_kpi[FieldNames.DATE_ORDER] = range(1, len(all_results_kpi) + 1)
    
    return all_results_kpi


def create_slo_category_view(all_results_kpi, services_categories):
    
    # Turn all_results_kpi into a format for Power BI SLO graph
    slo_met_percent_category = all_results_kpi[[FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.KPI_MET, FieldNames.
SERVICE_KPI, FieldNames.RESOLUTION_DATE_STRING, FieldNames.DATES]]
    
    # Left join services_categories on slo_met_percent using FieldNames.SERVICE as the key
    slo_met_percent_category = pd.merge(slo_met_percent_category, services_categories, on=FieldNames.SERVICE, how="left")
    
    # Drop the "Service-KPI" and SERVICE columns from slo_met_percent
    slo_met_percent_category = slo_met_percent_category.drop(columns=[FieldNames.SERVICE_KPI, FieldNames.SERVICE], errors='ignore')
    
    # Group by Service-KPI and ResolutionDate_yyyy_mm
    slo_met_percent_category = (
        slo_met_percent_category.groupby([FieldNames.CATEGORY, FieldNames.DATES])
        .agg({
            KPI_Met_Percentage=pd.NamedAgg(
                column=FieldNames.KPI_MET,
                aggfunc=lambda x: (x.sum() / len(x)) * 100 if x.dtype == bool else None
            )
        })
        .reset_index()
    )
    
    # Add DateOrder column
    slo_met_percent_category = slo_met_percent_category.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    slo_met_percent_category[FieldNames.DATE_ORDER] = range(1, len(slo_met_percent_category) + 1)
    
    return slo_met_percent_category


def create_slo_service_view(all_results_kpi):
    slo_met_percent_service = all_results_kpi[[FieldNames.SERVICE, FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.KPI_MET, FieldNames.
RESOLUTION_DATE_STRING, FieldNames.DATES]]
    
    # Group by Service-KPI and ResolutionDate_yyyy_mm
    slo_met_percent_service = (
        slo_met_percent_service.groupby([FieldNames.SERVICE, FieldNames.DATES])
        .agg({
            KPI_Met_Percentage=pd.NamedAgg(
                column=FieldNames.KPI_MET,
                aggfunc=lambda x: (x.sum() / len(x)) * 100 if x.dtype == bool else None
            )
        })
        .reset_index()
    )
    
    # Add DateOrder column
    slo_met_percent_service = slo_met_percent_service.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    slo_met_percent_service[FieldNames.DATE_ORDER] = range(1, len(slo_met_percent_service) + 1)
    return slo_met_percent_service

def calculate_change_impact(row):
    
    if row[KPI] == THROUGHPUT_KPI_TYPE:
        return PowerBiConfig.CHANGE_IMPACT_VALUES[PowerBiConfig.POSITIVE] if row[FieldNames.CHANGE_IN_KPI_VALUE] and row[FieldNames.
CHANGE_IN_KPI_VALUE] > 0 else PowerBiConfig.CHANGE_IMPACT_VALUES[PowerBiConfig.NEGATIVE]
    else:
        return PowerBiConfig.CHANGE_IMPACT_VALUES[PowerBiConfig.NEGATIVE] if row[FieldNames.CHANGE_IN_KPI_VALUE] and row[FieldNames.
CHANGE_IN_KPI_VALUE] > 0 else PowerBiConfig.POSITIVE


def _enhance_kpi_insights(kpi_insights, kpi_targets_df):
    kpi_insights = pd.merge(kpi_insights, kpi_targets_df, on=FieldNames.SERVICE_KPI, how="left")
    kpi_insights[FieldNames.CHANGE_IMPACT] = kpi_insights.apply(calculate_change_impact, axis=1)
    kpi_insights[FieldNames.CHANGE_DIRECTION] = kpi_insights[FieldNames.CHANGE_IN_KPI_VALUE].apply(
        lambda x: PowerBiConfig.CHANGE_DIRECTION_ARROWS[PowerBiConfig.POSITIVE] if pd.notna(x) and x > 0 else PowerBiConfig.
CHANGE_DIRECTION_ARROWS[PowerBiConfig.NEGATIVE]
    )
    return kpi_insights


def _create_matrix_rows(kpi_insights):
    rows = []
    
    for _, row in kpi_insights.iterrows():
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            FieldNames.STAT: MATRIX_STAT_TARGET,
            FieldNames.ARROW: DefaultValues.EMPTY_STRING,
            FieldNames.VALUE: row[FieldNames.TARGET],
            FieldNames.LINE_ORDER: LINE_ORDER_TARGET,
        })
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            FieldNames.STAT: MATRIX_STAT_CHANGE,
            FieldNames.ARROW: row[FieldNames.CHANGE_DIRECTION],
            FieldNames.VALUE: row[FieldNames.CHANGE_IN_KPI_VALUE],
            FieldNames.LINE_ORDER: LINE_ORDER_CHANGE,
        })
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            FieldNames.STAT: MATRIX_STAT_AVERAGE,
            FieldNames.ARROW: DefaultValues.EMPTY_STRING,
            FieldNames.VALUE: round(row[FieldNames.AVERAGE_KPI_VALUE], PowerBiConfig.DECIMAL_PLACES_ROUNDING),  # Rounded to 2 decimal
places
            FieldNames.LINE_ORDER: LINE_ORDER_AVERAGE,
        })
    
    return rows


def create_insights_matrix(kpi_insights, kpi_targets_df):
    
    kpi_insights = _enhance_kpi_insights(kpi_insights, kpi_targets_df)
    
    # Transform the data into the desired view
    rows = _create_matrix_rows(kpi_insights)
    # Create the new view DataFrame
    matrix_view = pd.DataFrame(rows)
    
    return kpi_insights, matrix_view

def assemble_final_dataframes_dict(matched_entries, standard_throughput_results, time_status_per_month, issue_rate, all_results_kpi, 
recent_results_kpi, kpi_insights, matrix_view, slo_met_percent_category, slo_met_percent_service, requestor_data_clean, 
requestor_data_clean_grouped):
    """Create the final OrderedDict of all dataframes"""
    
    dataframes_dict = OrderedDict([
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.MATCHED_ENTRIES], matched_entries),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.TICKETS_CLOSED], standard_throughput_results),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.TIME_KPIS], time_status_per_month),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.ISSUE_RATE], issue_rate),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.ALL_KPI_RESULTS], all_results_kpi),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.RECENT_KPI_RESULTS], recent_results_kpi),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.KPI_INSIGHTS], kpi_insights),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.KPI_MATRIX], matrix_view),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.CATEGORY_SLO], slo_met_percent_category),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.SERVICE_SLO], slo_met_percent_service),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.REQUEST_DATA_CLEAN], requestor_data_clean),
        (OutputConfig.DATAFRAME_NAMES[OutputConfig.REQUEST_DATA_GROUPED], requestor_data_clean_grouped),
    ])
    
    return dataframes_dict.copy()