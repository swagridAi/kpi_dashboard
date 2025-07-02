import pandas as pd
from output_formatter import convert_data_for_power_bi
from config import FieldNames, PowerBIConfig
from collections import OrderedDict

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
    
    # Left join definitions to the recent results on 'Service' and Rename the 'Definition' column to CATEGORY_DEFINITION in recent_results_kpi
    recent_results_kpi = pd.merge(recent_results_kpi, category_definitions_df, on="Service", how="left")
    recent_results_kpi = recent_results_kpi.rename(columns={PowerBIConfig.DEFINITION_COLUMN: FieldNames.CATEGORY_DEFINITION})
    
    # Left join kpi definitions on recent results on 'KPI Type' and Rename the 'Definition' column to KPI_DEFINITION in recent_results_kpi
    recent_results_kpi = pd.merge(recent_results_kpi, kpi_definitions_df, on="KPI Type", how="left")
    recent_results_kpi = recent_results_kpi.rename(columns={PowerBIConfig.DEFINITION_COLUMN: FieldNames.KPI_DEFINITION})
    
    return recent_results_kpi

def process_all_results_for_power(all_results_kpi):
    """Process all results for Power BI consumption"""
    
    # Drop rows of all_results_kpi which don't have a value for Target as there is no kpi assigned to it
    all_results_kpi = all_results_kpi.dropna(subset=[FieldNames.TARGET])
    # Sort the DataFrame by the Date column
    all_results_kpi = all_results_kpi.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    all_results_kpi["DateOrder"] = range(1, len(all_results_kpi) + 1)
    
    return all_results_kpi

def create_slo_category_view(all_results_kpi, services_categories):
    
    # Turn all_results_kpi into a format for Power BI SLO graph
    slo_met_percent_category = all_results_kpi[["Service", FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.KPI_MET, FieldNames.SERVICE_KPI, "ResolutionDate_string", FieldNames.DATES]]
    
    # Left join services_categories on slo_met_percent using 'Service' as the key
    slo_met_percent_category = pd.merge(slo_met_percent_category, services_categories, on="Service", how="left")
    
    # Drop the "Service-KPI" and "Service" columns from slo_met_percent
    slo_met_percent_category = slo_met_percent_category.drop(columns=[FieldNames.SERVICE_KPI, "Service"], errors='ignore')
    
    # Group by Service-KPI and ResolutionDate_yyyy_mm
    slo_met_percent_category = (
        slo_met_percent_category.groupby(["Category", FieldNames.DATES])
        .agg(
            KPI_Met_Percentage=pd.NamedAgg(
                column=FieldNames.KPI_MET,
                aggfunc=lambda x: (x.sum() / len(x)) * 100 if x.dtype == bool else None
            )
        )
        .reset_index()
    )
    
    # Add DateOrder column
    slo_met_percent_category = slo_met_percent_category.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    slo_met_percent_category["DateOrder"] = range(1, len(slo_met_percent_category) + 1)
    
    return slo_met_percent_category

def create_slo_service_view(all_results_kpi):
    slo_met_percent_service = all_results_kpi[["Service", FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.KPI_MET, "ResolutionDate_string", FieldNames.DATES]]
    
    # Group by Service-KPI and ResolutionDate_yyyy_mm
    slo_met_percent_service = (
        slo_met_percent_service.groupby(["Service", FieldNames.DATES])
        .agg(
            KPI_Met_Percentage=pd.NamedAgg(
                column=FieldNames.KPI_MET,
                aggfunc=lambda x: (x.sum() / len(x)) * 100 if x.dtype == bool else None
            )
        )
        .reset_index()
    )
    
    # Add DateOrder column
    slo_met_percent_service = slo_met_percent_service.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    slo_met_percent_service["DateOrder"] = range(1, len(slo_met_percent_service) + 1)
    return slo_met_percent_service

def calculate_change_impact(row):
    if row["KPI"] == PowerBIConfig.THROUGHPUT_KPI_TYPE:
        return PowerBIConfig.CHANGE_IMPACT_TRUE if row["Change in KPI Value"] and row["Change in KPI Value"] > 0 else PowerBIConfig.CHANGE_IMPACT_FALSE
    else:
        return PowerBIConfig.CHANGE_IMPACT_FALSE if row["Change in KPI Value"] and row["Change in KPI Value"] > 0 else PowerBIConfig.CHANGE_IMPACT_POSITIVE

def _enhance_kpi_insights(kpi_insights, kpi_targets_df):
    kpi_insights = pd.merge(kpi_insights, kpi_targets_df, on=FieldNames.SERVICE_KPI, how="left")
    kpi_insights["change_impact"] = kpi_insights.apply(calculate_change_impact, axis=1)
    kpi_insights["change_direction"] = kpi_insights["Change in KPI Value"].apply(
        lambda x: PowerBIConfig.CHANGE_DIRECTION_UP if pd.notna(x) and x > 0 else PowerBIConfig.CHANGE_DIRECTION_DOWN
    )
    return kpi_insights

def _create_matrix_rows(kpi_insights):
    rows = []
    for _, row in kpi_insights.iterrows():
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            "Stat": PowerBIConfig.MATRIX_STAT_TARGET,
            "Arrow": "",
            "Value": row[FieldNames.TARGET],
            "line_order": PowerBIConfig.LINE_ORDER_TARGET,
        })
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            "Stat": PowerBIConfig.MATRIX_STAT_CHANGE,
            "Arrow": row["change_direction"],
            "Value": row["Change in KPI Value"],
            "line_order": PowerBIConfig.LINE_ORDER_CHANGE,
        })
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            "Stat": PowerBIConfig.MATRIX_STAT_AVERAGE,
            "Arrow": "",
            "Value": round(row["Average KPI Value"], PowerBIConfig.DECIMAL_PLACES_ROUNDING),  # Rounded to 2 decimal places
            "line_order": PowerBIConfig.LINE_ORDER_AVERAGE,
        })
    return rows

def create_insights_matrix(kpi_insights, kpi_targets_df):
    
    kpi_insights = _enhance_kpi_insights(kpi_insights, kpi_targets_df)
    
    # Transform the data into the desired view
    rows = _create_matrix_rows(kpi_insights)
    # Create the new view DataFrame
    matrix_view = pd.DataFrame(rows)
    
    return kpi_insights, matrix_view

def assemble_final_dataframes_dict(matched_entries, standard_throughput_results, time_status_per_month, issue_rate, all_results_kpi, recent_results_kpi, kpi_insights, matrix_view, slo_met_percent_category, slo_met_percent_service, requestor_data_clean, requestor_data_clean_grouped):
    "Create the final OrderedDict of all dataframes"
    
    dataframes_dict = OrderedDict([
        ("Matched Entries", matched_entries),
        ("Tickets Closed", standard_throughput_results),
        ("Time KPIs", time_status_per_month),
        ("Issue Rate", issue_rate),
        ("All KPI Results", all_results_kpi),
        ("Recent KPI Results", recent_results_kpi),
        ("KPI Insights", kpi_insights),
        ("KPI Matrix", matrix_view),
        ("Category SLO Met Percent", slo_met_percent_category),
        ("Service SLO Met Percent", slo_met_percent_service),
        ("Request Data Clean", requestor_data_clean),
        ("Request Data Clean Grouped", requestor_data_clean_grouped),
    ])
    
    return dataframes_dict.copy()