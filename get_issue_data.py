# Standard imports
import json
import logging

from typing import Any, Dict, List, Tuple

# Third-party imports
import pandas as pd

# Import other python functionality
from config import FieldNames, ProcessingConfig, BusinessRules
from jira_parser import extract_name_description_pairs, extract_request_type, get_fields, extract_project_initiative, process_components
from data_processor import update_ticket_values, sort_changelog_entries, calculate_duration, process_changelog_entries, add_preferred_issue_type,merge_requestor_data, generate_unmapped_requestors_report, merge_mapping_tables
from kpi_calculator import standard_throughput_calculation, get_time_status_per_month, get_issue_rate
from output_formatter import convert_data_for_power_bi, get_requestor_data_clean
from business_rules import filter_matched_entries
from powerbi_formatter import create_kpi_result_views, enhance_recent_results, process_all_results_for_power, create_slo_category_view, create_slo_service_view, create_insights_matrix, assemble_final_dataframes_dict

# Constants
DEFAULT_VALUE_UNKNOWN = 'Unknown'
JIRA_FIELD_STATUS = 'status'
JIRA_FIELD_FIELD = 'field'
JIRA_FIELD_HISTORIES = 'histories'
JIRA_FIELD_AUTHOR = 'author'
JIRA_FIELD_DISPLAY_NAME = 'displayName'
JIRA_FIELD_CREATED = 'created'
JIRA_FIELD_ITEMS = 'items'
JIRA_FIELD_FROM_STRING = 'fromString'
JIRA_FIELD_TO_STRING = 'toString'
JIRA_FIELD_CHANGELOG = 'changelog'
JIRA_FIELD_FIELDS = 'fields'
JIRA_FIELD_COMPONENTS = 'components'
JIRA_FIELD_NAME = 'name'
JIRA_FIELD_DESCRIPTION = 'description'
JIRA_CUSTOM_FIELD_REQUEST_TYPE = 'customfield_23641'
JIRA_FIELD_REQUEST_TYPE = 'requestType'
OUTPUT_FILE_MATCHED_ENTRIES = 'matched_entries.csv'
FILE_ENCODING_UTF8 = 'utf-8'
FILE_MODE_READ = 'r'

# Start logging
logging.basicConfig(level=logging.INFO)
logging.info("Adding PreferredIssueType column to mapping_table.")

def _is_status_change_item(item):
    """Check if item represents a status change."""
    return item.get(JIRA_FIELD_FIELD, DEFAULT_VALUE_UNKNOWN) == JIRA_FIELD_STATUS

def _extract_changelog_entries(changelog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract relevant changelog entries where the field is 'status'.
    """
    if not changelog or JIRA_FIELD_HISTORIES not in changelog:
        return []
    
    entries = []
    for history in changelog[JIRA_FIELD_HISTORIES]:
        author = history.get(JIRA_FIELD_AUTHOR, {}).get(JIRA_FIELD_DISPLAY_NAME, DEFAULT_VALUE_UNKNOWN)
        change_created = history.get(JIRA_FIELD_CREATED, DEFAULT_VALUE_UNKNOWN)
        for item in history.get(JIRA_FIELD_ITEMS, []):
            if _is_status_change_item(item):
                entries.append({
                    'Author': author,
                    'ChangeCreated': change_created,
                    'Field': item.get(JIRA_FIELD_FIELD, DEFAULT_VALUE_UNKNOWN),
                    FieldNames.FROM: item.get(JIRA_FIELD_FROM_STRING, DEFAULT_VALUE_UNKNOWN),
                    'To': item.get(JIRA_FIELD_TO_STRING, DEFAULT_VALUE_UNKNOWN)
                })
    return entries

def _extract_request_type(entry):
    """Extract the request type from a JSON entry."""
    if entry is None:
        return {}
    
    fields = entry.get(JIRA_FIELD_FIELDS, {})
    custom_field = fields.get(JIRA_CUSTOM_FIELD_REQUEST_TYPE, {})
    if not isinstance(custom_field, dict):
        return {}
    
    return custom_field.get(JIRA_FIELD_REQUEST_TYPE, {})

def _process_jira_entries(json_data):
    """Process raw JSON entries into DataFrame"""
    matched_entries = []
    for entry in json_data:
        processed_entry = process_entry(entry, matched_entries)
        if processed_entry:
            matched_entries.append(processed_entry)
    return pd.DataFrame(matched_entries)

def _prepare_data_for_analysis(matched_entries, mapping_table, requestor_df):
    """Prepare and merge data for KPI analysis."""
    matched_entries = add_preferred_issue_type(matched_entries, FieldNames.REQUEST_TYPE, FieldNames.ISSUE_TYPE, FieldNames.PREFERRED_ISSUE_TYPE)
    mapping_table = add_preferred_issue_type(mapping_table, FieldNames.REQUEST_TYPE, FieldNames.ISSUE_TYPE, FieldNames.PREFERRED_ISSUE_TYPE)
    
    matched_entries = merge_requestor_data(matched_entries, requestor_df)
    unmapped_requestors = generate_unmapped_requestors_report(matched_entries)
    matched_entries.to_csv(OUTPUT_FILE_MATCHED_ENTRIES, index=False)
    matched_entries = merge_mapping_tables(matched_entries, mapping_table)
    
    matched_entries = filter_matched_entries(matched_entries, BusinessRules.PROJECT_COMPONENT_FILTERS)
    matched_entries[FieldNames.RESOLUTION_DATE_YYYY_MM] = matched_entries['close_date'].astype(str).str[:7]
    
    return matched_entries

def _calculate_kpi_results(matched_entries, kpi_targets_df):
    """Calculate all KPI results."""
    kpi_targets_df[FieldNames.SERVICE_KPI] = kpi_targets_df['Service'] + ' - ' + kpi_targets_df['KPI']
    
    services_categories = matched_entries[['Service', 'Category']].drop_duplicates()
    
    standard_throughput_results = standard_throughput_calculation(matched_entries, kpi_targets_df)
    time_status_per_month = get_time_status_per_month(matched_entries, kpi_targets_df)
    issue_rate = get_issue_rate(matched_entries)
    all_results_kpi, recent_results_kpi, kpi_insights = create_kpi_result_views(standard_throughput_results, time_status_per_month)
    
    return standard_throughput_results, time_status_per_month, issue_rate, all_results_kpi, recent_results_kpi, kpi_insights, services_categories

def _create_power_bi_views(all_results_kpi, recent_results_kpi, kpi_insights, services_categories, category_definitions_df, kpi_definitions_df, kpi_targets_df, matched_entries):
    """Create Power BI formatted views."""
    all_results_kpi = all_results_kpi.drop(columns=['KPI Type'], errors='ignore')
    recent_results_kpi = enhance_recent_results(recent_results_kpi, category_definitions_df, kpi_definitions_df)
    all_results_kpi = process_all_results_for_power(all_results_kpi)
    slo_met_percent_category = create_slo_category_view(all_results_kpi, services_categories)
    
    slo_met_percent_service = create_slo_service_view(all_results_kpi)
    
    kpi_insights, matrix_view = create_insights_matrix(kpi_insights, kpi_targets_df)
    
    requestor_data_clean, requestor_data_clean_grouped = get_requestor_data_clean(matched_entries)
    
    return all_results_kpi, recent_results_kpi, kpi_insights, matrix_view, slo_met_percent_category, slo_met_percent_service, requestor_data_clean, requestor_data_clean_grouped

def process_entry(entry: Dict[str, Any], matched_entries) -> None:
    """
    Process a single entry and calculate durations without matching name-description pairs.
    """
    ticket_values = get_fields(entry)
    if not ticket_values:
        return

    changelog = entry.get(JIRA_FIELD_CHANGELOG, None)
    components = entry.get(JIRA_FIELD_FIELDS, {}).get(JIRA_FIELD_COMPONENTS, None)
    request_type = _extract_request_type(entry)

    service_desk_request_type = request_type.get(JIRA_FIELD_NAME, None)
    service_desk_request_description = request_type.get(JIRA_FIELD_DESCRIPTION, None)
    ticket_values.update({
        FieldNames.REQUEST_TYPE: service_desk_request_type,
        'RequestDescription': service_desk_request_description
        })

    component_dict = process_components(components)
    ticket_values.update(component_dict)

    ticket_values = update_ticket_values(ticket_values)

    changelog_entries = _extract_changelog_entries(changelog)
    changelog_entries = sort_changelog_entries(changelog_entries)
    process_changelog_entries(
        changelog_entries, matched_entries, ticket_values
    )

def filter_json_with_status_durations(json_data, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, output_xlsx_path):
    """
    Filter the JSON data using the Name and Description values from the filtered DataFrame.
    Calculate how long each status was active. If the status is still active, use the current time as the end time.
    """
    matched_entries = _process_jira_entries(json_data)
    
    matched_entries = _prepare_data_for_analysis(matched_entries, mapping_table, requestor_df)
    
    standard_throughput_results, time_status_per_month, issue_rate, all_results_kpi, recent_results_kpi, kpi_insights, services_categories = _calculate_kpi_results(matched_entries, kpi_targets_df)
    
    all_results_kpi, recent_results_kpi, kpi_insights, matrix_view, slo_met_percent_category, slo_met_percent_service, requestor_data_clean, requestor_data_clean_grouped = _create_power_bi_views(all_results_kpi, recent_results_kpi, kpi_insights, services_categories, category_definitions_df, kpi_definitions_df, kpi_targets_df, matched_entries)
    
    dataframes_dict = assemble_final_dataframes_dict(matched_entries, standard_throughput_results, time_status_per_month, issue_rate, all_results_kpi, recent_results_kpi, kpi_insights, matrix_view, slo_met_percent_category, slo_met_percent_service, requestor_data_clean, requestor_data_clean_grouped)
    
    return dataframes_dict

def get_issue_data(input_json, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, output_xlsx_path):
    
    with open(input_json, FILE_MODE_READ, encoding=FILE_ENCODING_UTF8) as json_file:
        data = json.load(json_file)
    
    filtered_json = filter_json_with_status_durations(data, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df,  output_xlsx_path=output_xlsx_path)
    
    return filtered_json