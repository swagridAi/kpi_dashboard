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

# Start logging
logging.basicConfig(level=logging.INFO)
logging.info("Adding PreferredIssueType column to mapping_table.")

def process_entry(entry: Dict[str, Any], matched_entries) -> None:
    """
    Process a single entry and calculate durations without matching name-description pairs.
    """
    def extract_changelog_entries(changelog: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relevant changelog entries where the field is 'status'.
        """
        if not changelog or 'histories' not in changelog:
            return []
        
        entries = []
        for history in changelog['histories']:
            author = history.get('author', {}).get('displayName', 'Unknown')
            change_created = history.get('created', 'Unknown')
            for item in history.get('items', []):
                if item.get('field', 'Unknown') == 'status':
                    entries.append({
                        'Author': author,
                        'ChangeCreated': change_created,
                        'Field': item.get('field', 'Unknown'),
                        FieldNames.FROM: item.get('fromString', 'Unknown'),
                        'To': item.get('toString', 'Unknown')
                    })
        return entries

    def extract_request_type(entry):
        """Extract the request type from a JSON entry."""
        if entry is None:
            # Handle the case where entry is None
            return {}
        
        # Safely access nested fields with .get()
        fields = entry.get('fields', {})
        custom_field = fields.get('customfield_23641', {})
        if not isinstance(custom_field, dict):
            # If custom_field is not a dictionary, return an empty dictionary
            return {}
        
        # Return the requestType if it exists, otherwise return an empty dictionary
        return custom_field.get('requestType', {})

    ticket_values = get_fields(entry)
    if not ticket_values:
        # If no valid fields are found, skip processing this entry
        return

    #TODO: Add CCF date later in the process
    #TODO: Add the sub component for the PFI testing
    changelog = entry.get('changelog', None)
    components = entry.get('fields', {}).get('components', None)
    request_type = extract_request_type(entry)

    # Get the request type
    service_desk_request_type = request_type.get('name', None)
    service_desk_request_description = request_type.get('description', None)
    ticket_values.update({
        FieldNames.REQUEST_TYPE: service_desk_request_type,
        'RequestDescription': service_desk_request_description
        })

    # Process components
    component_dict = process_components(components)
    ticket_values.update(component_dict)

    # I want to add a new key and value to ticket_values, if the key project-issuetype == "DQMBAU-Data Quality Rule" then I want to create a new key called  "close_date" and set it to the value of FixVersion otherwise it should be "ResolutionDate"
    ticket_values = update_ticket_values(ticket_values)

    # Process changelogs
    changelog_entries = extract_changelog_entries(changelog)
    changelog_entries = sort_changelog_entries(changelog_entries)
    process_changelog_entries(
        changelog_entries, matched_entries, ticket_values
    )

def filter_json_with_status_durations(json_data, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, output_xlsx_path):
    """
    Filter the JSON data using the Name and Description values from the filtered DataFrame.
    Calculate how long each status was active. If the status is still active, use the current time as the end time.
    """
    
    def process_jira_entries(json_data):
        """Process raw JSON entries into DataFrame"""
        matched_entries = []
        # Iterate over each entry in the JSON data
        for entry in json_data:
            processed_entry = process_entry(entry, matched_entries)  # Process each entry individually
            if processed_entry:
                matched_entries.append(processed_entry)
        return pd.DataFrame(matched_entries)
    
    matched_entries = process_jira_entries(json_data)

    # Add PREFERRED_ISSUE_TYPE column to matched_entries and mapping_table
    matched_entries = add_preferred_issue_type(matched_entries, FieldNames.REQUEST_TYPE, FieldNames.ISSUE_TYPE, FieldNames.PREFERRED_ISSUE_TYPE)
    mapping_table = add_preferred_issue_type(mapping_table, FieldNames.REQUEST_TYPE, FieldNames.ISSUE_TYPE, FieldNames.PREFERRED_ISSUE_TYPE)
    
    # ADD the requestor column to matched_entries
    matched_entries = merge_requestor_data(matched_entries, requestor_df)
    unmapped_requestors = generate_unmapped_requestors_report(matched_entries)
    matched_entries.to_csv('matched_entries.csv', index=False)
    matched_entries = merge_mapping_tables(matched_entries, mapping_table)
    
    # Add mapping for Service and KPI
    kpi_targets_df[FieldNames.SERVICE_KPI] = kpi_targets_df['Service'] + ' - ' + kpi_targets_df['KPI']
    
    # Filter rows based on the conditions
    matched_entries = filter_matched_entries(matched_entries, BusinessRules.PROJECT_COMPONENT_FILTERS)
    
    # Add the RESOLUTION_DATE_YYYY_MM column to the matched_entries DataFrame
    matched_entries[FieldNames.RESOLUTION_DATE_YYYY_MM] = matched_entries['close_date'].astype(str).str[:7]
    
    # Get the service to category mapping
    services_categories = matched_entries[['Service', 'Category']].drop_duplicates()
    
    # Calculate KPIs
    standard_throughput_results = standard_throughput_calculation(matched_entries, kpi_targets_df)
    time_status_per_month = get_time_status_per_month(matched_entries, kpi_targets_df)
    issue_rate = get_issue_rate(matched_entries)
    all_results_kpi, recent_results_kpi, kpi_insights = create_kpi_result_views(standard_throughput_results, time_status_per_month)
    
    # Drop the 'KPI Type' column from all_results_kpi
    all_results_kpi = all_results_kpi.drop(columns=['KPI Type'], errors='ignore')
    recent_results_kpi = enhance_recent_results(recent_results_kpi, category_definitions_df, kpi_definitions_df)
    all_results_kpi = process_all_results_for_power(all_results_kpi)
    slo_met_percent_category = create_slo_category_view(all_results_kpi, services_categories)
    
    # Get Service KPI insights
    # Turn all_results_kpi into a format for Power BI SLO graph
    slo_met_percent_service = create_slo_service_view(all_results_kpi)
    
    # Merge kpi_targets on kpi_insights using SERVICE_KPI as the key
    kpi_insights, matrix_view = create_insights_matrix(kpi_insights, kpi_targets_df)
    
    # Get requestor information
    requestor_data_clean, requestor_data_clean_grouped = get_requestor_data_clean(matched_entries)
    
    dataframes_dict = assemble_final_dataframes_dict(matched_entries, standard_throughput_results, time_status_per_month, issue_rate, all_results_kpi, recent_results_kpi, kpi_insights, matrix_view, slo_met_percent_category, slo_met_percent_service, requestor_data_clean, requestor_data_clean_grouped)
    
    return dataframes_dict

def get_issue_data(input_json, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, output_xlsx_path):
    
    # Load the JSON file
    with open(input_json, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # Use the DataFrame to filter the JSON data
    filtered_json = filter_json_with_status_durations(data, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df,  output_xlsx_path=output_xlsx_path)
    
    return filtered_json