# Standard imports
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

# Third-party imports
import pandas as pd

# Import other python functionality
from config import FieldNames, ProcessingConfig
from jira_parser import extract_name_description_pairs, extract_request_type, get_fields, extract_project_initiative, process_components
from data_processor import update_ticket_values, sort_changelog_entries, calculate_duration, process_changelog_entries
from kpi_calculator import standard_throughput_calculation, get_time_status_per_month, get_issue_rate
from output_formatter import convert_data_for_power_bi, get_requestor_data_clean

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
    
    # I want to add a new key and value to ticket_values, if the key project-issuetype == "DQMMBAU-Data Quality Rule" then I want to create a new key called "close_date" and set it to the value of FixVersion otherwise it should be "ResolutionDate"
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
    matched_entries = []
    
    # Iterate over each entry in the JSON data
    for entry in json_data:
        processed_entry = process_entry(entry, matched_entries)  # Process each entry individually
        if processed_entry:
            matched_entries.append(processed_entry)
    matched_entries = pd.DataFrame(matched_entries)
    
    # Add PREFERRED_ISSUE_TYPE column to matched_entries and mapping_table
    matched_entries = add_preferred_issue_type(matched_entries, FieldNames.REQUEST_TYPE, FieldNames.ISSUE_TYPE, FieldNames.PREFERRED_ISSUE_TYPE)
    mapping_table = add_preferred_issue_type(mapping_table, FieldNames.REQUEST_TYPE, FieldNames.ISSUE_TYPE, FieldNames.PREFERRED_ISSUE_TYPE)
    
    # Add the requestor column to matched_entries
    matched_entries = matched_entries.merge(
        requestor_df,
        how='left',
        left_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.FROM],  # Columns in matched_entries
        right_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.STATUS]  # Columns in mapping_table
    )
    # I want to see the [PROJECT_INITIATIVE_L1_COLUMN, PROJECT_INITIATIVE_L2_COLUMN] which are not mapped to a requestor. construct a df which shows me this
    unmapped_requestors = matched_entries[matched_entries[FieldNames.SERVICE_USER_COLUMN].isnull()][[FieldNames.PROJECT_INITIATIVE_L1_COLUMN, FieldNames.PROJECT_INITIATIVE_L2_COLUMN]].drop_duplicates()
    unmapped_requestors.to_csv('unmapped_requestors.csv', index=False)
    matched_entries.to_csv('matched_entries.csv', index=False)
    
    # Drop the unnecessary columns from the mapping_table
    columns_to_drop = [FieldNames.ISSUE_TYPE, FieldNames.REQUEST_TYPE]  # Columns to drop
    #mapping_table.to_csv("data_before_rowdrop.csv", index=False)
    mapping_table = mapping_table.drop(columns=columns_to_drop, errors='ignore')
    matched_entries = matched_entries.drop(columns=columns_to_drop, errors='ignore')
    #mapping_table.to_csv("data_after_rowdrop.csv", index=False)
    
    # I want to join the matched_entries with the mapping_table
    # It should be a left join with mapping joining onto matched_entries on the both PROJECT and ISSUE_TYPE columns. The relevant columns from mapping is Project Key and Issue Type
    matched_entries.to_csv("data_before_merge.csv", index=False)
    matched_entries = pd.merge(
        matched_entries,
        mapping_table,  # Select relevant columns
        how='left',
        left_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.FROM],  # Columns in matched_entries
        right_on=[FieldNames.PROJECT, FieldNames.PREFERRED_ISSUE_TYPE, FieldNames.STATUS]  # Columns in mapping_table
    )
    matched_entries.to_csv("data_after_merge.csv", index=False)
    
    def filter_matched_entries(matched_entries, project_issues_to_check):
        """
        Filters the matched_entries DataFrame based on allowed components for specific project-issuetypes.
        
        Args:
            matched_entries (pd.DataFrame): The DataFrame containing project-issuetype and ComponentName columns.
            project_issues_to_check (dict): A dictionary specifying allowed ComponentName values for certain project-issuetypes.
        
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        def is_row_allowed(row):
            project_issuetype = row["project-issuetype"]
            component_name = row["ComponentName"]
            

            # If the project-issuetype is not in project_issues_to_check, keep the row
            if project_issuetype not in project_issues_to_check:
                return True
            
            # If the project-issuetype is in project_issues_to_check, check if the component_name is allowed
            if component_name in project_issues_to_check[project_issuetype]:
                return True
            
            # Otherwise, exclude the row
            return False
        
        # Generate the boolean mask
        mask = matched_entries.apply(is_row_allowed, axis=1)
        
        # Apply the filtering logic
        filtered_entries = matched_entries.loc[mask]
        
        return filtered_entries
    

    # I want to do some filtering on this post-join DataFrame.
    # There are many different "ComponentName" values for each value of "project-issuetype" within matched_entries for certain project-issuetypes I want to only keep  some ComponentNames                #
    # Example dictionary specifying allowed ComponentNames for certain project-issuetypes
    project_issues_to_check = {
        "DQMMBAU-Data Quality Rule": ["DQMMBAU - New DQ Rule", "DQMMBAU - Update DQ Rule"],
        "DQMMBAU-Consumer Validation": ["DQMMBAU - New DQ Rule", "DQMMBAU - Update DQ Rule"],
        "FMGDIRIS-PFI Sub-Task": ["Testing - PFI E2E"],
    }
    
    # Add mapping for Service and KPI
    kpi_targets_df[FieldNames.SERVICE_KPI] = kpi_targets_df['Service'] + ' - ' + kpi_targets_df['KPI']
    
    # Filter rows based on the conditions
    matched_entries = filter_matched_entries(matched_entries, project_issues_to_check)
    
    # Add the RESOLUTION_DATE_YYYY_MM column to the matched_entries DataFrame
    matched_entries[FieldNames.RESOLUTION_DATE_YYYY_MM] = matched_entries['close_date'].astype(str).str[:7]
    
    # Get the service to category mapping
    services_categories = matched_entries[['Service', 'Category']].drop_duplicates()
    
    # Calculate KPIs
    standard_throughput_results = standard_throughput_calculation(matched_entries, kpi_targets_df)
    time_status_per_month = get_time_status_per_month(matched_entries, kpi_targets_df)
    issue_rate = get_issue_rate(matched_entries)
    
    # Get data into a power BI format
    throughput_results_kpi, recent_throughput_results_kpi, throughput_insights = convert_data_for_power_bi(standard_throughput_results)
    time_results_kpi, recent_time_results_kpi, recent_time_insights = convert_data_for_power_bi(time_status_per_month)
    
    # Concatenate the results for Power BI
    all_results_kpi = pd.concat([throughput_results_kpi, time_results_kpi], ignore_index=True)
    recent_results_kpi = pd.concat([recent_throughput_results_kpi, recent_time_results_kpi], ignore_index=True)
    kpi_insights = pd.concat([throughput_insights, recent_time_insights], ignore_index=True)
    
    # Drop the 'KPI Type' column from all_results_kpi
    all_results_kpi = all_results_kpi.drop(columns=['KPI Type'], errors='ignore')
    
    # Drop rows of all_results_kpi which don't have a value for Target as there is no kpi assigned to it
    recent_results_kpi = recent_results_kpi.dropna(subset=[FieldNames.TARGET])
    # Left join definitions to the recent results on 'Service'
    recent_results_kpi = pd.merge(recent_results_kpi, category_definitions_df, on="Service", how="left")
    # I want to rename the 'Definition' column to CATEGORY_DEFINITION in recent_results_kpi
    recent_results_kpi = recent_results_kpi.rename(columns={'Definition': FieldNames.CATEGORY_DEFINITION})
    
    # Left join kpi definitions on recent results on 'KPI Type'
    recent_results_kpi = pd.merge(recent_results_kpi, kpi_definitions_df, on="KPI Type", how="left")
    # I want to rename the 'Definition' column to KPI_DEFINITION in recent_results_kpi
    recent_results_kpi = recent_results_kpi.rename(columns={'Definition': FieldNames.KPI_DEFINITION})
    
    # Drop rows of all_results_kpi which don't have a value for Target as there is no kpi assigned to it
    all_results_kpi = all_results_kpi.dropna(subset=[FieldNames.TARGET])
    # Sort the DataFrame by the Date column
    all_results_kpi = all_results_kpi.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    all_results_kpi["DateOrder"] = range(1, len(all_results_kpi) + 1)
    
    # Turn all_results_kpi into a format for Power BI SLO graph
    slo_met_percent_category = all_results_kpi[["Service", FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.KPI_MET, FieldNames.SERVICE_KPI, "ResolutionDate_string", FieldNames.DATES]]
    
    # Left join services_categories on slo_met_percent using 'Service' as the key
    slo_met_percent_category = pd.merge(slo_met_percent_category, services_categories, on="Service", how="left")
    
    # Drop the "Service-KPI" and "Service" columns from slo_met_percent
    slo_met_percent_category = slo_met_percent_category.drop(columns=[FieldNames.SERVICE_KPI, "Service"], errors='ignore')
    
    # Step 1: Group by Service-KPI and ResolutionDate_yyyy_mm
    slo_met_percent_category = (
        slo_met_percent_category.groupby(["Category", FieldNames.DATES])
        .agg(
            KPI_Met_Percentage=pd.NamedAgg(
                column=FieldNames.KPI_MET,
                aggfunc = lambda x: (x.sum() / len(x)) * 100 if x.dtype == bool else None
            )
        )
        .reset_index()
    )
    
    # Step 2: Add DateOrder column
    slo_met_percent_category = slo_met_percent_category.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    
    # Add a new column for ordering in Power BI
    slo_met_percent_category["DateOrder"] = range(1, len(slo_met_percent_category) + 1)
    
    # Get Service KPI insights
    # Turn all_results_kpi into a format for Power BI SLO graph
    slo_met_percent_service = all_results_kpi[["Service", FieldNames.RESOLUTION_DATE_YYYY_MM, FieldNames.KPI_MET, "ResolutionDate_string", FieldNames.DATES]]
    
    # Step 1: Group by Service-KPI and ResolutionDate_yyyy_mm
    slo_met_percent_service = (
        slo_met_percent_service.groupby(["Service", FieldNames.DATES])
        .agg(
            KPI_Met_Percentage = pd.NamedAgg(
                column=FieldNames.KPI_MET,
                aggfunc=lambda x: (x.sum() / len(x)) * 100 if x.dtype == bool else None
            )
        )
        .reset_index()
    )
    
    # Step 2: Add DateOrder column
    slo_met_percent_service = slo_met_percent_service.sort_values(by=FieldNames.DATES).reset_index(drop=True)
    # Add a new column for ordering in Power BI
    slo_met_percent_service["DateOrder"] = range(1, len(slo_met_percent_service) + 1)
    
    # Merge kpi targets on kpi_insights using SERVICE_KPI as the key
    def calculate_change_impact(row):
        if row["KPI"] == "Throughput":
            return "True" if row["Change in KPI Value"] and row["Change in KPI Value"] > 0 else "False"
        else:
            return "False" if row["Change in KPI Value"] and row["Change in KPI Value"] > 0 else "Positive"
    
    kpi_insights = pd.merge(kpi_insights, kpi_targets_df, on=FieldNames.SERVICE_KPI, how="left")
    kpi_insights["change impact"] = kpi_insights.apply(calculate_change_impact, axis=1)
    kpi_insights["change direction"] = kpi_insights["Change in KPI Value"].apply(
        lambda x: "↑" if pd.notna(x) and x > 0 else "↓"
    )

    # Transform the data into the desired view
    rows = []
    for _, row in kpi_insights.iterrows():
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            "Stat": "Target (red line)",
            "Arrow": "",
            "Value": row[FieldNames.TARGET],
            "Line order": 1,
        })
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            "Stat": "Change (Month on Month)",
            "Arrow": row["change_direction"],
            "Value": row["Change in KPI Value"],
            "Line order": 2,
        })
        rows.append({
            FieldNames.SERVICE_KPI: row[FieldNames.SERVICE_KPI],
            "Stat": "6-Month Average",
            "Arrow": "",
            "Value": round(row["Average KPI Value"], 2),  # Rounded to 2 decimal places
            "Line order": 3,
        })
    
    # Get requestor information
    requestor_data_clean, requestor_data_clean_grouped = get_requestor_data_clean(matched_entries)
    
    # Create the new view DataFrame
    matrix_view = pd.DataFrame(rows)
    
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

def get_issue_data(input_json, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, output_xlsx_path):
    
    # Load the JSON file
    with open(input_json, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # Use the DataFrame to filter the JSON data
    filtered_json = filter_json_with_status_durations(data, mapping_table, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, output_xlsx_path=output_xlsx_path)
    
    return filtered_json