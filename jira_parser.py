# Standard packages
from typing import Any, Dict, List, Tuple

from config import FieldNames, ProcessingConfig, JiraFields, DataFrameColumns, DefaultValues, ComponentsProcessing
from data_processor import calculate_duration

# Default values
EMPTY_LIST = []
EMPTY_DICT = {}
PROJECT_KEY_SEPARATOR = '-'
FALSE_RETURN = False
COMPONENT_SPLITTER = ": "

def extract_name_description_pairs(filtered_df):
    """Extract Name and Description pairs from the DataFrame."""
    return filtered_df[[DataFrameColumns.NAME, DataFrameColumns.DESCRIPTION]].to_records(index=False)

def extract_request_type(entry):
    """
    Extract the request type from the entry safely.
    """
    fields = entry.get(JiraFields.FIELDS, EMPTY_DICT)
    custom_field = fields.get(JiraFields.REQUEST_TYPE_CUSTOM_FIELD, EMPTY_DICT)
    
    # Ensure custom_field is a dictionary before accessing 'requestType'
    if isinstance(custom_field, dict):
        return custom_field.get(JiraFields.REQUEST_TYPE_SUBFIELD, EMPTY_DICT)
    
    # Return an empty dictionary if custom_field is None or not a dictionary
    return EMPTY_DICT

def _extract_entry_fields(entry: Dict[str, Any]) -> Tuple[str, str, bool, str, Dict[str, str], str, str, List[Dict[str, Any]], int]:
    """Extract basic fields from entry."""
    entry_key = entry.get(JiraFields.KEY, None)
    resolution_date = entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.RESOLUTION_DATE, None)
    has_issues = bool(entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.ISSUE_LINKS, EMPTY_LIST))
    issue_type_field = entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.ISSUE_TYPE, None)
    issue_type = issue_type_field.get(JiraFields.NAME_SUBFIELD) if isinstance(issue_type_field, dict) and JiraFields.NAME_SUBFIELD in issue_type_field else ProcessingConfig.UNKNOWN_VALUE
    project_initiative = extract_project_initiative(entry)
    created_date = entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.CREATED, None)
    ccf_date = entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.CCF_DATE_CUSTOM_FIELD, None)
    fix_version = entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.FIX_VERSIONS, EMPTY_LIST)
    timespent_seconds = entry.get(JiraFields.FIELDS, EMPTY_DICT).get(JiraFields.TIME_SPENT, None)
    return entry_key, resolution_date, has_issues, issue_type, project_initiative, created_date, ccf_date, fix_version, timespent_seconds

def get_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract fields from the entry.
    """
    entry_key, resolution_date, has_issues, issue_type, project_initiative, created_date, ccf_date, fix_version, timespent_seconds = _extract_entry_fields(entry)
    
    if not resolution_date:
        return FALSE_RETURN
    
    ticket_duration = calculate_duration(created_date, resolution_date)
    
    ticket_values = {
        FieldNames.KEY: entry_key,
        FieldNames.RESOLUTION_DATE: resolution_date,
        FieldNames.PROJECT: entry_key.split(PROJECT_KEY_SEPARATOR)[ProcessingConfig.FIRST_ELEMENT_INDEX],
        FieldNames.ISSUE_TYPE: issue_type,
        FieldNames.PROJECT_ISSUE_TYPE: f"{entry_key.split(PROJECT_KEY_SEPARATOR)[ProcessingConfig.FIRST_ELEMENT_INDEX]}{PROJECT_KEY_SEPARATOR}{issue_type}",
        FieldNames.CREATED: created_date,
        FieldNames.HAS_ISSUES: has_issues,
        FieldNames.CCF_DATE: ccf_date,
        FieldNames.TICKET_DURATION: ticket_duration,
        FieldNames.FIX_VERSION: fix_version[ProcessingConfig.FIRST_ELEMENT_INDEX][JiraFields.NAME_SUBFIELD] if fix_version else None,
        FieldNames.TIME_SPENT_SECONDS: timespent_seconds,
        FieldNames.PROJECT_INITIATIVE_L1_COLUMN: project_initiative[FieldNames.PARENT_VALUE],
        FieldNames.PROJECT_INITIATIVE_L2_COLUMN: project_initiative[FieldNames.CHILD_VALUE]
    }
    
    return ticket_values

def extract_project_initiative(entry):
    """Extract the project and initiative from a JSON entry."""
    
    if entry is None:
        # Handle the case where entry is None
        return {FieldNames.PARENT_VALUE: ProcessingConfig.EMPTY_STRING, FieldNames.CHILD_VALUE: ProcessingConfig.EMPTY_STRING}
    
    # Safely access nested fields with .get()
    fields = entry.get(JiraFields.FIELDS, EMPTY_DICT)
    custom_field = fields.get(JiraFields.PROJECT_INITIATIVE_CUSTOM_FIELD, EMPTY_DICT)
    if not isinstance(custom_field, dict):
        # If custom_field is not a dictionary, return an empty dictionary
        return {FieldNames.PARENT_VALUE: ProcessingConfig.EMPTY_STRING, FieldNames.CHILD_VALUE: ProcessingConfig.EMPTY_STRING}
    
    # Extract the parent value
    parent_value = custom_field.get(JiraFields.VALUE_SUBFIELD, None)
    
    # Extract the child value if it exists
    child = custom_field.get(JiraFields.CHILD_SUBFIELD, EMPTY_DICT)
    child_value = child.get(JiraFields.VALUE_SUBFIELD, None) if isinstance(child, dict) else None
    
    # Return both parent and child values in a dictionary
    return {
        FieldNames.PARENT_VALUE: parent_value,
        FieldNames.CHILD_VALUE: child_value
    }

def process_components(
    components: List[Dict[str, Any]]
) -> None:
    """
    Process changelog entries and calculate status durations.
    """
    
    if components == EMPTY_LIST:
        return EMPTY_DICT
    
    component_dict = {
        FieldNames.COMPONENT_NAME: DefaultValues.EMPTY_STRING,
        FieldNames.COMPONENT_DESCRIPTION: DefaultValues.EMPTY_STRING
    }
    
    if len(components) == 1:
        # We want to get the only element
        components = components[ProcessingConfig.FIRST_ELEMENT_INDEX]
        # Update the dictionary with the relevant keys
        component_dict[FieldNames.COMPONENT_NAME] = components.get(JiraFields.NAME_SUBFIELD, DefaultValues.EMPTY_STRING)
        component_dict[FieldNames.COMPONENT_DESCRIPTION] = components.get(JiraFields.DESCRIPTION_SUBFIELD, DefaultValues.EMPTY_STRING)
    
    else:
        # Iterate over the components list
        for component in components:
            # Extract the 'name' field
            name_field = component.get(JiraFields.NAME_SUBFIELD, DefaultValues.EMPTY_STRING)
            
            # Split the 'name' field into key and value (e.g., "category: Work")
            if COMPONENT_SPLITTER in name_field:
                key, value = name_field.split(ComponentsProcessing.COMPONENT_DELIMITER, 1)
                if key in ComponentsProcessing.COMPONENT_ATTRIBUTES.keys():
                    component_dict[ComponentsProcessing.COMPONENT_ATTRIBUTES[key]][ComponentsProcessing.COMPONENT_COLUMN_NAME] = value
    
    return component_dict