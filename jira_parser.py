# Standard packages
from typing import Any, Dict, List, Tuple

from config import FieldNames, ProcessingConfig
from data_processor import calculate_duration


def extract_name_description_pairs(filtered_df):
    """Extract Name and Description pairs from the DataFrame."""
    return filtered_df[['Name', 'Description']].to_records(index=False)

def extract_request_type(entry):
    """
    Extract the request type from the entry safely.
    """
    fields = entry.get('fields', {})
    custom_field = fields.get('customfield_23641', {})
    
    # Ensure custom_field is a dictionary before accessing 'requestType'
    if isinstance(custom_field, dict):
        return custom_field.get('requestType', {})
    
    # Return an empty dictionary if custom_field is None or not a dictionary
    return {}

def get_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract fields from the entry.
    """
    entry_key = entry.get('key', None)
    resolution_date = entry.get('fields', {}).get('resolutiondate', None)
    has_issues = bool(entry.get('fields', {}).get('issuelinks', []))
    issue_type_field = entry.get('fields', {}).get('issuetype', None)
    issue_type = issue_type_field['name'] if isinstance(issue_type_field, dict) and 'name' in issue_type_field else 'Unknown'
    project_initiative = extract_project_initiative(entry)
    created_date = entry.get('fields', {}).get('created', None)
    ccf_date = entry.get('fields', {}).get('customfield_13454', None)
    fix_version = entry.get('fields', {}).get('fixVersions', [])
    timespent_seconds = entry.get('fields', {}).get('timespent', None)
    
    if not resolution_date:
        return False
    
    ticket_duration = calculate_duration(created_date, resolution_date)
    
    ticket_values = {
        'Key': entry_key,
        'ResolutionDate': resolution_date,
        FieldNames.PROJECT: entry_key.split('-')[0],
        FieldNames.ISSUE_TYPE: issue_type,
        'project-issuetype': f"{entry_key.split('-')[0]}-{issue_type}",
        'Created': created_date,
        'HasIssues': has_issues,
        'CCFDate': ccf_date,
        'TicketDuration': ticket_duration,
        'FixVersion': fix_version[0]['name'] if fix_version else None,
        'TimespentSeconds': timespent_seconds,
        FieldNames.PROJECT_INITIATIVE_L1_COLUMN: project_initiative["parent_value"],
        FieldNames.PROJECT_INITIATIVE_L2_COLUMN: project_initiative["child_value"]
    }
    return ticket_values


def extract_project_initiative(entry):
    """Extract the project and initiative from a JSON entry."""
    
    if entry is None:
        # Handle the case where entry is None
        return {'parent_value': '', 'child_value': ''}
    
    # Safely access nested fields with .get()
    fields = entry.get('fields', {})
    custom_field = fields.get('customfield_28846', {})
    if not isinstance(custom_field, dict):
        # If custom_field is not a dictionary, return an empty dictionary
        return {'parent_value': '', 'child_value': ''}
    
    # Extract the parent value
    parent_value = custom_field.get('value', None)
    
    # Extract the child value if it exists
    child = custom_field.get('child', {})
    child_value = child.get('value', None) if isinstance(child, dict) else None
    
    # Return both parent and child values in a dictionary
    return {
        'parent_value': parent_value,
        'child_value': child_value
    }

def process_components(
    components: List[Dict[str, Any]]
) -> None:
    """
    Process changelog entries and calculate status durations.
    """
    if components == []:
        return {}
    
    components = components[0]
    name = components.get('name', None)
    
    component_dict = {
        'ComponentName': name,
        'ComponentDescription': components.get('description', None)
    }
    
    return component_dict