# Standard packages
from typing import Any, Dict, List, Tuple

from config import FieldNames, ProcessingConfig
from data_processor import calculate_duration

# Jira field constants
FIELDS_KEY = 'fields'
KEY_FIELD = 'key'
RESOLUTION_DATE_FIELD = 'resolutiondate'
ISSUE_LINKS_FIELD = 'issuelinks'
ISSUE_TYPE_FIELD = 'issuetype'
CREATED_FIELD = 'created'
FIX_VERSIONS_FIELD = 'fixVersions'
TIME_SPENT_FIELD = 'timespent'
NAME_FIELD = 'name'
VALUE_FIELD = 'value'
CHILD_FIELD = 'child'
DESCRIPTION_FIELD = 'description'
REQUEST_TYPE_FIELD = 'requestType'

# Custom field identifiers
REQUEST_TYPE_CUSTOM_FIELD = 'customfield_23641'
CCF_DATE_CUSTOM_FIELD = 'customfield_13454'
PROJECT_INITIATIVE_CUSTOM_FIELD = 'customfield_28846'

# Dictionary keys
PARENT_VALUE_KEY = 'parent_value'
CHILD_VALUE_KEY = 'child_value'

# DataFrame columns
NAME_COLUMN = 'Name'
DESCRIPTION_COLUMN = 'Description'

# Default values
UNKNOWN_VALUE = 'Unknown'
EMPTY_STRING = ''
EMPTY_LIST = []
EMPTY_DICT = {}
PROJECT_KEY_SEPARATOR = '-'
FIRST_ELEMENT_INDEX = 0
FALSE_RETURN = False

# Ticket field names
KEY_TICKET_FIELD = 'Key'
RESOLUTION_DATE_TICKET_FIELD = 'ResolutionDate'
PROJECT_ISSUE_TYPE_FIELD = 'project-issuetype'
CREATED_TICKET_FIELD = 'Created'
HAS_ISSUES_FIELD = 'HasIssues'
CCF_DATE_TICKET_FIELD = 'CCFDate'
TICKET_DURATION_FIELD = 'TicketDuration'
FIX_VERSION_TICKET_FIELD = 'FixVersion'
TIME_SPENT_SECONDS_FIELD = 'TimespentSeconds'
COMPONENT_NAME_FIELD = 'ComponentName'
COMPONENT_DESCRIPTION_FIELD = 'ComponentDescription'


def extract_name_description_pairs(filtered_df):
    """Extract Name and Description pairs from the DataFrame."""
    return filtered_df[[NAME_COLUMN, DESCRIPTION_COLUMN]].to_records(index=False)

def extract_request_type(entry):
    """
    Extract the request type from the entry safely.
    """
    fields = entry.get(FIELDS_KEY, EMPTY_DICT)
    custom_field = fields.get(REQUEST_TYPE_CUSTOM_FIELD, EMPTY_DICT)
    
    # Ensure custom_field is a dictionary before accessing 'requestType'
    if isinstance(custom_field, dict):
        return custom_field.get(REQUEST_TYPE_FIELD, EMPTY_DICT)
    
    # Return an empty dictionary if custom_field is None or not a dictionary
    return EMPTY_DICT

def _extract_entry_fields(entry: Dict[str, Any]) -> Tuple[str, str, bool, str, Dict[str, str], str, str, List[Dict[str, Any]], int]:
    """Extract basic fields from entry."""
    entry_key = entry.get(KEY_FIELD, None)
    resolution_date = entry.get(FIELDS_KEY, EMPTY_DICT).get(RESOLUTION_DATE_FIELD, None)
    has_issues = bool(entry.get(FIELDS_KEY, EMPTY_DICT).get(ISSUE_LINKS_FIELD, EMPTY_LIST))
    issue_type_field = entry.get(FIELDS_KEY, EMPTY_DICT).get(ISSUE_TYPE_FIELD, None)
    issue_type = issue_type_field[NAME_FIELD] if isinstance(issue_type_field, dict) and NAME_FIELD in issue_type_field else UNKNOWN_VALUE
    project_initiative = extract_project_initiative(entry)
    created_date = entry.get(FIELDS_KEY, EMPTY_DICT).get(CREATED_FIELD, None)
    ccf_date = entry.get(FIELDS_KEY, EMPTY_DICT).get(CCF_DATE_CUSTOM_FIELD, None)
    fix_version = entry.get(FIELDS_KEY, EMPTY_DICT).get(FIX_VERSIONS_FIELD, EMPTY_LIST)
    timespent_seconds = entry.get(FIELDS_KEY, EMPTY_DICT).get(TIME_SPENT_FIELD, None)
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
        KEY_TICKET_FIELD: entry_key,
        RESOLUTION_DATE_TICKET_FIELD: resolution_date,
        FieldNames.PROJECT: entry_key.split(PROJECT_KEY_SEPARATOR)[FIRST_ELEMENT_INDEX],
        FieldNames.ISSUE_TYPE: issue_type,
        PROJECT_ISSUE_TYPE_FIELD: f"{entry_key.split(PROJECT_KEY_SEPARATOR)[FIRST_ELEMENT_INDEX]}{PROJECT_KEY_SEPARATOR}{issue_type}",
        CREATED_TICKET_FIELD: created_date,
        HAS_ISSUES_FIELD: has_issues,
        CCF_DATE_TICKET_FIELD: ccf_date,
        TICKET_DURATION_FIELD: ticket_duration,
        FIX_VERSION_TICKET_FIELD: fix_version[FIRST_ELEMENT_INDEX][NAME_FIELD] if fix_version else None,
        TIME_SPENT_SECONDS_FIELD: timespent_seconds,
        FieldNames.PROJECT_INITIATIVE_L1_COLUMN: project_initiative[PARENT_VALUE_KEY],
        FieldNames.PROJECT_INITIATIVE_L2_COLUMN: project_initiative[CHILD_VALUE_KEY]
    }
    return ticket_values


def extract_project_initiative(entry):
    """Extract the project and initiative from a JSON entry."""
    
    if entry is None:
        # Handle the case where entry is None
        return {PARENT_VALUE_KEY: EMPTY_STRING, CHILD_VALUE_KEY: EMPTY_STRING}
    
    # Safely access nested fields with .get()
    fields = entry.get(FIELDS_KEY, EMPTY_DICT)
    custom_field = fields.get(PROJECT_INITIATIVE_CUSTOM_FIELD, EMPTY_DICT)
    if not isinstance(custom_field, dict):
        # If custom_field is not a dictionary, return an empty dictionary
        return {PARENT_VALUE_KEY: EMPTY_STRING, CHILD_VALUE_KEY: EMPTY_STRING}
    
    # Extract the parent value
    parent_value = custom_field.get(VALUE_FIELD, None)
    
    # Extract the child value if it exists
    child = custom_field.get(CHILD_FIELD, EMPTY_DICT)
    child_value = child.get(VALUE_FIELD, None) if isinstance(child, dict) else None
    
    # Return both parent and child values in a dictionary
    return {
        PARENT_VALUE_KEY: parent_value,
        CHILD_VALUE_KEY: child_value
    }

def process_components(
    components: List[Dict[str, Any]]
) -> None:
    """
    Process changelog entries and calculate status durations.
    """
    if components == EMPTY_LIST:
        return EMPTY_DICT
    
    components = components[FIRST_ELEMENT_INDEX]
    name = components.get(NAME_FIELD, None)
    
    component_dict = {
        COMPONENT_NAME_FIELD: name,
        COMPONENT_DESCRIPTION_FIELD: components.get(DESCRIPTION_FIELD, None)
    }
    
    return component_dict