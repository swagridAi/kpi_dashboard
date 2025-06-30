# Standard packages
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
import pandas as pd
import numpy as np

from config import FieldNames, ProcessingConfig, JiraFields, DefaultValues, DataFrameColumns
from data_processor import calculate_duration


def extract_name_description_pairs(filtered_df: pd.DataFrame) -> np.ndarray:
    """
    Extract Name and Description pairs from a filtered DataFrame for JIRA matching operations.
    
    This function is used to create name-description pairs that will be used to match
    JIRA entries against predefined criteria in the mapping logic.
    
    Args:
        filtered_df (pd.DataFrame): DataFrame containing at minimum 'Name' and 'Description' columns
        
    Returns:
        np.ndarray: Array of tuples containing (Name, Description) pairs
        
    Example:
        >>> df = pd.DataFrame({'Name': ['Issue1', 'Issue2'], 'Description': ['Desc1', 'Desc2']})
        >>> pairs = extract_name_description_pairs(df)
        >>> print(pairs)
        [('Issue1', 'Desc1'), ('Issue2', 'Desc2')]
    """
    return filtered_df[[DataFrameColumns.NAME, DataFrameColumns.DESCRIPTION]].to_records(index=False)


# ========================================================================
# GENERIC UTILITY FUNCTIONS FOR SAFE DATA ACCESS
# ========================================================================

def _safe_get_nested_field(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely traverse nested dictionary structure with multiple keys.
    
    This is a defensive programming utility that prevents KeyError exceptions
    when accessing deeply nested JIRA API response data. It's essential because
    JIRA API responses can have inconsistent structure depending on field configuration.
    
    Args:
        data (Dict[str, Any]): Source dictionary to traverse
        *keys (str): Sequence of keys to traverse in order (e.g., 'fields', 'customfield_123')
        default (Any, optional): Value to return if any key in the path is missing. Defaults to None.
        
    Returns:
        Any: Value found at the nested path, or default value if path doesn't exist
        
    Raises:
        None: This function is designed to never raise exceptions
        
    Example:
        >>> data = {'fields': {'customfield_123': {'value': 'test'}}}
        >>> result = _safe_get_nested_field(data, 'fields', 'customfield_123', 'value')
        >>> print(result)  # 'test'
        >>> 
        >>> result = _safe_get_nested_field(data, 'fields', 'missing_field', default='fallback')
        >>> print(result)  # 'fallback'
    """
    current: Any = data
    
    # Traverse each key in the path sequentially
    for key in keys:
        # Defensive check: ensure current level is a dictionary and contains the key
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current


def _safe_get_subfield(data: Dict[str, Any], subfield: str, default: Any = None) -> Any:
    """
    Safely extract a subfield from a dictionary with comprehensive type checking.
    
    This function handles the common pattern of accessing dictionary values where
    the source dictionary might be None, not a dictionary, or missing the key.
    
    Args:
        data (Dict[str, Any]): Source dictionary that may contain the subfield
        subfield (str): Key name to extract from the dictionary
        default (Any, optional): Value to return if extraction fails. Defaults to None.
        
    Returns:
        Any: Value of the subfield or default if not found/invalid
        
    Note:
        This function is more forgiving than dict.get() because it handles
        cases where data itself is not a dictionary.
    """
    if not isinstance(data, dict):
        return default
    return data.get(subfield, default)


def _safe_get_first_element(data: List[Any], default: Any = None) -> Any:
    """
    Safely extract the first element from a list with bounds checking.
    
    JIRA often returns arrays where we only need the first element (e.g., fixVersions).
    This function handles cases where the list might be empty, None, or not a list.
    
    Args:
        data (List[Any]): Source list that may contain elements
        default (Any, optional): Value to return if list is empty/invalid. Defaults to None.
        
    Returns:
        Any: First element of the list or default if list is empty/invalid
        
    Business Logic:
        - fixVersions: Usually contains multiple versions, but we only use the first
        - components: May contain multiple components, but business rules specify first only
    """
    if not isinstance(data, list) or not data:
        return default
    return data[ProcessingConfig.FIRST_ELEMENT_INDEX]


def _validate_dict_type(data: Any) -> bool:
    """
    Validate that data is a dictionary type for safe field access.
    
    This validation is critical because JIRA API can return different data types
    for the same field depending on configuration or data state.
    
    Args:
        data (Any): Data of unknown type to validate
        
    Returns:
        bool: True if data is a dictionary and can be safely accessed, False otherwise
        
    Note:
        Used before attempting dictionary access operations to prevent TypeError.
    """
    return isinstance(data, dict)


def _get_fields_container(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the 'fields' container from a JIRA entry with safe access.
    
    The 'fields' container holds all the main JIRA field data. This function
    provides a standardized way to access it safely across all extraction functions.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        Dict[str, Any]: Fields dictionary or empty dict if not found
        
    Business Context:
        JIRA API structure: {key: 'PROJ-123', fields: {resolutiondate: '...', ...}}
    """
    return _safe_get_subfield(entry, JiraFields.FIELDS, {})


# ========================================================================
# CUSTOM FIELD ACCESS FUNCTIONS
# ========================================================================

def _extract_custom_field(entry: Dict[str, Any], custom_field_id: str) -> Dict[str, Any]:
    """
    Generic function to safely extract any custom field from a JIRA entry.
    
    Custom fields in JIRA have numeric IDs that can vary between environments.
    This function provides a standardized way to extract them with proper error handling.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        custom_field_id (str): The custom field identifier (e.g., 'customfield_23641')
        
    Returns:
        Dict[str, Any]: Custom field dictionary or empty dict if not found/invalid
        
    Business Context:
        - customfield_23641: Service desk request type information
        - customfield_13454: CCF (Change Control Form) date
        - customfield_28846: Project initiative hierarchy
        
    Note:
        Custom field IDs are environment-specific and should be configured in constants.
    """
    if entry is None:
        return {}
    
    # Navigate to: entry['fields']['customfield_xxxxx']
    custom_field: Any = _safe_get_nested_field(entry, JiraFields.FIELDS, custom_field_id, {})
    
    # Custom fields must be dictionaries to be useful; some may be strings or null
    return custom_field if _validate_dict_type(custom_field) else {}


def _extract_subfield_from_custom_field(custom_field: Dict[str, Any], subfield: str) -> Any:
    """
    Extract a specific subfield from a custom field dictionary.
    
    Custom fields often have nested structures with specific subfields like 'value',
    'displayName', 'name', etc. This function standardizes access to these subfields.
    
    Args:
        custom_field (Dict[str, Any]): Custom field dictionary already extracted and validated
        subfield (str): Name of the subfield to extract (e.g., 'value', 'name')
        
    Returns:
        Any: Subfield value or None if not found
        
    Common Subfields:
        - 'value': Primary value of the custom field
        - 'displayName': Human-readable name
        - 'name': Internal name identifier
        - 'child': Nested child object (for hierarchical fields)
    """
    return _safe_get_subfield(custom_field, subfield, None)


# ========================================================================
# FIELD VALUE EXTRACTION FUNCTIONS
# ========================================================================

def _extract_standard_field_value(entry: Dict[str, Any], field_name: str) -> Any:
    """
    Extract a standard JIRA field value from the entry's fields container.
    
    Standard fields are core JIRA fields that exist in all installations,
    unlike custom fields which are organization-specific.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        field_name (str): Name of the standard JIRA field to extract
        
    Returns:
        Any: Field value or None if not found
        
    Standard Fields Typically Extracted:
        - resolutiondate: When the issue was resolved
        - created: When the issue was created
        - issuetype: Type of issue (Bug, Story, etc.)
        - fixVersions: Array of fix version objects
        - timespent: Time spent in seconds
        - issuelinks: Array of linked issues
    """
    return _safe_get_nested_field(entry, JiraFields.FIELDS, field_name, None)


def _extract_name_from_object(obj: Any, default: str = DefaultValues.UNKNOWN) -> str:
    """
    Extract 'name' field from any JIRA object with standardized fallback handling.
    
    Many JIRA objects (issue types, components, versions, users) have a 'name' field.
    This function provides consistent extraction with business-appropriate defaults.
    
    Args:
        obj (Any): JIRA object that may contain a 'name' field
        default (str, optional): Default value if name not found. Defaults to 'Unknown'.
        
    Returns:
        str: Name value or default if not found
        
    Common Use Cases:
        - Issue type objects: {'name': 'Bug', 'description': '...'}
        - Component objects: {'name': 'Frontend', 'description': '...'}
        - Fix version objects: {'name': 'v1.2.3', 'released': true}
        
    Business Logic:
        'Unknown' is used as default because it's meaningful in business reports,
        whereas None or empty string would cause display issues.
    """
    if _validate_dict_type(obj) and JiraFields.NAME_SUBFIELD in obj:
        return obj[JiraFields.NAME_SUBFIELD]
    return default


def _extract_value_from_nested_object(obj: Any, *path: str) -> Optional[str]:
    """
    Extract a value from a nested object structure using a flexible path specification.
    
    This function handles complex nested structures like project initiative hierarchies
    where data is nested multiple levels deep.
    
    Args:
        obj (Any): Source object to traverse
        *path (str): Variable number of path elements to traverse in sequence
        
    Returns:
        Optional[str]: Value found at the specified path, or None if path doesn't exist
        
    Example Usage:
        >>> obj = {'child': {'value': 'test'}}
        >>> result = _extract_value_from_nested_object(obj, 'child', 'value')
        >>> print(result)  # 'test'
        
    Business Context:
        Project initiative structure: {value: 'Parent', child: {value: 'Child'}}
        This allows extraction of both parent and child values using different paths.
    """
    current: Any = obj
    
    # Traverse each path element sequentially
    for key in path:
        if not _validate_dict_type(current) or key not in current:
            return None
        current = current[key]
    
    return current


# ========================================================================
# BUSINESS LOGIC FUNCTIONS
# ========================================================================

def extract_request_type(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract service desk request type information from a JIRA entry.
    
    Service desk request types provide additional context about how the issue
    was created and what type of service request it represents.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        Dict[str, Any]: Request type dictionary containing name, description, etc.
                       Returns empty dict if request type not found or invalid.
        
    Business Context:
        Service desk issues have request types like 'Bug Report', 'Feature Request',
        'Access Request', etc. This information is used for categorization and reporting.
        
    Expected Structure:
        customfield_23641: {
            requestType: {
                name: 'Bug Report',
                description: 'Report a software bug',
                ...
            }
        }
    """
    custom_field: Dict[str, Any] = _extract_custom_field(entry, JiraFields.REQUEST_TYPE_CUSTOM_FIELD)
    return _safe_get_subfield(custom_field, JiraFields.REQUEST_TYPE_SUBFIELD, {})


def _extract_basic_jira_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic JIRA fields that require minimal processing into a standardized dictionary.
    
    These are core fields that are used across multiple business logic functions
    and benefit from being extracted once and reused.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        Dict[str, Any]: Dictionary containing basic field values with standardized keys
        
    Extracted Fields:
        - Key: JIRA issue key (e.g., 'PROJ-123')
        - ResolutionDate: When issue was resolved (ISO timestamp)
        - Created: When issue was created (ISO timestamp)
        - CCFDate: Change Control Form date (business-specific)
        - TimespentSeconds: Time logged in seconds
        
    Note:
        All values may be None if the corresponding field is not set in JIRA.
    """
    return {
        FieldNames.KEY: _safe_get_subfield(entry, JiraFields.KEY, None),
        FieldNames.RESOLUTION_DATE: _extract_standard_field_value(entry, JiraFields.RESOLUTION_DATE),
        FieldNames.CREATED: _extract_standard_field_value(entry, JiraFields.CREATED),
        FieldNames.CCF_DATE: _extract_standard_field_value(entry, JiraFields.CCF_DATE_CUSTOM_FIELD),
        FieldNames.TIME_SPENT_SECONDS: _extract_standard_field_value(entry, JiraFields.TIME_SPENT)
    }


def _extract_issue_type_name(entry: Dict[str, Any]) -> str:
    """
    Extract the issue type name from a JIRA entry with business-appropriate fallback.
    
    Issue type determines how the issue is categorized and processed in business rules.
    Different issue types may have different SLA requirements and processing workflows.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        str: Issue type name (e.g., 'Bug', 'Story', 'Task') or 'Unknown' if not found
        
    Business Context:
        Issue types affect KPI calculations:
        - 'Data Quality Rule': Uses special close date calculation logic
        - 'Consumer Validation': Has specific component filtering rules
        - 'PFI Sub-Task': Subject to testing-specific business rules
    """
    issue_type_field: Any = _extract_standard_field_value(entry, JiraFields.ISSUE_TYPE)
    return _extract_name_from_object(issue_type_field)


def _extract_fix_version_name(entry: Dict[str, Any]) -> Optional[str]:
    """
    Extract the name of the first fix version from a JIRA entry.
    
    Fix versions indicate which software release will contain the fix for this issue.
    Business rules specify using only the first fix version for calculations.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        Optional[str]: Fix version name (e.g., 'v1.2.3', 'Sprint 24') or None if not set
        
    Business Logic:
        - Only the first fix version is used even if multiple are set
        - Fix version names are used for close date calculation in specific project types
        - Format typically includes month abbreviation and year: 'Release Mar25'
        
    Note:
        Some issues may have multiple fix versions, but business rules require
        using only the first one for consistency in reporting.
    """
    fix_versions: Any = _extract_standard_field_value(entry, JiraFields.FIX_VERSIONS)
    first_version: Any = _safe_get_first_element(fix_versions)
    return _extract_name_from_object(first_version, None) if first_version else None


def _check_has_issue_links(entry: Dict[str, Any]) -> bool:
    """
    Determine if a JIRA entry has any linked issues for issue rate calculations.
    
    Issue links indicate relationships between issues (blocks, is blocked by, relates to, etc.).
    The presence of links is used as a proxy indicator for issue complexity.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        bool: True if the entry has any issue links, False otherwise
        
    Business Context:
        Issues with links are considered to have higher complexity and may indicate:
        - Dependencies between work items
        - Related bugs or improvements
        - Blocked or blocking relationships
        
    KPI Impact:
        This boolean is used in issue rate calculations to determine what percentage
        of resolved issues had complications (indicated by having links).
    """
    issue_links: Any = _extract_standard_field_value(entry, JiraFields.ISSUE_LINKS)
    return bool(issue_links)  # Empty list evaluates to False, non-empty to True


def _extract_project_key_from_entry_key(entry_key: str) -> str:
    """
    Extract the project key portion from a full JIRA issue key.
    
    JIRA keys follow the format 'PROJECTKEY-NUMBER' (e.g., 'MYPROJ-123').
    The project key is used for categorization and business rule application.
    
    Args:
        entry_key (str): Full JIRA issue key (e.g., 'MYPROJ-123')
        
    Returns:
        str: Project key portion (e.g., 'MYPROJ')
        
    Business Rules:
        Different projects have different:
        - SLA requirements
        - Component filtering rules
        - Close date calculation methods
        - KPI targets
        
    Example:
        'DQMMBAU-123' -> 'DQMMBAU' (Data Quality project)
        'PMCIRIS-456' -> 'PMCIRIS' (PMC Iris project)
    """
    return entry_key.split(DefaultValues.PROJECT_SEPARATOR)[ProcessingConfig.PROJECT_KEY_INDEX]


def _build_project_issue_type_key(project_key: str, issue_type: str) -> str:
    """
    Build a composite key combining project and issue type for business rule matching.
    
    This composite key is used throughout the system to apply project-issue-type
    specific business rules and filtering logic.
    
    Args:
        project_key (str): The project identifier (e.g., 'DQMMBAU')
        issue_type (str): The issue type name (e.g., 'Data Quality Rule')
        
    Returns:
        str: Composite key in format 'PROJECT-IssueType'
        
    Business Context:
        Business rules are defined at the project-issuetype level because:
        - Same issue type in different projects may have different rules
        - Component filtering varies by project-issuetype combination
        - Close date calculation methods depend on both project and issue type
        
    Examples:
        'DQMMBAU' + 'Data Quality Rule' -> 'DQMMBAU-Data Quality Rule'
        'PMCIRIS' + 'PFI Sub-Task' -> 'PMCIRIS-PFI Sub-Task'
    """
    return f"{project_key}{DefaultValues.PROJECT_SEPARATOR}{issue_type}"


def _validate_required_fields(basic_fields: Dict[str, Any]) -> bool:
    """
    Validate that all business-critical fields are present for further processing.
    
    This validation ensures that we only process JIRA entries that have sufficient
    data for meaningful business analysis and KPI calculations.
    
    Args:
        basic_fields (Dict[str, Any]): Dictionary containing basic field values
        
    Returns:
        bool: True if all required fields are present and valid, False otherwise
        
    Required Fields:
        - ResolutionDate: Must be present for KPI calculations
        
    Business Logic:
        Issues without resolution dates cannot be included in:
        - Throughput calculations
        - Cycle time analysis
        - Monthly reporting
        - SLA compliance reporting
        
    TODO: Consider adding validation for other critical fields like Created date
    """
    # Resolution date is mandatory for all KPI calculations
    return bool(basic_fields.get(FieldNames.RESOLUTION_DATE))


def _build_ticket_values_dictionary(
    basic_fields: Dict[str, Any],
    issue_type: str,
    project_key: str,
    has_issues: bool,
    ticket_duration: Any,
    fix_version: Optional[str],
    project_initiative: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Construct the complete ticket values dictionary with all processed data.
    
    This function assembles all extracted and calculated values into the final
    standardized format used throughout the rest of the processing pipeline.
    
    Args:
        basic_fields (Dict[str, Any]): Basic extracted field values
        issue_type (str): Processed issue type name
        project_key (str): Extracted project identifier
        has_issues (bool): Whether the ticket has issue links
        ticket_duration (Any): Calculated duration from created to resolved
        fix_version (Optional[str]): Fix version name if present
        project_initiative (Dict[str, Any]): Project initiative hierarchy data
        
    Returns:
        Dict[str, Any]: Complete standardized ticket data dictionary
        
    Output Schema:
        The returned dictionary contains all fields needed for:
        - KPI calculations (duration, dates, categorization)
        - Business rule application (project, issue type, components)
        - Reporting and analysis (initiative hierarchy, metadata)
        
    Business Impact:
        This standardized format ensures consistency across all downstream
        processing functions and enables reliable data analysis.
    """
    return {
        # Core identifiers
        FieldNames.KEY: basic_fields[FieldNames.KEY],
        FieldNames.PROJECT: project_key,
        FieldNames.ISSUE_TYPE: issue_type,
        FieldNames.PROJECT_ISSUE_TYPE: _build_project_issue_type_key(project_key, issue_type),
        
        # Date fields for KPI calculations
        FieldNames.RESOLUTION_DATE: basic_fields[FieldNames.RESOLUTION_DATE],
        FieldNames.CREATED: basic_fields[FieldNames.CREATED],
        FieldNames.CCF_DATE: basic_fields[FieldNames.CCF_DATE],
        
        # Calculated metrics
        FieldNames.TICKET_DURATION: ticket_duration,
        FieldNames.HAS_ISSUES: has_issues,
        
        # Version and time tracking
        FieldNames.FIX_VERSION: fix_version,
        FieldNames.TIME_SPENT_SECONDS: basic_fields[FieldNames.TIME_SPENT_SECONDS],
        
        # Project hierarchy for reporting
        FieldNames.PROJECT_INITIATIVE_L1_COLUMN: project_initiative[FieldNames.PARENT_VALUE],
        FieldNames.PROJECT_INITIATIVE_L2_COLUMN: project_initiative[FieldNames.CHILD_VALUE]
    }


def get_fields(entry: Dict[str, Any]) -> Union[Dict[str, Any], bool]:
    """
    Main orchestration function to extract and process all relevant fields from a JIRA entry.
    
    This is the primary entry point for converting raw JIRA API data into the standardized
    format used throughout the KPI calculation system. It coordinates all field extraction
    and applies business logic transformations.
    
    Args:
        entry (Dict[str, Any]): Raw JIRA entry dictionary from API response
        
    Returns:
        Union[Dict[str, Any], bool]: 
            - Dict containing processed ticket values if successful
            - False if validation fails (missing required fields)
            
    Raises:
        None: Function is designed to handle all error cases gracefully
        
    Processing Steps:
        1. Extract basic fields (key, dates, etc.)
        2. Validate required fields are present
        3. Extract complex fields (issue type, components, etc.)
        4. Calculate derived values (duration, composite keys)
        5. Assemble final standardized dictionary
        
    Business Context:
        This function is called for every JIRA issue in the dataset. Performance
        and reliability are critical since it processes thousands of records.
        
    Error Handling:
        Returns False for issues that cannot be processed, allowing the caller
        to skip invalid records rather than failing the entire batch.
        
    Example:
        >>> jira_entry = {'key': 'PROJ-123', 'fields': {...}}
        >>> result = get_fields(jira_entry)
        >>> if result:
        ...     print(f"Processed ticket {result['Key']}")
        ... else:
        ...     print("Skipped invalid entry")
    """
    # Step 1: Extract basic fields that are used across multiple operations
    basic_fields: Dict[str, Any] = _extract_basic_jira_fields(entry)
    
    # Step 2: Early validation to avoid processing incomplete records
    if not _validate_required_fields(basic_fields):
        return False  # Skip records without required fields
    
    # Step 3: Extract complex fields that require specialized logic
    issue_type: str = _extract_issue_type_name(entry)
    project_key: str = _extract_project_key_from_entry_key(basic_fields[FieldNames.KEY])
    has_issues: bool = _check_has_issue_links(entry)
    fix_version: Optional[str] = _extract_fix_version_name(entry)
    project_initiative: Dict[str, Any] = extract_project_initiative(entry)
    
    # Step 4: Calculate derived values that depend on multiple fields
    ticket_duration: Any = calculate_duration(
        basic_fields[FieldNames.CREATED], 
        basic_fields[FieldNames.RESOLUTION_DATE]
    )
    
    # Step 5: Assemble final standardized dictionary
    return _build_ticket_values_dictionary(
        basic_fields=basic_fields,
        issue_type=issue_type,
        project_key=project_key,
        has_issues=has_issues,
        ticket_duration=ticket_duration,
        fix_version=fix_version,
        project_initiative=project_initiative
    )


def _create_default_project_initiative() -> Dict[str, str]:
    """
    Factory function to create a default project initiative dictionary with empty values.
    
    Used when project initiative data is missing or invalid to ensure consistent
    data structure throughout the processing pipeline.
    
    Returns:
        Dict[str, str]: Dictionary with empty string values for parent and child initiatives
        
    Business Context:
        Project initiatives are used for portfolio-level reporting. When missing,
        empty strings are preferred over None values to avoid display issues in reports.
    """
    return {
        FieldNames.PARENT_VALUE: DefaultValues.EMPTY_STRING, 
        FieldNames.CHILD_VALUE: DefaultValues.EMPTY_STRING
    }


def extract_project_initiative(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract project initiative hierarchy information from a JIRA entry.
    
    Project initiatives represent a two-level hierarchy used for portfolio management
    and reporting. The hierarchy typically represents business units and specific projects.
    
    Args:
        entry (Dict[str, Any]): JIRA entry dictionary from API response
        
    Returns:
        Dict[str, str]: Dictionary containing 'parent_value' and 'child_value' keys
        
    Data Structure:
        customfield_28846: {
            value: 'Business Unit A',     # Parent level
            child: {
                value: 'Project Alpha'    # Child level
            }
        }
        
    Business Context:
        - Parent value: Usually represents business unit or department
        - Child value: Represents specific project or initiative within the unit
        - Used for portfolio-level KPI reporting and resource allocation analysis
        
    Error Handling:
        Returns empty strings for both values if data is missing or malformed,
        ensuring consistent data structure for downstream processing.
        
    Example:
        >>> entry = {'fields': {'customfield_28846': {'value': 'IT', 'child': {'value': 'CRM'}}}}
        >>> result = extract_project_initiative(entry)
        >>> print(result)
        {'parent_value': 'IT', 'child_value': 'CRM'}
    """
    if entry is None:
        return _create_default_project_initiative()
    
    # Extract the project initiative custom field
    custom_field: Dict[str, Any] = _extract_custom_field(entry, JiraFields.PROJECT_INITIATIVE_CUSTOM_FIELD)
    
    if not custom_field:
        return _create_default_project_initiative()
    
    # Extract parent value (direct value field)
    parent_value: Optional[str] = _extract_subfield_from_custom_field(custom_field, JiraFields.VALUE_SUBFIELD)
    
    # Extract child value (nested under 'child' object)
    child_value: Optional[str] = _extract_value_from_nested_object(
        custom_field, 
        JiraFields.CHILD_SUBFIELD, 
        JiraFields.VALUE_SUBFIELD
    )
    
    return {
        FieldNames.PARENT_VALUE: parent_value,
        FieldNames.CHILD_VALUE: child_value
    }


def _build_component_dictionary(name: Optional[str], description: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Factory function to build a standardized component dictionary.
    
    Centralizes the structure of component data to ensure consistency
    across the application and make future schema changes easier.
    
    Args:
        name (Optional[str]): Component name (e.g., 'Frontend', 'Backend')
        description (Optional[str]): Component description/purpose
        
    Returns:
        Dict[str, Optional[str]]: Standardized component dictionary
        
    Schema:
        {
            'ComponentName': str | None,
            'ComponentDescription': str | None
        }
    """
    return {
        FieldNames.COMPONENT_NAME: name,
        FieldNames.COMPONENT_DESCRIPTION: description
    }


def process_components(components: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """
    Process JIRA component list and extract the first component's metadata.
    
    JIRA issues can have multiple components, but business rules specify using
    only the first component for filtering and categorization purposes.
    
    Args:
        components (List[Dict[str, Any]]): List of component dictionaries from JIRA API
        
    Returns:
        Dict[str, Optional[str]]: Component dictionary with name and description,
                                 or empty dict if no components exist
                                 
    Business Rules:
        - Only the first component is processed (business requirement)
        - Component filtering is applied to specific project-issuetype combinations
        - Components determine which issues are included in KPI calculations
        
    Component Structure:
        Each component object contains:
        - name: Component identifier (used for filtering)
        - description: Human-readable description
        - id: Internal JIRA ID (not used in our processing)
        
    Example:
        >>> components = [
        ...     {'name': 'DQMMBAU - New DQ Rule', 'description': 'New data quality rules'},
        ...     {'name': 'DQMMBAU - Update DQ Rule', 'description': 'Updates to existing rules'}
        ... ]
        >>> result = process_components(components)
        >>> print(result)
        {'ComponentName': 'DQMMBAU - New DQ Rule', 'ComponentDescription': 'New data quality rules'}
        
    Empty List Handling:
        Returns empty dictionary when no components exist, which is handled
        appropriately by downstream business rule filters.
    """
    if not components:
        return {}  # No components to process
    
    # Business rule: Use only the first component
    first_component: Any = _safe_get_first_element(components)
    if not first_component:
        return {}  # Safety check in case list contains invalid elements
    
    # Extract name and description using standardized extraction functions
    component_name: Optional[str] = _extract_name_from_object(first_component, None)
    component_description: Optional[str] = _extract_subfield_from_custom_field(
        first_component, 
        JiraFields.DESCRIPTION_SUBFIELD
    )
    
    return _build_component_dictionary(component_name, component_description)