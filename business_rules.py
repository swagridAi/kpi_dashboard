# Constants
PROJECT_ISSUE_TYPE_COLUMN = "project-issuetype"
COMPONENT_NAME_COLUMN = "ComponentName"
PANDAS_COLUMN_AXIS = 1

def _is_row_allowed(row, project_issues_to_check):
    project_issuetype = row[PROJECT_ISSUE_TYPE_COLUMN]
    component_name = row[COMPONENT_NAME_COLUMN]
    
    # If the project-issuetype is not in project_issues_to_check, keep the row
    if project_issuetype not in project_issues_to_check:
        return True
    
    # If the project-issuetype is in project_issues_to_check, check if the component_name is allowed
    if component_name in project_issues_to_check[project_issuetype]:
        return True
    
    # Otherwise, exclude the row
    return False


def filter_matched_entries(matched_entries, project_issues_to_check):
    """
    Filters the matched_entries DataFrame based on allowed_components for specific project-issuetypes.
    
    Args:
        matched_entries (pd.DataFrame): The DataFrame containing project-issuetype and ComponentName columns.
        project_issues_to_check (dict): A dictionary specifying allowed ComponentName values for certain project-issuetypes.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Generate the boolean mask
    mask = matched_entries.apply(lambda row: _is_row_allowed(row, project_issues_to_check), axis=PANDAS_COLUMN_AXIS)
    
    # Apply the filtering logic
    filtered_entries = matched_entries.loc[mask]
    
    return filtered_entries