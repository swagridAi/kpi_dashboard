def filter_matched_entries(matched_entries, project_issues_to_check):
    """
    Filters the matched_entries DataFrame based on allowed_components for specific project-issuetypes.
    
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