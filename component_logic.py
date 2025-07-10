def apply_component_mapping(filtered_df, component_attributes):
    """
    Applies the mapping functionality to transform the filtered rows.
    
    Args:
        filtered_df (pd.DataFrame): Filtered data frame containing rows to transform.
        component_attributes (dict): Dictionary defining the mapping for components.
    
    Returns:
        pd.DataFrame: Updated data frame with transformed values.
    """
    for component, attributes in component_attributes.items():
        target_column = attributes.get(ComponentsProcessing.COMPONENT_COLUMN_NAME)
        source_column = attributes.get(ComponentsProcessing.COMPONENT_COLUMN_MAP)
        
        if source_column and target_column in filtered_df.columns:
            # Move values from source_column to target_column
            filtered_df[target_column] = filtered_df[source_column]
        elif not source_column:
            # Skip components without a source column mapping
            continue
        else:
            raise KeyError(f"Source column '{source_column}' not found in the data frame.")
    
    return filtered_df


"""
This module processes and transforms component data from Abhi's projects
into column values suitable for integration with another program.
"""

from config import FieldNames, ComponentsProcessing

import pandas as pd


def process_component_data(df, project_identifiers = ComponentsProcessing.PROJECT_COMPONENT_FIELDS, component_attributes = ComponentsProcessing.COMPONENT_ATTRIBUTES):
    """
    Processes and transforms component data for integration.
    
    Args:
        df (pd.DataFrame): Input data frame.
        project_identifier (str): Identifier for filtering project-specific tickets.
        column_mapping (dict): Mapping of source columns to target columns for transformation.
    
    Returns:
        pd.DataFrame: Updated data frame with transformed values.
    """
    if component_attributes is None:
        raise ValueError("Column mapping must be provided for transformations.")
    
    # Step 2: Filter rows to include only tickets belonging to the specified project.
    filtered_df = df[df[FieldNames.PROJECT].isin(project_identifiers)].copy()
    
    # Step 3: Apply the required transformation logic to the filtered rows.
    filtered_df = apply_component_mapping(filtered_df, component_attributes)
    
    # Step 4: Return the updated data frame.
    return filtered_df