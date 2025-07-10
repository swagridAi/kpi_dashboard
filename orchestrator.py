import pandas as pd
from typing import List
from get_issue_data import get_issue_data  # Example function from get_issue_data
from jira_get_everything_api import get_jira_data  # Example function from jira_get_everything_api
import tempfile
import logging
import os
from config import FieldNames

# Constants for column names
CATEGORY_COLUMN = FieldNames.CATEGORY
PROJECT_KEY_COLUMN = FieldNames.PROJECT
ISSUE_TYPE_COLUMN = FieldNames.ISSUE_TYPE

# Path to the Excel file
MAPPING_LOGIC_FILE_PATH = "mapping_files\\mapping_logic.xlsx"
KPI_TARGET_FILE_PATH = "mapping_files\\kpi_mapping.xlsx"
CATEGORY_DEFINITIONS_FILE_PATH = "mapping_files\\category_definitions.xlsx"
KPI_DEFINITIONS_FILE_PATH = "mapping_files\\kpi_definitions.xlsx"
REQUESTOR_MAPPING_FILE_PATH = "mapping_files\\portfolio_requestor_mapping.xlsx"

# Ensure the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create file handler
file_handler = logging.FileHandler("logs/orchestrator.log", mode="w")
file_handler.setLevel(logging.INFO)

# Create stream handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add handlers to the root logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def read_excel_file(file_path: str) -> pd.DataFrame:
    """Reads an Excel file into a DataFrame."""
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise


def filter_data_by_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Filters rows where the specified column is not null."""
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the data.")
    return data[data[column].notna()]


def get_unique_values(data: pd.DataFrame, column: str) -> List:
    """Returns a list of unique values from the specified column."""
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the data.")
    return data[column].unique().tolist()


def get_mapping_logic(mapping_file_path):
    # Read the Excel file
    data = read_excel_file(mapping_file_path)

    # Filter rows where 'Category' is not null
    filtered_data = filter_data_by_column(data, CATEGORY_COLUMN)

    # Get unique values from 'Project Key' and 'Issue Type'
    unique_project_keys = get_unique_values(filtered_data, PROJECT_KEY_COLUMN)
    unique_issue_types = get_unique_values(filtered_data, ISSUE_TYPE_COLUMN)

    return filtered_data, unique_project_keys, unique_issue_types


def save_to_excel(dataframes_dict, file_name="output.xlsx"):
    """
    Save multiple DataFrames to an Excel file with each on a separate sheet.

    :param dataframes_dict: Dictionary where keys are sheet names and values are DataFrames.
    :param file_name: Name of the Excel file to save.
    """
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        for sheet_name, dataframe in dataframes_dict.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Data saved to {file_name}")


def normalize_dataframe(df):
    """
    Normalize a DataFrame by resetting the index, sorting columns, and stripping whitespace.
    Args:
        df (pd.DataFrame): The DataFrame to normalize.
    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    return (
        df.reset_index(drop=True)
        .sort_index(axis=1)
        .applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Handle NoneType
        .astype(str)
    )


def compare_and_highlight(new_dataframes, old_file, diff_file):
    """
    Compare new dataframes with an old Excel file and highlight differences.
    Args:
        new_dataframes (dict): Dictionary of new DataFrames to compare.
        old_file (str): Path to the old Excel file.
        diff_file (str): Path to save the differences file.
    Returns:
        bool: True if differences are found, False otherwise.
    """
    logging.info("Starting comparison of new dataframes with old file: %s", old_file)
    try:
        old_data = pd.read_excel(old_file, sheet_name=None)
        logging.info("Successfully read old Excel file: %s", old_file)
        differences_found = False
        diff_dataframes = {}

        for sheet_name, new_df in new_dataframes.items():
            logging.info("Processing sheet: %s", sheet_name)
            if sheet_name in old_data:
                old_df = old_data[sheet_name]

                # Save both DataFrames as temporary Excel files
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_new_file, \
                     tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_old_file:
                    temp_new_file.close()  # Close the file to release the lock
                    temp_old_file.close()  # Close the file to release the lock
                    new_df.to_excel(temp_new_file.name, index=False)
                    old_df.to_excel(temp_old_file.name, index=False)

                    # Read them back to standardize data types
                    standardized_new_df = pd.read_excel(temp_new_file.name)
                    standardized_old_df = pd.read_excel(temp_old_file.name)

                # Clean up temporary files
                os.unlink(temp_new_file.name)
                os.unlink(temp_old_file.name)

                # Normalize and compare
                normalized_new_df = standardized_new_df.reset_index(drop=True).sort_index(axis=1)
                normalized_old_df = standardized_old_df.reset_index(drop=True).sort_index(axis=1)

                # Identify rows that are unique to either DataFrame
                merged_df = normalized_new_df.merge(
                    normalized_old_df,
                    how="outer",
                    indicator=True
                )
                diff_rows = merged_df[merged_df["_merge"] != "both"]

                if not diff_rows.empty:
                    logging.info("Differences found in sheet: %s", sheet_name)
                    differences_found = True

                    # Add the differences to the diff_dataframes dictionary
                    diff_dataframes[sheet_name] = diff_rows.drop(columns=["_merge"])
            else:
                logging.info("Sheet %s is new and not found in old file.", sheet_name)
                differences_found = True
                diff_dataframes[sheet_name] = new_df

        for sheet_name in old_data.keys():
            if sheet_name not in new_dataframes:
                logging.info("Sheet %s is missing in new dataframes.", sheet_name)
                differences_found = True
                diff_dataframes[sheet_name] = old_data[sheet_name]

        if differences_found:
            logging.info("Differences found. Saving to file: %s", diff_file)
            with pd.ExcelWriter(diff_file) as writer:
                for sheet_name, diff_df in diff_dataframes.items():
                    diff_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            logging.info("No differences found.")

        return differences_found

    except FileNotFoundError:
        logging.warning("Old output file '%s' not found. Assuming no previous data.", old_file)
        return True  # Treat as differences found
    except Exception as e:
        logging.error("Error comparing outputs: %s", e, exc_info=True)
        return True  # Treat as differences found


def main():
    logging.info("Starting main function.")
    # Get the issues and projects that we're interested in seeing
    try:
        # Get the issues and projects that we're interested in seeing
        logging.info("Fetching mapping logic from file: %s", MAPPING_LOGIC_FILE_PATH)
        mapping_df, unique_project_keys, unique_issue_types = get_mapping_logic(MAPPING_LOGIC_FILE_PATH)
        print(unique_issue_types)
        logging.info("Successfully fetched mapping logic.")

        logging.info("Reading KPI targets from file: %s", KPI_TARGET_FILE_PATH)
        kpi_targets_df = read_excel_file(KPI_TARGET_FILE_PATH)
        logging.info("Successfully read KPI targets.")

        logging.info("Reading category definitions from file: %s", CATEGORY_DEFINITIONS_FILE_PATH)
        category_definitions_df = read_excel_file(CATEGORY_DEFINITIONS_FILE_PATH)
        logging.info("Successfully read category definitions.")

        logging.info("Reading KPI definitions from file: %s", KPI_DEFINITIONS_FILE_PATH)
        kpi_definitions_df = read_excel_file(KPI_DEFINITIONS_FILE_PATH)
        logging.info("Successfully read KPI definitions.")

        requestor_df = read_excel_file(REQUESTOR_MAPPING_FILE_PATH)

        # Find the projects and issues that we're interested in
        # file_location = get_jira_data(unique_project_keys, unique_issue_types, days=180)

        # Find the projects and issues that we're interested in
        file_location = "task_api_response\\jira_issues_response.json"

        # Generate the new data
        dataframes_dict = get_issue_data(file_location, mapping_df, kpi_targets_df, category_definitions_df, kpi_definitions_df, requestor_df, 'service_desk_data_test.xlsx')

        # Compare with the old output
        old_output_file = 'service_desk_data_test.xlsx'
        new_output_file = 'service_desk_data_test_new.xlsx'
        diff_output_file = 'service_desk_data_test_diff.xlsx'

        differences_found = compare_and_highlight(dataframes_dict, old_output_file, diff_output_file)

        if differences_found:
            print("Differences found. Saving new file and differences file.")
            save_to_excel(dataframes_dict, file_name=new_output_file)
        else:
            print("No differences found. Replacing the old file.")
            save_to_excel(dataframes_dict, file_name=old_output_file)
        logging.info("Main function completed successfully.")

    except Exception as e:
        logging.error("Error in main function: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
    for handler in logging.getLogger().handlers:
        print(f"Handler: {handler}, Level: {handler.level}")
        if isinstance(handler, logging.FileHandler):
            handler.flush()