from jira import JIRA
import json
import os
import yaml

def fetch_issues_from_jira(jira_instance, project_keys, issue_types, days=300):
    """
    Fetch issues created in the past `days` for the given project keys.
    Returns the raw API response for later interpretation.
    """
    
    all_issues = []
    issue_types_str = ', '.join(f'"{issue_type}"' for issue_type in issue_types)  # Create a string for JQL
    for project_key in project_keys:
        #jql_query = f'project = "{project_key}" AND created >= -{days}d AND issueType IN ({issue_types_str})'
        jql_query = f'project = "{project_key}" AND resolutiondate >= -{days}d AND issueType IN ({issue_types_str})'
        print(f"Fetching issues for JQL: {jql_query}")
        issues = jira_instance.search_issues(jql_query, maxResults=False, expand='changelog')
        all_issues.extend(issues)
    return all_issues

def load_config(file_path: str) -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise

def save_issues_to_json(issues, base_folder="api_response", base_filename="jira_issues_response.json"):
    """
    Save all issues to a single JSON file in the specified folder.
    """
    # Ensure the folder exists
    os.makedirs(base_folder, exist_ok=True)
    
    # Collect all issue data
    all_issues = [issue.raw for issue in issues]
    
    # Save all issues to a single JSON file
    output_file = os.path.join(base_folder, base_filename)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=4)
    
    print(f"Saved {len(all_issues)} issues to {output_file}")
    return output_file

def get_jira_data(project_keys, issue_types, days):
    
    # Load the configuration
    config = load_config("config.yaml")
    
    # Access Jira credentials
    jira_user = config["jira"]["user"]
    jira_password = config["jira"]["password"]
    
    # Replace with your Jira server URL and credentials
    jira_server = ["jira"]['server']
    ca_bundle_path = ["jira"]["certs"]  # Path to your custom CA bundle
    
    # Connect to Jira with a custom CA bundle
    jira_instance = JIRA(server=jira_server, basic_auth=(jira_user, jira_password), options={'verify': ca_bundle_path})
    
    # Fetch issues created in the past 300 days
    issues = fetch_issues_from_jira(jira_instance, project_keys, issue_types, days=days)
    
    # Save the API response to JSON files in the "api_response" folder
    return save_issues_to_json(issues, base_folder="task_api_response", base_filename="jira_issues_response.json")