import os
import csv

def list_python_files_single_column(root_dir, output_csv):
    python_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                python_files.append([full_path])

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['path'])  # Header
        writer.writerows(python_files)

    print(f"CSV created: {output_csv} ({len(python_files)} Python files listed)")

# Example usage
if __name__ == '__main__':
    target_directory = r'C:\Users\User\python_code\csv_excel_parser'  # <- Change this
    output_csv_path = 'python_files.csv'
    list_python_files_single_column(target_directory, output_csv_path)
