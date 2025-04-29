import pandas as pd
import os
import glob
import shutil

# First, delete all existing maze_summary files
current_folder = os.path.dirname(__file__)
existing_summary_files = glob.glob(os.path.join(current_folder, "*_maze_summary.xlsx"))
for summary_file in existing_summary_files:
    try:
        os.remove(summary_file)
        print(f"Deleted existing summary file: {summary_file}")
    except Exception as e:
        print(f"Failed to delete {summary_file}: {e}")

# Folder and CSV setup
csv_files = [f for f in glob.glob(os.path.join(current_folder, "*.csv")) if "_maze_summary" not in os.path.basename(f)]

maze_groups = {
    'Easy Mazes': [22, 24, 25],
    'Difficult Mazes': [0, 5, 8, 15, 26],
    'Unknown Difficult Mazes': [4, 7, 14]
}

fixed_mapping = {
    'Maze_number': 'Maze_number',
    'Map-Abgedeckt': 'Map-Abgedeckt',
    'Wand_Kollisions': 'Wand_Kollisions',
    'Reward_End': 'Reward_End'
}

def find_flytime_column(columns):
    for col in columns:
        col_clean = str(col).strip().lower()
        if any(term in col_clean for term in ['flugzeit', 'flight', 'flug zeit', 'flug-zeit', 'flytime']):
            return col
    return None

for input_csv_path in csv_files:
    print(f"Processing {input_csv_path}...")

    try:
        # Properly read CSV data
        data = pd.read_csv(input_csv_path)
        
        # Check if we need to skip rows (check first few rows for header)
        if 'Flytime' not in str(data.columns) and 'flytime' not in str(data.columns).lower():
            data = pd.read_csv(input_csv_path, skiprows=0, header=0)

        flytime_col = find_flytime_column(data.columns)
        
        # If still not found, try to use the column name directly from the CSV
        if flytime_col is None and 'Flytime' in data.columns:
            flytime_col = 'Flytime'
        
        # Check for alternative column names based on the sample files
        if flytime_col is None:
            for col in data.columns:
                if col == 'Flytime' or col == 'Flugzeit':
                    flytime_col = col
                    break

        if flytime_col is None:
            print(f"⚠️ No FlyTime column found in {input_csv_path}, skipping this file.")
            continue
        else:
            print(f"✅ Found FlyTime column: {flytime_col}")

        column_mapping = {
            **fixed_mapping,
            'FlyTime': flytime_col
        }

        # Ensure numeric data
        data[column_mapping['FlyTime']] = pd.to_numeric(data[column_mapping['FlyTime']], errors='coerce')
        data = data[data[column_mapping['FlyTime']] > 0]

        # Create a dictionary to store results with metrics and stats as rows and maze groups as columns
        results_data = {}
        
        for metric, col_name in column_mapping.items():
            if metric == 'Maze_number':
                continue
                
            # Initialize metric stats in the dictionary
            for stat in ['Average', 'Min', 'Max', 'Std']:
                metric_key = f"{metric}_{stat}"
                if metric_key not in results_data:
                    results_data[metric_key] = {}
            
            # Calculate stats for each maze group
            for group_name, maze_list in maze_groups.items():
                group_data = data[pd.to_numeric(data[column_mapping['Maze_number']], errors='coerce').isin(maze_list)]
                
                # Use .loc to avoid SettingWithCopyWarning
                group_data.loc[:, col_name] = pd.to_numeric(group_data[col_name], errors='coerce')
                
                # Check if the column exists and has data
                if col_name not in group_data.columns or group_data[col_name].isnull().all():
                    print(f"⚠️ Column {col_name} not found or contains no valid data in {input_csv_path}")
                    continue
                
                mean = group_data[col_name].mean()
                min_val = group_data[col_name].min()
                max_val = group_data[col_name].max()
                std = group_data[col_name].std()
                
                # Add the statistics for this group
                results_data[f"{metric}_Average"][group_name] = f"{mean:.2f}"
                results_data[f"{metric}_Min"][group_name] = f"{min_val:.2f}"
                results_data[f"{metric}_Max"][group_name] = f"{max_val:.2f}"
                results_data[f"{metric}_Std"][group_name] = f"{std:.2f}"

        # Convert the dictionary to a DataFrame with metrics as rows and maze groups as columns
        output_df = pd.DataFrame.from_dict(results_data, orient='index')
        
        # Ensure all maze group columns are present (even if empty)
        for group_name in maze_groups.keys():
            if group_name not in output_df.columns:
                output_df[group_name] = ""
        
        # Reorder columns to match the desired order
        output_df = output_df[['Easy Mazes', 'Difficult Mazes', 'Unknown Difficult Mazes']]
        
        # Save results to Excel only (no CSV)
        base_name, _ = os.path.splitext(os.path.basename(input_csv_path))
        output_excel_path = os.path.join(current_folder, f"{base_name}_maze_summary.xlsx")
        output_df.to_excel(output_excel_path)
        print(f"✅ Saved summary to {output_excel_path}")

    except Exception as e:
        print(f"❌ Error processing {input_csv_path}: {e}")

print("🎯 All files processed successfully!")
