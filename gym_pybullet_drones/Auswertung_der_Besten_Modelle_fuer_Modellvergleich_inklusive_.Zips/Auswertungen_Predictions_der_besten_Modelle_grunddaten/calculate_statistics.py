import pandas as pd
import os
import glob

# Folder and CSV setup
current_folder = os.path.dirname(__file__)
csv_files = [f for f in glob.glob(os.path.join(current_folder, "*.csv")) if "_maze_summary" not in os.path.basename(f)]

maze_groups = {
    'Easy Mazes': [22, 24, 25],
    'Difficult Mazes': [0, 5, 8, 15, 26],
    'Unknown Difficult Mazes': [4, 7, 14]
}

fixed_mapping = {
    'Maze_number': 'Maze_number',
    'Map-Abgedeckt': 'Map-Abgedeckt',
    'Wand berührungen(Raycast<=0,1)': 'Wand berührungen(Raycast<=0,1)'
}

def find_flytime_column(columns):
    for col in columns:
        col_clean = col.strip().lower()
        if any(term in col_clean for term in ['flugzeit', 'flight', 'flug zeit', 'flug-zeit']):
            return col
    return None

for input_csv_path in csv_files:
    print(f"Processing {input_csv_path}...")

    try:
        # Properly read CSV data
        data = pd.read_csv(input_csv_path, skiprows=2, header=0)

        flytime_col = find_flytime_column(data.columns)

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

        results = []

        for group_name, maze_list in maze_groups.items():
            group_data = data[pd.to_numeric(data[column_mapping['Maze_number']], errors='coerce').isin(maze_list)]

            for metric, col_name in column_mapping.items():
                if metric == 'Maze_number':
                    continue
                group_data[col_name] = pd.to_numeric(group_data[col_name], errors='coerce')
                mean = group_data[col_name].mean()
                min_val = group_data[col_name].min()
                max_val = group_data[col_name].max()
                std = group_data[col_name].std()

                # Create string with tabs for Excel compatibility
                combined = f"{mean:.2f}\t{min_val:.2f}\t{max_val:.2f}\t{std:.2f}"

                results.append({
                    'Group': group_name,
                    'Metric': metric,
                    'Mean': f"{mean:.2f}",
                    'Min': f"{min_val:.2f}",
                    'Max': f"{max_val:.2f}",
                    'Std': f"{std:.2f}"
                })

        # Save results
        output_df = pd.DataFrame(results)
        base_name, _ = os.path.splitext(os.path.basename(input_csv_path))
        output_excel_path = os.path.join(current_folder, f"{base_name}_maze_summary.xlsx")
        output_df.to_excel(output_excel_path, index=False)

        # Also save as CSV for easy copying
        output_csv_path = os.path.join(current_folder, f"{base_name}_maze_summary.csv")
        output_df.to_csv(output_csv_path, index=False, sep='\t')
        print(f"✅ Saved summary to {output_excel_path} and {output_csv_path}")

    except Exception as e:
        print(f"❌ Error processing {input_csv_path}: {e}")

print("🎯 All files processed successfully!")
