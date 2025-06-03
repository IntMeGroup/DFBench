import pandas as pd
import os

def process_csv_and_summarize(input_csv_path, output_dir):
    """
    Reads a CSV, splits it into new CSVs based on the first part of the 'image' column,
    saves them to a specified directory, and creates a statistics TXT file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading input CSV {input_csv_path}: {e}")
        return

    if 'image' not in df.columns or 'answer' not in df.columns:
        print(f"Error: Required columns 'image' and/or 'answer' not found in {input_csv_path}.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Extract subfolder identifier
    # Handle cases where 'image' might not contain '/'
    df['subfolder'] = df['image'].apply(lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else 'unknown_folder')

    grouped = df.groupby('subfolder')
    statistics = []

    for name, group in grouped:
        output_csv_filename = f"{name}.csv"
        output_csv_path = os.path.join(output_dir, output_csv_filename)
        
        # Select only original columns for the new CSV, excluding the temporary 'subfolder' column
        group_to_save = group.drop(columns=['subfolder'])
        group_to_save.to_csv(output_csv_path, index=False)
        print(f"Saved {output_csv_path}")

        # Count A: and B: in the 'answer' column for the current group
        # Ensure 'answer' column is treated as string for reliable checking
        count_a = group['answer'].astype(str).str.startswith('A').sum()
        count_b = group['answer'].astype(str).str.startswith('B').sum()
        
        statistics.append({
            'csv_file': output_csv_filename,
            'count_a': count_a,
            'count_b': count_b
        })

    # Write statistics to TXT file
    stats_txt_path = os.path.join(output_dir, "statistics.txt")
    with open(stats_txt_path, 'w') as f:
        for stat in statistics:
            f.write(f"File: {stat['csv_file']}\n")
            f.write(f"  A: occurrences: {stat['count_a']}\n")
            f.write(f"  B: occurrences: {stat['count_b']}\n")
            f.write("-\n")
    print(f"Statistics saved to {stats_txt_path}")

if __name__ == '__main__':
    input_file = r"E:\MMcode\MM_DATA\combined_final_predictions.csv" 
    # Corrected relative path for the tool to find the file
    # The tool seems to prefer paths relative to workspace_root if not absolute and outside
    # However, for pandas, the absolute path given by user should work if accessible by python.
    # If the tool environment has issues, we might need to adjust. For now, assume r"E:\..." works for pandas.
    
    output_directory = r"E:\MMcode\72B\combine"
    
    process_csv_and_summarize(input_file, output_directory)
