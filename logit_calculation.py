import pandas as pd
import numpy as np
import os

def combine_predictions():
    """
    Reads data from three CSV files (skipping the first row of each),
    finds common images, sums their respective prob_A and prob_B contributions
    (treating missing/non-numeric as 0), determines a final answer (A or B),
    and saves the results to a new CSV file.
    """
    base_path = r"E:\MMcode\MM_DATA"
    file_info = [
        {"path": os.path.join(base_path, "resultsqwen25.csv"), "suffix": "f1"},
        {"path": os.path.join(base_path, "resultsintern25.csv"), "suffix": "f2"},
        {"path": os.path.join(base_path, "resultsintern3.csv"), "suffix": "f3"}
    ]
    output_file_path = os.path.join(base_path, "combined_final_predictionswjt.csv")

    dataframes = []

    for info in file_info:
        file_path = info["path"]
        suffix = info["suffix"]
        try:
            # Skip the first row (header) and continue to use no explicit header for column indexing
            df = pd.read_csv(file_path, header=None, skiprows=1)
            
            if df.empty:
                print(f"Warning: File {file_path} is empty after skipping the first row. This file will be skipped.")
                continue # Skip to the next file
            
            # Expecting 5 columns after skipping header for image, ss, answer, prob_A, prob_B
            if df.shape[1] < 5:
                print(f"Error: File {file_path} (after skipping header) has {df.shape[1]} columns, but expected at least 5.")
                # If one file is malformed, we might not want to proceed with partial data for merging.
                # Depending on requirements, one might choose to skip this file or halt entirely.
                # For now, halting if a file is malformed after header skip.
                return 
            
            # Select first 5 columns and name them
            df = df.iloc[:, :5]
            df.columns = [
                'image',
                f'ss_{suffix}',
                f'answer_{suffix}', # This is the original answer from the file
                f'prob_A_{suffix}',
                f'prob_B_{suffix}'
            ]

            prob_a_col = f'prob_A_{suffix}'
            prob_b_col = f'prob_B_{suffix}'
            
            df[prob_a_col] = pd.to_numeric(df[prob_a_col], errors='coerce')
            df[prob_b_col] = pd.to_numeric(df[prob_b_col], errors='coerce')

            nan_in_a = df[prob_a_col].isnull().sum()
            nan_in_b = df[prob_b_col].isnull().sum()

            if nan_in_a > 0:
                print(f"Warning: In {file_path}, {nan_in_a} values in {prob_a_col} (from 4th data column) were non-numeric/missing and will be treated as 0.0.")
                df[prob_a_col] = df[prob_a_col].fillna(0.0)
            if nan_in_b > 0:
                print(f"Warning: In {file_path}, {nan_in_b} values in {prob_b_col} (from 5th data column) were non-numeric/missing and will be treated as 0.0.")
                df[prob_b_col] = df[prob_b_col].fillna(0.0)
            
            if df.empty:
                print(f"Warning: File {file_path} became empty after processing (e.g. all rows had issues). This file will be skipped.")
                continue

            dataframes.append(df)
            print(f"Successfully loaded and prepared {file_path} (skipped first row).")

        except FileNotFoundError:
            print(f"Error: File not found - {file_path}. Cannot proceed.")
            return
        except pd.errors.EmptyDataError: # Catch if file is empty AFTER header skip attempt
            print(f"Warning: File {file_path} is empty or only contained a header. This file will be skipped.")
            continue
        except Exception as e:
            print(f"Error loading or processing file {file_path}: {e}. Cannot proceed.")
            return

    # Filter out any completely empty dataframes that might have resulted from skipped files
    dataframes = [d for d in dataframes if not d.empty]
    if len(dataframes) < 3:
        print(f"Fewer than 3 files have valid data after loading ({len(dataframes)} available). Halting combination.")
        return

    df1, df2, df3 = dataframes[0], dataframes[1], dataframes[2]

    merged_df = pd.merge(df1, df2, on='image', how='inner')
    if merged_df.empty:
        print("No common images found between the first two files. Output will be empty.")
    
    merged_df = pd.merge(merged_df, df3, on='image', how='inner')

    if merged_df.empty:
        print("No common images found across all three files after merging. The output file will be empty or not created.")
        final_output_df = pd.DataFrame(columns=['image', 'ss', 'total_prob_A', 'total_prob_B', 'answer'])
    else:
        print(f"Found {len(merged_df)} common images across all three files.")
        merged_df['total_prob_A'] = merged_df['prob_A_f1'] + merged_df['prob_A_f2'] + merged_df['prob_A_f3']
        merged_df['total_prob_B'] = merged_df['prob_B_f1'] + merged_df['prob_B_f2'] + merged_df['prob_B_f3']

        # Determine final answer based on summed probabilities
        merged_df['final_answer'] = np.where(merged_df['total_prob_A'] > merged_df['total_prob_B'], 'A', 'B')

        # Prepare final output dataframe
        # Retain 'ss' from the first file (f1) and the common 'image'
        # Also includes the determined 'final_answer'. The individual answer_f1, answer_f2, answer_f3 are available in merged_df if needed.
        final_output_df = merged_df[[
            'image',
            'ss_f1', 
            'total_prob_A',
            'total_prob_B',
            'final_answer' # Use the newly determined final answer
        ]]
        # Rename columns for the final output
        final_output_df = final_output_df.rename(columns={'ss_f1': 'ss', 'final_answer': 'answer'})

    try:
        final_output_df.to_csv(output_file_path, index=False)
        if final_output_df.empty:
            print(f"Output file {output_file_path} created but is empty as no common images were found or no files had valid data.")
        else:
            print(f"Successfully processed files. Output saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving output file {output_file_path}: {e}")

if __name__ == '__main__':
    combine_predictions() 