import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_input(input_str_list, default_freq=250):
    input_dict_list = []
    for idx_window, window in enumerate(input_str_list):
        legends = window.strip('[]').split('::')  # Split by '::' separator
        for idx_legend, legend in enumerate(legends):
            input_dict = {}
            parts = legend.strip().split(':')
            file_name = parts[0].strip()
            pattern = parts[1].strip()
            # Check if frequency is provided
            if len(parts) > 2:
                freq = float(parts[2].strip())
            else:
                freq = default_freq  # Use default frequency if not specified
            unique_file_name = f"{file_name}_{idx_window}_{idx_legend}"  # Keep the unique key
            input_dict['file'] = file_name  # Store the original file path
            input_dict['pattern'] = pattern
            input_dict['frequency'] = freq
            input_dict['unique_key'] = unique_file_name
            input_dict_list.append(input_dict)
    return input_dict_list

def plot_data(file_column_mapping_list):
    # Dictionary to group file mappings by window index
    window_groupings = {}

    # Group file-column mappings by window index
    for idx, file_column_mapping in enumerate(file_column_mapping_list):
        window_idx = file_column_mapping['unique_key'].split('_')[-2]
        if window_idx not in window_groupings:
            window_groupings[window_idx] = []
        window_groupings[window_idx].append(file_column_mapping)

    # Plot for each window index group
    for window_idx, file_patterns in window_groupings.items():
        if len(file_patterns) < 1:
            print(f"Skipping window {window_idx}: no files.")
            continue

        # Extract file names, column patterns, and frequencies
        files = [fp['file'] for fp in file_patterns]
        patterns = [fp['pattern'] for fp in file_patterns]
        frequencies = [fp['frequency'] for fp in file_patterns]

        # Read CSV files using the original file paths
        data_list = []
        for f in files:
            try:
                data = pd.read_csv(f)
                data_list.append(data)
            except FileNotFoundError:
                print(f"File not found: {f}")
                continue

        # Ensure we have data for all files
        if len(data_list) < len(files):
            print(f"Skipping window {window_idx} due to missing files.")
            continue

        # Find matching columns in all files based on the patterns
        matching_cols_list = []
        for data, pattern in zip(data_list, patterns):
            matching_cols = [col for col in data.columns if re.match(pattern + r'\d*', col)]
            matching_cols_list.append(matching_cols)

        # Determine the minimum number of matching columns
        num_subplots = min(len(matching_cols) for matching_cols in matching_cols_list)
        if num_subplots == 0:
            print(f"No matching columns found for window {window_idx}")
            continue

        # Optionally, check if all files have the same number of matching columns
        if not all(len(matching_cols) == num_subplots for matching_cols in matching_cols_list):
            print(f"Warning: Mismatch in the number of matching columns between files for window {window_idx}")
            # Decide whether to proceed or skip. For now, we'll proceed with the minimum number of columns.

        # Plot subplots for corresponding columns in the same figure
        plt.figure(figsize=(18, 12))
        for i in range(num_subplots):
            plt.subplot(num_subplots, 1, i + 1)
            for data, matching_cols, file_name, freq in zip(data_list, matching_cols_list, files, frequencies):
                time = np.arange(len(data)) / freq
                # Remove the .csv extension from the file name
                base_file_name = os.path.splitext(os.path.basename(file_name))[0]
                plt.plot(time, np.array(data[matching_cols[i]]), label=f'{base_file_name} {matching_cols[i]}')
            plt.legend()
            plt.xlabel("Time (s)")
            # Use the base of the column name as ylabel (excluding any trailing numbers)
            base_label_match = re.match(r'([a-zA-Z_]+)', matching_cols_list[0][i])
            base_label = base_label_match.group(1) if base_label_match else matching_cols_list[0][i]
            plt.ylabel(base_label)
            plt.grid(True)
        plt.suptitle(f'Window {window_idx} Comparison')
        plt.tight_layout()
        plt.show()  # Ensure this is blocking as per previous advice


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Plot data from CSV files based on columns with wildcards.")
    parser.add_argument('--window', type=str, required=True, action='append',
                        help="Input format: '[csv_file_1:pattern_1[:freq_1]::csv_file_2:pattern_2[:freq_2]::...]'")

    # Parse the arguments
    args = parser.parse_args()

    # Parse the input string for file names, patterns, and frequencies
    file_column_mapping_list = parse_input(args.window)

    # Plot the data based on the provided mapping
    plot_data(file_column_mapping_list)

    # Keep figures open
    # input("Press Enter to close all figures and exit...")