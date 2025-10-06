#!/usr/bin/env python3
"""
Interactive CSV Plotter
A tool for plotting CSV files with automatic path conversion and interactive configuration.
"""

import os
import pandas as pd
import re
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional


class CSVPlotter:
    """A class to handle CSV plotting with interactive configuration."""
    
    def __init__(self, frequency: float = 100):
        self.frequency = frequency
        self.input_files = []
        self.output_files = []
        self.output_directory = None
    
    def convert_input_path(self, input_name: str) -> str:
        return f"input_files/idealpd/{input_name}.csv"
    
    def convert_output_path(self, output_name: str) -> str:
        return f"output_files/tm/{output_name}/recorded_data.csv"
    
    def get_output_directory(self, output_name: str) -> str:
        return f"output_files/tm/{output_name}/"
    
    def get_input_configuration(self) -> None:
        """Get input file and column configurations."""
        print("=== Input File Configuration ===")
        
        # Get input file name
        input_name = input("Enter input file name (e.g., 'ft_idealpd'): ").strip()
        if not input_name:
            print("No input file specified.")
            return
        
        input_path = self.convert_input_path(input_name)
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Warning: File '{input_path}' does not exist!")
        
        # Get column patterns for input file
        print("\nEnter column patterns for input file (press Enter to finish):")
        input_columns = []
        while True:
            column = input("Column pattern: ").strip()
            if not column:
                break
            input_columns.append(column)
        
        # Store input file configuration
        for column in input_columns:
            self.input_files.append({
                'file': input_path,
                'pattern': column,
                'frequency': self.frequency,
                'type': 'input'
            })
    
    def get_output_configuration(self) -> None:
        print("\n=== Output File Configuration ===")
        
        # Get output folder name
        output_name = input("Enter output folder name: ").strip()
        if not output_name:
            print("No output folder specified.")
            return
        
        output_path = self.convert_output_path(output_name)
        self.output_directory = self.get_output_directory(output_name)

        # Check if file exists
        if not os.path.exists(output_path):
            print(f"Warning: File '{output_path}' does not exist!")
        
        # Get column patterns for output file
        print("\nEnter column patterns for output file (press Enter to finish):")
        output_columns = []
        while True:
            column = input("Column pattern: ").strip()
            if not column:
                break
            output_columns.append(column)
        
        # Store output file configuration
        for column in output_columns:
            self.output_files.append({
                'file': output_path,
                'pattern': column,
                'frequency': self.frequency,
                'type': 'output'
            })
    
    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV data with error handling."""
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def find_matching_columns(self, data: pd.DataFrame, pattern: str) -> List[str]:
        """Find columns matching the given pattern."""
        regex_pattern = pattern + r'\d*'
        return [col for col in data.columns if re.match(regex_pattern, col)]
    
    def create_time_axis(self, data_length: int) -> np.ndarray:
        """Create time axis based on data length and frequency."""
        return np.arange(data_length) / self.frequency
    
    def create_plots(self) -> None:
        """Create and save plots."""
        all_files = self.input_files + self.output_files
        
        if not all_files:
            print("No files to plot.")
            return
        
        # Group files by unique file path to avoid loading the same file multiple times
        file_data = {}
        file_columns = {}
        
        # Load data and find matching columns for each file/pattern combination
        for file_config in all_files:
            file_path = file_config['file']
            pattern = file_config['pattern']
            
            # Load file data if not already loaded
            if file_path not in file_data:
                data = self.load_csv_data(file_path)
                if data is None:
                    continue
                file_data[file_path] = data
            
            # Find matching columns
            matching_cols = self.find_matching_columns(file_data[file_path], pattern)
            if matching_cols:
                key = f"{file_path}::{pattern}"
                file_columns[key] = {
                    'data': file_data[file_path],
                    'columns': matching_cols,
                    'file_path': file_path,
                    'pattern': pattern,
                    'type': file_config['type']
                }
        
        if not file_columns:
            print("No matching columns found in any files.")
            return
        
        # Determine the maximum number of columns to plot
        max_columns = max(len(info['columns']) for info in file_columns.values())
        
        if max_columns == 0:
            print("No columns to plot.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(max_columns, 1, figsize=(18, 12))
        if max_columns == 1:
            axes = [axes]
        
        # Plot each column index across all files
        for col_idx in range(max_columns):
            ax = axes[col_idx]
            
            for key, info in file_columns.items():
                if col_idx < len(info['columns']):
                    data = info['data']
                    column_name = info['columns'][col_idx]                    
                    time_axis = self.create_time_axis(len(data))
                    
                    # Create label with file type and column name
                    label = f"{column_name}"
                    
                    ax.plot(time_axis, data[column_name].values, label=label)
                    ax.set_ylabel(label)
            
            ax.legend()
            ax.set_xlabel("Time (s)")
            ax.grid(True)

        # Create subtitle with all column names
        all_columns = []
        for info in file_columns.values():
            all_columns.extend(info['columns'])
        unique_columns = list(set(all_columns))  # Remove duplicates
        subtitle = f"{', '.join(unique_columns)}"
        
        # plt.suptitle(subtitle)
        plt.tight_layout()
        
        # Save plot
        if self.output_directory:
            # Create output directory if it doesn't exist
            plot_name = input("Enter output plot name (default: 'comparison_plot.pdf'): ").strip()
            Path(self.output_directory).mkdir(parents=True, exist_ok=True)
            
            output_file = Path(self.output_directory) / plot_name
            plt.savefig(output_file, format='pdf', bbox_inches='tight')
        
        plt.show()
    
    def run(self) -> None:
        """Main execution method."""
        
        # Get input configuration
        self.get_input_configuration()
        
        # Get output configuration
        self.get_output_configuration()
        
        # Check if we have any files to plot
        if not self.input_files and not self.output_files:
            print("No files configured for plotting. Exiting.")
            return
        
        self.create_plots()

def main():
    """Main function to run the CSV plotter."""
    plotter = CSVPlotter()
    plotter.run()


if __name__ == "__main__":
    main()