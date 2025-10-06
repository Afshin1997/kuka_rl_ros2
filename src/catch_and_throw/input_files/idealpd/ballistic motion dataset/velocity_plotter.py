import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_csv_files(columns_to_plot, x_column=None):
    """
    Read all CSV files in a folder and plot specified columns in subplots
    
    Parameters:
    columns_to_plot (list): List of 3 column names to plot
    x_column (str): Column to use as x-axis (optional, uses index if None)
    """
    
    # Get all CSV files in the folder
    # csv_files = glob.glob("*.csv")
    csv_files = ['49.csv', '50.csv', '51.csv', '52.csv', '54.csv']  # Example list of files, replace with glob.glob("*.csv") for actual use

    # Create subplots - 3 columns in one row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CSV Data Visualization', fontsize=16)
    
    # Read and plot each CSV file
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            
            print(f"Processing: {filename}")
            print(f"Available columns: {list(df.columns)}")
            
            # Plot each specified column in its subplot
            for i, col in enumerate(columns_to_plot):
                if col in df.columns:
                    if x_column and x_column in df.columns:
                        axes[i].plot(df[x_column], df[col], label=filename, marker='o', markersize=3)
                        axes[i].set_xlabel(x_column)
                    else:
                        axes[i].plot(df[col], label=filename, marker='o', markersize=3)
                        axes[i].set_xlabel('Index')
                    
                    axes[i].set_ylabel(col)
                    axes[i].set_title(f'{col}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                else:
                    print(f"Warning: Column '{col}' not found in {filename}")
                    
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Specify your folder path and columns to plot
    columns_to_plot = ["ball_raw_vx", "ball_raw_vy", "ball_raw_vz"]  # Change these to your column names
    x_column = None  # Optional: specify x-axis column name, or leave as None to use index
    
    # Alternative examples:
    # x_column = "time"  # if you want to plot against a time column
    # x_column = "date"  # if you want to plot against a date column
    
    plot_csv_files(columns_to_plot, x_column)
    
    # Example with specific settings:
    # plot_csv_files("./data", ["temperature", "humidity", "pressure"], "timestamp")