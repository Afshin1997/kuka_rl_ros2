import pandas as pd
import numpy as np
import os
import glob

def calculate_restitution_coefficient(csv_file):
    """
    Calculate the restitution coefficient from a single CSV file.
    Heights are normalized so that the lowest point (impact point) is zero.
    
    Parameters:
    csv_file (str): Path to the CSV file
    
    Returns:
    float: Restitution coefficient
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get the ball z positions
        z_positions = df['ball_pos_z'].values
        
        # Find the minimum height (impact point)
        min_height = np.min(z_positions)
        
        # Normalize heights so that impact point is at zero
        normalized_heights = z_positions - min_height
        
        # Initial height (drop height) - first row after normalization
        h_drop = normalized_heights[0]
        
        # Rebound height - last row after normalization
        h_rebound = normalized_heights[-1]
        
        # Calculate restitution coefficient
        # e = sqrt(h_rebound / h_drop)
        if h_drop > 0:  # Avoid division by zero
            e = np.sqrt(h_rebound / h_drop)
        else:
            print(f"Warning: Drop height is 0 or negative in {csv_file}")
            return None
            
        # Optional: Print debug information
        print(f"\nFile: {os.path.basename(csv_file)}")
        print(f"  Original heights: min={min_height:.4f}, initial={z_positions[0]:.4f}, final={z_positions[-1]:.4f}")
        print(f"  Normalized: drop_height={h_drop:.4f}, rebound_height={h_rebound:.4f}")
        print(f"  Restitution coefficient: e={e:.4f}")
            
        return e
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        return None

def process_folder(folder_path, file_pattern='*.csv', verbose=True):
    """
    Process all CSV files in a folder and calculate average restitution coefficient.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    file_pattern (str): Pattern to match CSV files (default: '*.csv')
    verbose (bool): Print detailed information for each file
    
    Returns:
    dict: Dictionary containing results
    """
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    print("=" * 60)
    
    # Calculate restitution coefficient for each file
    restitution_coefficients = []
    file_results = {}
    
    for csv_file in csv_files:
        e = calculate_restitution_coefficient(csv_file)
        if e is not None:
            restitution_coefficients.append(e)
            file_results[os.path.basename(csv_file)] = e
    
    # Calculate average
    if restitution_coefficients:
        avg_e = np.mean(restitution_coefficients)
        std_e = np.std(restitution_coefficients)
        min_e = np.min(restitution_coefficients)
        max_e = np.max(restitution_coefficients)
        
        results = {
            'average_restitution': avg_e,
            'std_deviation': std_e,
            'min_restitution': min_e,
            'max_restitution': max_e,
            'num_files': len(restitution_coefficients),
            'individual_results': file_results
        }
        
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Number of valid files: {len(restitution_coefficients)}")
        print(f"Average restitution coefficient: {avg_e:.4f}")
        print(f"Standard deviation: {std_e:.4f}")
        print(f"Minimum: {min_e:.4f}")
        print(f"Maximum: {max_e:.4f}")
        print(f"Range: {max_e - min_e:.4f}")
        
        return results
    else:
        print("No valid restitution coefficients calculated")
        return None

def save_results_to_file(results, folder_path):
    """Save analysis results to a text file."""
    output_file = os.path.join(folder_path, 'restitution_results.txt')
    with open(output_file, 'w') as f:
        f.write("Ball Drop Restitution Coefficient Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folder: {folder_path}\n\n")
        f.write("Summary Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Average restitution coefficient: {results['average_restitution']:.4f}\n")
        f.write(f"Standard deviation: {results['std_deviation']:.4f}\n")
        f.write(f"Minimum: {results['min_restitution']:.4f}\n")
        f.write(f"Maximum: {results['max_restitution']:.4f}\n")
        f.write(f"Range: {results['max_restitution'] - results['min_restitution']:.4f}\n")
        f.write(f"Number of files analyzed: {results['num_files']}\n\n")
        f.write("Individual File Results:\n")
        f.write("-" * 60 + "\n")
        for filename, e in sorted(results['individual_results'].items()):
            f.write(f"{filename}: {e:.4f}\n")
    return output_file

# Main script
if __name__ == "__main__":
    # Specify your folder path here
    folder_path = input("Enter the folder path containing CSV files: ").strip()
    
    # Ask for verbosity
    verbose_input = input("Show detailed information for each file? (y/n, default=y): ").strip().lower()
    verbose = verbose_input != 'n'
    
    # Process the folder
    results = process_folder(folder_path, verbose=verbose)
    
    # Optional: Save results to a file
    if results:
        save_results = input("\nSave results to file? (y/n): ").strip().lower()
        if save_results == 'y':
            output_file = save_results_to_file(results, folder_path)
            print(f"\nResults saved to: {output_file}")
            
        # Optional: Export to CSV
        export_csv = input("\nExport individual results to CSV? (y/n): ").strip().lower()
        if export_csv == 'y':
            csv_output = os.path.join(folder_path, 'restitution_coefficients.csv')
            df_results = pd.DataFrame(list(results['individual_results'].items()), 
                                    columns=['Filename', 'Restitution_Coefficient'])
            df_results = df_results.sort_values('Filename')
            df_results.to_csv(csv_output, index=False)
            print(f"CSV results saved to: {csv_output}")