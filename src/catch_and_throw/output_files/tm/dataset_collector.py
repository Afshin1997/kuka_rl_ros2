import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Set GUI backend for matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

def post_process_trajectory(df, z_threshold=0.6):
    """
    Post-process trajectory data by cutting it when ball goes below threshold with negative velocity.
    
    Args:
        df: DataFrame with trajectory data
        z_threshold: Z-position threshold in meters (default 0.6m)
    
    Returns:
        DataFrame with truncated trajectory
    """
    # Check if required columns exist
    if 'ball_pos_z' not in df.columns or 'ball_ema_vz' not in df.columns:
        print("  Warning: Required columns for post-processing not found")
        return df
    
    # Find the cut point
    cut_index = None
    for idx in range(len(df)):
        x_pos = df.loc[idx, 'ball_pos_x']
        y_pos = df.loc[idx, 'ball_pos_y']
        z_pos = df.loc[idx, 'ball_pos_z']
        x_vel = df.loc[idx, 'ball_raw_vx']
        y_vel = df.loc[idx, 'ball_ema_vy']
        z_vel = df.loc[idx, 'ball_raw_vz']
        y_vel_raw = df.loc[idx, 'ball_raw_vy']
        
        # Check if z position is below threshold AND velocity is negative (going down)
        if (z_pos < z_threshold and z_vel < 0) or (y_pos > -0.2) or (y_pos < -0.7) or (x_pos < 1.0 and z_vel > 0.0) or (x_vel > 0.0) or (y_vel < -0.4) or (y_vel > 0.4) or (x_vel > -2.3) or (y_vel_raw < -0.5) or (y_vel_raw > 0.5) or (y_vel_raw < -0.5):
            cut_index = idx
            break
    
    if cut_index is not None:
        # Cut the data at this point
        df_processed = df.iloc[:cut_index].copy()
        print(f"    Cut trajectory at index {cut_index} (z={z_pos:.3f}m, vz={z_vel:.3f}m/s)")
        print(f"    Kept {len(df_processed)}/{len(df)} points")
        return df_processed
    else:
        print(f"    No cut needed (trajectory stays above {z_threshold}m or never has negative velocity)")
        return df.copy()

def copy_and_process_csv_files(source_dir, target_dir, filename="recorded_data.csv", 
                              apply_post_processing=True, z_threshold=0.6):
    """
    Copy specific columns from CSV files and optionally apply post-processing.
    
    Args:
        source_dir: Root directory containing folders with CSV files
        target_dir: Directory where processed CSV files will be saved
        filename: Name of the CSV file to look for in each folder
        apply_post_processing: If True, cut trajectories when ball goes below threshold
        z_threshold: Z-position threshold for cutting trajectories
    """
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Columns to extract
    columns_to_keep = [
        'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
        'ball_raw_vx', 'ball_raw_vy', 'ball_raw_vz',
        'ball_ema_vx', 'ball_ema_vy', 'ball_ema_vz'
    ]
    
    processed_files = []
    processing_stats = {
        'total_files': 0,
        'processed_files': 0,
        'cut_files': 0,
        'total_points_before': 0,
        'total_points_after': 0
    }
    counter = 0
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Skip the source directory itself and the target directory
        if root == source_dir or root.startswith(os.path.join(source_dir, target_dir)):
            continue
            
        if filename in files:
            csv_path = os.path.join(root, filename)
            folder_name = os.path.basename(root)
            processing_stats['total_files'] += 1
            
            try:
                print(f"\nProcessing: {folder_name}/{filename}")
                
                # Read the CSV file
                df = pd.read_csv(csv_path)
                original_length = len(df)
                processing_stats['total_points_before'] += original_length
                
                # Check which columns exist
                existing_columns = [col for col in columns_to_keep if col in df.columns]
                
                if not existing_columns:
                    print(f"  Warning: No matching columns found in {csv_path}")
                    continue
                
                # Extract only the desired columns
                df_filtered = df[existing_columns]
                
                # Apply post-processing if requested
                if apply_post_processing:
                    df_filtered = post_process_trajectory(df_filtered, z_threshold)
                    if len(df_filtered) < original_length:
                        processing_stats['cut_files'] += 1
                
                processing_stats['total_points_after'] += len(df_filtered)
                
                # Only save if we have data left
                if len(df_filtered) > 80 and df_filtered['ball_pos_x'].values[-1] > -0.2:
                    # Save to target directory with folder name prefix
                    output_filename = f"{counter}.csv"
                    output_path = os.path.join(target_dir, output_filename)
                    df_filtered.to_csv(output_path, index=False)
                    
                    processed_files.append(output_path)
                    processing_stats['processed_files'] += 1
                    print(f"  Saved: {output_filename}")
                    counter += 1
                else:
                    print(f"  Skipped: No data points remaining after processing")
                
            except Exception as e:
                print(f"  Error processing {csv_path}: {str(e)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files found: {processing_stats['total_files']}")
    print(f"Files successfully processed: {processing_stats['processed_files']}")
    if apply_post_processing:
        print(f"Files with trajectories cut: {processing_stats['cut_files']}")
        print(f"Total data points before: {processing_stats['total_points_before']}")
        print(f"Total data points after: {processing_stats['total_points_after']}")
        reduction = 100 * (1 - processing_stats['total_points_after']/max(1, processing_stats['total_points_before']))
        print(f"Data reduction: {reduction:.1f}%")
    
    return processed_files

def plot_trajectories_from_files(csv_files_dir, trajectories_per_plot=10, save_plots=False, 
                                 show_threshold_plane=False, z_threshold=0.6):
    """
    Plot 3D trajectories from CSV files in batches.
    
    Args:
        csv_files_dir: Directory containing CSV files
        trajectories_per_plot: Number of trajectories to show per plot
        save_plots: If True, save plots as images instead of showing them
        show_threshold_plane: If True, show the z-threshold plane in the plot
        z_threshold: Z-position threshold to visualize
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_files_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_files_dir}")
        return
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Calculate number of plots needed
    num_plots = (len(csv_files) + trajectories_per_plot - 1) // trajectories_per_plot
    print(f"Creating {num_plots} plots with up to {trajectories_per_plot} trajectories each")
    
    # Color map for different trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Create plots in batches
    for plot_idx in range(num_plots):
        start_idx = plot_idx * trajectories_per_plot
        end_idx = min(start_idx + trajectories_per_plot, len(csv_files))
        batch_files = csv_files[start_idx:end_idx]
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        successful_plots = 0
        all_x, all_y = [], []  # To determine plane size
        
        # Plot each trajectory in the batch
        for i, csv_file in enumerate(batch_files):
            csv_path = os.path.join(csv_files_dir, csv_file)
            
            try:
                # Read CSV file
                df = pd.read_csv(csv_path)
                
                # Check if position columns exist
                required_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
                if all(col in df.columns for col in required_cols):
                    # Convert to numpy arrays to avoid indexing issues
                    x = df['ball_pos_x'].values
                    y = df['ball_pos_y'].values
                    z = df['ball_pos_z'].values
                    
                    # Remove any NaN values
                    valid_indices = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
                    x = x[valid_indices]
                    y = y[valid_indices]
                    z = z[valid_indices]
                    
                    if len(x) > 0:  # Only plot if we have valid data
                        all_x.extend(x)
                        all_y.extend(y)
                        
                        # Get trajectory name from filename
                        trajectory_name = os.path.splitext(csv_file)[0]
                        # Shorten name if too long for legend
                        if len(trajectory_name) > 30:
                            trajectory_name = trajectory_name[:27] + "..."
                        
                        # Plot the trajectory
                        ax.plot(x, y, z, 
                               label=f"Traj {start_idx + i + 1}: {trajectory_name}", 
                               color=colors[i % 20],
                               alpha=0.8, 
                               linewidth=2)
                        
                        # Mark start and end points
                        ax.scatter(x[0], y[0], z[0], 
                                  color=colors[i % 20], 
                                  marker='o', s=50, alpha=0.8)  # Start point
                        ax.scatter(x[-1], y[-1], z[-1], 
                                  color=colors[i % 20], 
                                  marker='^', s=50, alpha=0.8)  # End point
                        
                        successful_plots += 1
                        print(f"  Plotted: {csv_file} ({len(x)} points)")
                    else:
                        print(f"  Warning: No valid data points in {csv_file}")
                else:
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    print(f"  Warning: Missing columns {missing_cols} in {csv_file}")
                    
            except Exception as e:
                print(f"  Error plotting {csv_file}: {str(e)}")
        
        if successful_plots > 0:
            # Add threshold plane if requested
            if show_threshold_plane and all_x and all_y:
                # Create a semi-transparent plane at z_threshold
                x_range = [min(all_x), max(all_x)]
                y_range = [min(all_y), max(all_y)]
                
                # Extend ranges by 10% for better visualization
                x_margin = (x_range[1] - x_range[0]) * 0.1
                y_margin = (y_range[1] - y_range[0]) * 0.1
                
                xx, yy = np.meshgrid(
                    np.linspace(x_range[0] - x_margin, x_range[1] + x_margin, 10),
                    np.linspace(y_range[0] - y_margin, y_range[1] + y_margin, 10)
                )
                zz = np.ones_like(xx) * z_threshold
                
                ax.plot_surface(xx, yy, zz, alpha=0.2, color='red', 
                               label=f'Threshold plane (z={z_threshold}m)')
                
                # Add a text label for the plane
                ax.text(np.mean(x_range), np.mean(y_range), z_threshold, 
                       f'z = {z_threshold}m threshold', 
                       color='red', fontsize=10, weight='bold')
            
            # Set labels and title
            ax.set_xlabel('X Position (m)', fontsize=11, labelpad=20)
            ax.set_ylabel('Y Position (m)', fontsize=11, labelpad=20)
            ax.set_zlabel('Z Position (m)', fontsize=11, labelpad=20)
            ax.set_title(f'Ball Trajectories - Batch {plot_idx + 1}/{num_plots}\n'
                        f'Files {start_idx + 1}-{end_idx} ({successful_plots} trajectories)',
                        fontsize=13, pad=20)
            
            # Add legend
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                     fontsize=9, framealpha=0.9)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set viewing angle for better visualization
            ax.view_init(elev=20, azim=45)
            
            # Adjust layout
            plt.tight_layout()
            
            if save_plots:
                # Save plot as image
                plot_filename = f"trajectory_plot_batch_{plot_idx + 1}.png"
                output_path = os.path.join("dataset", plot_filename)
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                print(f"Saved plot to {plot_filename}")
                plt.close()
            else:
                # Show plot
                plt.show()
                
            print(f"Completed plot {plot_idx + 1}/{num_plots} with {successful_plots} trajectories\n")
        else:
            print(f"No valid trajectories in batch {plot_idx + 1}, skipping plot\n")
            plt.close()

def main():
    """
    Main function to execute the script with post-processing options.
    """
    # Configuration - modify these paths as needed
    SOURCE_DIR = "."  # Current directory
    TARGET_DIR = "dataset"  # Target directory for processed files
    TRAJECTORIES_PER_PLOT = 20  # Number of trajectories per plot
    Z_THRESHOLD = 0.6  # Z-position threshold in meters
    
    print("="*60)
    print("CSV File Processing with Trajectory Cutting")
    print("="*60)
    print(f"Post-processing: Cut trajectories when z < {Z_THRESHOLD}m with negative velocity")
    print("="*60)
    
    # Ask about processing mode
    print("\nSelect processing mode:")
    print("1. Copy and post-process files (cut at z-threshold)")
    print("2. Copy files without post-processing")
    print("3. Just plot existing files")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '2']:
        apply_post_processing = (choice == '1')
        
        if apply_post_processing:
            # Ask for custom threshold
            custom_threshold = input(f"\nEnter z-threshold in meters (press Enter for {Z_THRESHOLD}m): ").strip()
            if custom_threshold:
                try:
                    Z_THRESHOLD = float(custom_threshold)
                    print(f"Using z-threshold: {Z_THRESHOLD}m")
                except ValueError:
                    print(f"Invalid input, using default: {Z_THRESHOLD}m")
        
        print(f"\nCopying {'and post-processing' if apply_post_processing else ''} CSV files...")
        print("-"*40)
        
        processed_files = copy_and_process_csv_files(
            SOURCE_DIR, 
            TARGET_DIR,
            apply_post_processing=apply_post_processing,
            z_threshold=Z_THRESHOLD
        )
        
        if not processed_files:
            print("\nNo files were processed. Please check your input directory.")
            return
        
        plot_dir = TARGET_DIR
    else:
        # Option 3: Plot existing files
        plot_dir = input("\nEnter the directory path containing CSV files (or press Enter for 'dataset'): ").strip()
        if not plot_dir:
            plot_dir = TARGET_DIR
    
    # Ask about plotting
    plot_choice = input("\nDo you want to plot the trajectories? (y/n): ").lower().strip()
    
    if plot_choice == 'y':
        print(f"\nPlotting trajectories from {plot_dir}...")
        print("-"*40)
        
        # Ask about saving plots
        save_option = input("Save plots as images? (y/n): ").lower().strip()
        save_plots = (save_option == 'y')
        
        # Ask about showing threshold plane
        show_plane = input(f"Show z={Z_THRESHOLD}m threshold plane in plots? (y/n): ").lower().strip()
        show_threshold = (show_plane == 'y')
        
        plot_trajectories_from_files(
            plot_dir, 
            TRAJECTORIES_PER_PLOT, 
            save_plots,
            show_threshold,
            Z_THRESHOLD
        )
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()