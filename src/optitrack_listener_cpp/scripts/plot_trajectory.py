#!/usr/bin/env python3
"""
3D Trajectory Plotter for OptiTrack C++ Package Data
Plots both raw and smoothed trajectories from CSV files
Usage: python plot_trajectory.py data.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import argparse

def load_and_process_data(file_path):
    """Load CSV file and process the trajectory data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}: {len(df)} data points")
        
        # Map column names based on what's available
        column_mapping = {
            'pos_world_x': 'x',
            'pos_world_y': 'y', 
            'pos_world_z': 'z',
            'pos_smooth_x': 'x_smooth',
            'pos_smooth_y': 'y_smooth',
            'pos_smooth_z': 'z_smooth',
            'vel_x': 'vx',
            'vel_y': 'vy',
            'vel_z': 'vz'
        }
        
        # Rename columns if they exist
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # Check required columns
        required_cols = ['timestamp', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Missing required columns. Found: {list(df.columns)}")
            return None
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate relative time
        df['time_relative'] = df['timestamp'] - df['timestamp'].min()
        
        # Calculate speed if velocity components available
        if all(col in df.columns for col in ['vx', 'vy', 'vz']):
            df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
        else:
            # Estimate from position
            dt = np.gradient(df['time_relative'])
            dt[dt == 0] = 1e-6
            df['vx'] = np.gradient(df['x']) / dt
            df['vy'] = np.gradient(df['y']) / dt
            df['vz'] = np.gradient(df['z']) / dt
            df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
        
        # Check if smoothed data exists
        has_smooth = all(col in df.columns for col in ['x_smooth', 'y_smooth', 'z_smooth'])
        
        return df, has_smooth
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, False

def print_statistics(df, has_smooth):
    """Print comprehensive trajectory statistics"""
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    print(f"Total data points: {len(df)}")
    print(f"Duration: {df['time_relative'].max():.2f} seconds")
    print(f"Sampling rate: ~{len(df)/df['time_relative'].max():.1f} Hz")
    
    # Raw position statistics
    print(f"\nRaw Position Range:")
    print(f"  X: {df['x'].min():.3f} to {df['x'].max():.3f} m (range: {df['x'].max()-df['x'].min():.3f} m)")
    print(f"  Y: {df['y'].min():.3f} to {df['y'].max():.3f} m (range: {df['y'].max()-df['y'].min():.3f} m)")
    print(f"  Z: {df['z'].min():.3f} to {df['z'].max():.3f} m (range: {df['z'].max()-df['z'].min():.3f} m)")
    
    # Smoothed position statistics if available
    if has_smooth:
        print(f"\nSmoothed Position Range:")
        print(f"  X: {df['x_smooth'].min():.3f} to {df['x_smooth'].max():.3f} m")
        print(f"  Y: {df['y_smooth'].min():.3f} to {df['y_smooth'].max():.3f} m")
        print(f"  Z: {df['z_smooth'].min():.3f} to {df['z_smooth'].max():.3f} m")
    
    # Speed statistics
    if 'speed' in df.columns:
        print(f"\nSpeed Statistics:")
        print(f"  Max speed: {df['speed'].max():.3f} m/s")
        print(f"  Average speed: {df['speed'].mean():.3f} m/s")
        print(f"  Min speed: {df['speed'].min():.3f} m/s")
        
        # Distance traveled
        dt = np.gradient(df['time_relative'])
        distance = np.sum(df['speed'] * dt)
        print(f"  Total distance traveled: {distance:.3f} m")
    
    print("="*60)

def plot_3d_trajectory(df, has_smooth, show_raw=True):
    """Create comprehensive 3D trajectory plots"""
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Plot raw trajectory if requested
    if show_raw:
        x, y, z = df['x'].values, df['y'].values, df['z'].values
        ax1.plot(x, y, z, 'b-', linewidth=1, alpha=0.4, label='Raw')
    
    # Plot smoothed trajectory if available
    if has_smooth:
        xs, ys, zs = df['x_smooth'].values, df['y_smooth'].values, df['z_smooth'].values
        ax1.plot(xs, ys, zs, 'r-', linewidth=2, alpha=0.8, label='Smoothed')
        
        # Use smoothed for markers
        plot_x, plot_y, plot_z = xs, ys, zs
    else:
        plot_x, plot_y, plot_z = x, y, z
    
    # Plot start and end points
    ax1.scatter(plot_x[0], plot_y[0], plot_z[0], color='green', s=150, label='Start', marker='o')
    ax1.scatter(plot_x[-1], plot_y[-1], plot_z[-1], color='red', s=150, label='End', marker='s')
    
    # Add time-based coloring for trajectory points
    if len(df) > 100:
        scatter = ax1.scatter(plot_x[::20], plot_y[::20], plot_z[::20], 
                            c=df['time_relative'][::20], 
                            cmap='viridis', s=20, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, pad=0.1)
        cbar.set_label('Time (seconds)')
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax_limits = np.array([ax1.get_xlim3d(), ax1.get_ylim3d(), ax1.get_zlim3d()])
    ranges = ax_limits[:, 1] - ax_limits[:, 0]
    max_range = ranges.max()
    centers = ax_limits.mean(axis=1)
    for i, (center, ax_set) in enumerate(zip(centers, [ax1.set_xlim3d, ax1.set_ylim3d, ax1.set_zlim3d])):
        ax_set(center - max_range/2, center + max_range/2)
    
    # Position vs time comparison
    ax2 = fig.add_subplot(2, 2, 2)
    t = df['time_relative'].values
    
    if show_raw:
        ax2.plot(t, df['x'], 'r-', alpha=0.4, linewidth=1)
        ax2.plot(t, df['y'], 'g-', alpha=0.4, linewidth=1)
        ax2.plot(t, df['z'], 'b-', alpha=0.4, linewidth=1)
    
    if has_smooth:
        ax2.plot(t, df['x_smooth'], 'r-', label='X', linewidth=2)
        ax2.plot(t, df['y_smooth'], 'g-', label='Y', linewidth=2)
        ax2.plot(t, df['z_smooth'], 'b-', label='Z', linewidth=2)
    else:
        ax2.plot(t, df['x'], 'r-', label='X', linewidth=2)
        ax2.plot(t, df['y'], 'g-', label='Y', linewidth=2)
        ax2.plot(t, df['z'], 'b-', label='Z', linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position Components vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Velocity components
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, df['vx'], 'r-', label='Vx', linewidth=1.5, alpha=0.8)
    ax3.plot(t, df['vy'], 'g-', label='Vy', linewidth=1.5, alpha=0.8)
    ax3.plot(t, df['vz'], 'b-', label='Vz', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Components vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Speed vs time
    ax4 = fig.add_subplot(2, 2, 4)
    if 'speed' in df.columns:
        ax4.plot(t, df['speed'], 'purple', linewidth=2)
        ax4.fill_between(t, 0, df['speed'], alpha=0.3, color='purple')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed vs Time')
        ax4.grid(True, alpha=0.3)
        
        # Add average speed line
        avg_speed = df['speed'].mean()
        ax4.axhline(y=avg_speed, color='orange', linestyle='--', 
                   label=f'Average: {avg_speed:.2f} m/s')
        ax4.legend()
    
    plt.tight_layout()
    return fig

def plot_trajectory_analysis(df, has_smooth):
    """Create additional analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # XY projection (top view)
    if has_smooth:
        x, y = df['x_smooth'].values, df['y_smooth'].values
    else:
        x, y = df['x'].values, df['y'].values
    
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.scatter(x[0], y[0], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(x[-1], y[-1], color='red', s=100, label='End', zorder=5)
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('XY Projection (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # XZ projection (side view)
    if has_smooth:
        z = df['z_smooth'].values
    else:
        z = df['z'].values
    
    ax2.plot(x, z, 'b-', linewidth=2)
    ax2.scatter(x[0], z[0], color='green', s=100, label='Start', zorder=5)
    ax2.scatter(x[-1], z[-1], color='red', s=100, label='End', zorder=5)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('XZ Projection (Side View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Speed histogram
    if 'speed' in df.columns:
        ax3.hist(df['speed'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax3.axvline(df['speed'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["speed"].mean():.2f} m/s')
        ax3.set_xlabel('Speed (m/s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Speed Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Position heatmap (2D density)
    from scipy.stats import gaussian_kde
    if len(df) > 100:
        # Create 2D kernel density estimate
        xy = np.vstack([x, y])
        z_kde = gaussian_kde(xy)(xy)
        
        scatter = ax4.scatter(x, y, c=z_kde, s=20, cmap='hot', alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label='Density')
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Position Density (XY)')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    else:
        ax4.plot(x, y, 'b-', linewidth=2)
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('XY Trajectory')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Plot OptiTrack trajectory data from CSV files')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--no-raw', action='store_true', help='Hide raw trajectory (show only smoothed)')
    parser.add_argument('--analysis', action='store_true', help='Show additional analysis plots')
    parser.add_argument('--save', action='store_true', help='Save plots without asking')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures (default: 300)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found!")
        return
    
    # Load and process data
    df, has_smooth = load_and_process_data(args.csv_file)
    if df is None:
        return
    
    # Print statistics
    print_statistics(df, has_smooth)
    
    # Create main plot
    try:
        fig1 = plot_3d_trajectory(df, has_smooth, show_raw=not args.no_raw)
        
        # Create analysis plots if requested
        if args.analysis:
            fig2 = plot_trajectory_analysis(df, has_smooth)
        
        # Show plots
        plt.show()
        
        # Save plots
        if args.save or input("\nSave plots? (y/n): ").lower().strip() in ['y', 'yes']:
            base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
            
            # Save main trajectory plot
            output_name1 = f"{base_name}_trajectory.png"
            fig1.savefig(output_name1, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved: {output_name1}")
            
            # Save analysis plot if created
            if args.analysis:
                output_name2 = f"{base_name}_analysis.png"
                fig2.savefig(output_name2, dpi=args.dpi, bbox_inches='tight')
                print(f"Saved: {output_name2}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()