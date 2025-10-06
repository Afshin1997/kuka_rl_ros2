#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class BallTrajectoryPlotter:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load trajectory data from CSV file"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} data points from {self.csv_file}")
            
            # Calculate time relative to start
            self.df['time'] = self.df['timestamp'] - self.df['timestamp'].iloc[0]
            
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)
    
    def plot_positions_and_velocities(self):
        """Create plots for positions and velocities"""
        # Create figure with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Ball Position and Velocity Analysis\n{os.path.basename(self.csv_file)}', 
                     fontsize=16, fontweight='bold')
        
        time = self.df['time']
        
        # Position plots (top row)
        # X position
        axes[0, 0].plot(time, self.df['x_world'], 'r-', linewidth=2)
        axes[0, 0].set_ylabel('X Position [m]')
        axes[0, 0].set_title('X Position vs Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel('Time [s]')
        
        # Y position
        axes[0, 1].plot(time, self.df['y_world'], 'g-', linewidth=2)
        axes[0, 1].set_ylabel('Y Position [m]')
        axes[0, 1].set_title('Y Position vs Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel('Time [s]')
        
        # Z position
        axes[0, 2].plot(time, self.df['z_world'], 'b-', linewidth=2)
        axes[0, 2].set_ylabel('Z Position [m]')
        axes[0, 2].set_title('Z Position vs Time')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlabel('Time [s]')
        
        # Velocity plots (bottom row)
        # X velocity
        axes[1, 0].plot(time, self.df['vx_world'], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Vx [m/s]')
        axes[1, 0].set_title('X Velocity vs Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        # Y velocity
        axes[1, 1].plot(time, self.df['vy_world'], 'g--', linewidth=2)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Vy [m/s]')
        axes[1, 1].set_title('Y Velocity vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        # Z velocity
        axes[1, 2].plot(time, self.df['vz_world'], 'b--', linewidth=2)
        axes[1, 2].set_xlabel('Time [s]')
        axes[1, 2].set_ylabel('Vz [m/s]')
        axes[1, 2].set_title('Z Velocity vs Time')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0, color='k', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        # Print statistics
        self.print_statistics()
        
        plt.show()
    
    def plot_separate_figures(self):
        """Create separate figures for positions and velocities"""
        # Figure 1: Positions only
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
        fig1.suptitle(f'Ball Positions\n{os.path.basename(self.csv_file)}', 
                      fontsize=16, fontweight='bold')
        
        time = self.df['time']
        
        # X position
        axes1[0].plot(time, self.df['x_world'], 'r-', linewidth=2.5)
        axes1[0].set_xlabel('Time [s]')
        axes1[0].set_ylabel('X Position [m]')
        axes1[0].set_title('X Position vs Time')
        axes1[0].grid(True, alpha=0.3)
        
        # Y position
        axes1[1].plot(time, self.df['y_world'], 'g-', linewidth=2.5)
        axes1[1].set_xlabel('Time [s]')
        axes1[1].set_ylabel('Y Position [m]')
        axes1[1].set_title('Y Position vs Time')
        axes1[1].grid(True, alpha=0.3)
        
        # Z position
        axes1[2].plot(time, self.df['z_world'], 'b-', linewidth=2.5)
        axes1[2].set_xlabel('Time [s]')
        axes1[2].set_ylabel('Z Position [m]')
        axes1[2].set_title('Z Position vs Time')
        axes1[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Print statistics
        self.print_statistics()
        
        plt.show()
    
    def print_statistics(self):
        """Print trajectory statistics"""
        print("\n" + "="*50)
        print("TRAJECTORY STATISTICS")
        print("="*50)
        print(f"Duration: {self.df['time'].iloc[-1]:.3f} seconds")
        print(f"Number of points: {len(self.df)}")
        print(f"Average sampling rate: {len(self.df)/self.df['time'].iloc[-1]:.1f} Hz")
        print("\nPosition ranges:")
        print(f"  X: [{self.df['x_world'].min():.3f}, {self.df['x_world'].max():.3f}] m")
        print(f"  Y: [{self.df['y_world'].min():.3f}, {self.df['y_world'].max():.3f}] m")
        print(f"  Z: [{self.df['z_world'].min():.3f}, {self.df['z_world'].max():.3f}] m")
        print("\nVelocity statistics:")
        print(f"  Vx: [{self.df['vx_world'].min():.3f}, {self.df['vx_world'].max():.3f}] m/s")
        print(f"  Vy: [{self.df['vy_world'].min():.3f}, {self.df['vy_world'].max():.3f}] m/s")
        print(f"  Vz: [{self.df['vz_world'].min():.3f}, {self.df['vz_world'].max():.3f}] m/s")
        print(f"  Max Speed: {self.df['speed_raw'].max():.3f} m/s")
        print(f"  Mean Speed: {self.df['speed_raw'].mean():.3f} m/s")
        print("\nTrajectory endpoints:")
        print(f"  Start: ({self.df['x_world'].iloc[0]:.3f}, {self.df['y_world'].iloc[0]:.3f}, {self.df['z_world'].iloc[0]:.3f}) m")
        print(f"  End:   ({self.df['x_world'].iloc[-1]:.3f}, {self.df['y_world'].iloc[-1]:.3f}, {self.df['z_world'].iloc[-1]:.3f}) m")
        print("="*50 + "\n")


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_ball_trajectory.py <csv_file> [--separate]")
        print("\nOptions:")
        print("  --separate : Create separate figures for positions and velocities")
        print("\nExample:")
        print("  python plot_ball_trajectory.py ball_trajectory_20240119_143052.csv")
        print("  python plot_ball_trajectory.py ball_trajectory_20240119_143052.csv --separate")
        
        # List available CSV files
        if os.path.exists('ball_trajectories'):
            files = [f for f in os.listdir('ball_trajectories') if f.endswith('.csv')]
            if files:
                print("\nAvailable trajectory files:")
                for f in sorted(files):
                    print(f"  - ball_trajectories/{f}")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(csv_file):
        # Try looking in ball_trajectories directory
        alt_path = os.path.join('ball_trajectories', csv_file)
        if os.path.exists(alt_path):
            csv_file = alt_path
        else:
            print(f"Error: File '{csv_file}' not found")
            sys.exit(1)
    
    # Create plotter and generate plots
    plotter = BallTrajectoryPlotter(csv_file)
    
    # Check for separate flag
    if '--separate' in sys.argv:
        plotter.plot_separate_figures()
    else:
        plotter.plot_positions_and_velocities()


if __name__ == '__main__':
    main()