#!/usr/bin/env python3
"""
3D Trajectory Plotter for OptiTrack Data
This script reads all trajectory CSV files from a specified folder and creates
3D scatter plots of the EMA world positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import glob
import argparse
from datetime import datetime
import sys

class TrajectoryPlotter:
    def __init__(self, trajectory_folder='marker_trajectories'):
        self.trajectory_folder = trajectory_folder
        self.all_positions = []
        self.trajectory_data = []  # Store data for each trajectory separately
        self.colors = []
        
        # Recording zone boundaries (from the original script)
        self.x_min = 0.0
        self.x_max = 3.0
        self.z_min = 0.0
        
    def load_trajectories(self):
        """Load all trajectory CSV files from the specified folder"""
        csv_files = sorted(glob.glob(os.path.join(self.trajectory_folder, "trajectory_*.csv")))
        
        if len(csv_files) == 0:
            print(f"No trajectory files found in '{self.trajectory_folder}'")
            return False
        
        print(f"Found {len(csv_files)} trajectory files")
        
        # Use different colors for different trajectories
        colormap = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))
        
        for idx, csv_file in enumerate(csv_files):
            try:
                trajectory_positions = []
                trajectory_info = {
                    'filename': os.path.basename(csv_file),
                    'positions': [],
                    'color': colormap[idx]
                }
                
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # Extract EMA world positions
                        x = float(row['world_x_ema'])
                        y = float(row['world_y_ema'])
                        z = float(row['world_z_ema'])
                        position = [x, y, z]
                        trajectory_positions.append(position)
                        self.all_positions.append(position)
                        self.colors.append(colormap[idx])
                
                trajectory_info['positions'] = np.array(trajectory_positions)
                self.trajectory_data.append(trajectory_info)
                print(f"  Loaded {len(trajectory_positions)} points from {trajectory_info['filename']}")
            
            except Exception as e:
                print(f"  Error reading {csv_file}: {str(e)}")
        
        return len(self.all_positions) > 0
    
    def plot_3d_scatter(self, show_individual=False, save_plot=True, show_plot=True):
        """Create 3D scatter plot of all trajectories"""
        if len(self.all_positions) == 0:
            print("No data to plot")
            return
        
        positions = np.array(self.all_positions)
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_individual:
            # Plot each trajectory with its own color and label
            for traj in self.trajectory_data:
                if len(traj['positions']) > 0:
                    ax.scatter(traj['positions'][:, 0], 
                             traj['positions'][:, 1], 
                             traj['positions'][:, 2],
                             c=[traj['color']], 
                             alpha=0.6, 
                             s=20,
                             label=traj['filename'][:20] + '...' if len(traj['filename']) > 20 else traj['filename'])
        else:
            # Plot all points with their respective colors
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               c=self.colors, alpha=0.6, s=20)
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D Scatter Plot of Marker Trajectories (EMA World Positions)', fontsize=14)
        
        # Add recording zone boundaries
        self._draw_recording_zone(ax)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text2D(0.02, 0.98, f"Trajectories: {len(self.trajectory_data)}\nTotal points: {len(positions)}", 
                  transform=ax.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Add legend if showing individual trajectories
        if show_individual and len(self.trajectory_data) <= 10:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)
        
        # Save plot
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = os.path.join(self.trajectory_folder, 
                                       f"trajectory_plot_3d_{timestamp}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {plot_filename}")
        
        if show_plot:
            plt.show()
        
        return fig, ax
    
    def plot_projections(self, save_plot=True, show_plot=True):
        """Create 2D projections of the trajectories (XY, XZ, YZ planes)"""
        if len(self.all_positions) == 0:
            print("No data to plot")
            return
        
        positions = np.array(self.all_positions)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY projection
        axes[0].scatter(positions[:, 0], positions[:, 1], c=self.colors, alpha=0.6, s=20)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('XY Plane Projection')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # XZ projection
        axes[1].scatter(positions[:, 0], positions[:, 2], c=self.colors, alpha=0.6, s=20)
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].set_title('XZ Plane Projection')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')
        # Add recording zone
        axes[1].axvline(x=self.x_min, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(x=self.x_max, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=self.z_min, color='r', linestyle='--', alpha=0.5)
        
        # YZ projection
        axes[2].scatter(positions[:, 1], positions[:, 2], c=self.colors, alpha=0.6, s=20)
        axes[2].set_xlabel('Y (m)')
        axes[2].set_ylabel('Z (m)')
        axes[2].set_title('YZ Plane Projection')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_aspect('equal')
        axes[2].axhline(y=self.z_min, color='r', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'2D Projections of {len(self.trajectory_data)} Trajectories ({len(positions)} points)', 
                     fontsize=14)
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = os.path.join(self.trajectory_folder, 
                                       f"trajectory_projections_{timestamp}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved projections to: {plot_filename}")
        
        if show_plot:
            plt.show()
    
    def create_animation(self, output_filename=None):
        """Create a rotating animation of the 3D plot"""
        try:
            import imageio
        except ImportError:
            print("Please install 'imageio' to create animations: pip install imageio")
            return
        
        if len(self.all_positions) == 0:
            print("No data to animate")
            return
        
        fig, ax = self.plot_3d_scatter(show_plot=False, save_plot=False)
        
        print("Creating rotating animation...")
        images = []
        
        for angle in range(0, 360, 3):
            ax.view_init(elev=20, azim=angle)
            fig.canvas.draw()
            
            # Convert to numpy array
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(buf)
        
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = os.path.join(self.trajectory_folder, 
                                         f"trajectory_animation_{timestamp}.gif")
        
        imageio.mimsave(output_filename, images, fps=15)
        print(f"Saved animation to: {output_filename}")
        plt.close(fig)
    
    def _draw_recording_zone(self, ax):
        """Draw the recording zone boundaries on the 3D plot"""
        # Get plot limits
        y_min, y_max = ax.get_ylim()
        
        # Draw recording zone as a box
        # Bottom rectangle
        x_corners = [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min]
        y_corners = [y_min, y_min, y_max, y_max, y_min]
        z_corners = [self.z_min] * 5
        
        ax.plot(x_corners, y_corners, z_corners, 'r--', alpha=0.5, linewidth=2)
        
        # Vertical lines
        for x, y in [(self.x_min, y_min), (self.x_max, y_min), 
                     (self.x_max, y_max), (self.x_min, y_max)]:
            ax.plot([x, x], [y, y], [self.z_min, ax.get_zlim()[1]], 'r--', alpha=0.3, linewidth=1)
        
        # Add text label
        ax.text(np.mean([self.x_min, self.x_max]), y_min, self.z_min, 
                'Recording Zone', color='red', fontsize=10, alpha=0.7)
    
    def get_statistics(self):
        """Print statistics about the loaded trajectories"""
        if len(self.trajectory_data) == 0:
            print("No data loaded")
            return
        
        print("\n=== Trajectory Statistics ===")
        print(f"Total trajectories: {len(self.trajectory_data)}")
        print(f"Total points: {len(self.all_positions)}")
        
        positions = np.array(self.all_positions)
        print(f"\nPosition ranges:")
        print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
        print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
        print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
        
        print(f"\nIndividual trajectories:")
        for i, traj in enumerate(self.trajectory_data):
            print(f"  {i+1}. {traj['filename']}: {len(traj['positions'])} points")


def main():
    parser = argparse.ArgumentParser(description='Plot 3D trajectories from OptiTrack CSV files')
    parser.add_argument('--folder', '-f', type=str, default='marker_trajectories',
                       help='Folder containing trajectory CSV files (default: marker_trajectories)')
    parser.add_argument('--no-3d', action='store_true',
                       help='Skip 3D scatter plot')
    parser.add_argument('--no-projections', action='store_true',
                       help='Skip 2D projection plots')
    parser.add_argument('--animate', '-a', action='store_true',
                       help='Create rotating animation (requires imageio)')
    parser.add_argument('--individual', '-i', action='store_true',
                       help='Show individual trajectories with labels')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Show trajectory statistics')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to files')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    # Create plotter instance
    plotter = TrajectoryPlotter(args.folder)
    
    # Load data
    if not plotter.load_trajectories():
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        plotter.get_statistics()
    
    # Create plots
    if not args.no_3d:
        plotter.plot_3d_scatter(
            show_individual=args.individual,
            save_plot=not args.no_save,
            show_plot=not args.no_show
        )
    
    if not args.no_projections:
        plotter.plot_projections(
            save_plot=not args.no_save,
            show_plot=not args.no_show
        )
    
    if args.animate:
        plotter.create_animation()


if __name__ == '__main__':
    main()