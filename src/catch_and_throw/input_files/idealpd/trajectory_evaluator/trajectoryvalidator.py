import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class TrajectoryValidator:
    def __init__(self, init_pos=(3.0, -0.45, 0.5), final_pos=(0.51, -0.45, 0.65),
                 x_range=0.3, y_range=0.25, z_range=0.2, g=9.81, t_min=0.8, t_max=1.0):
        """Initialize validator with training parameters"""
        self.init_pos = np.array(init_pos)
        self.final_pos = np.array(final_pos)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.g = g
        self.t_min = t_min
        self.t_max = t_max
        
        # Define expected ranges based on your random_throw function
        self.init_ranges = {
            'X': (init_pos[0] - x_range, init_pos[0] + x_range),
            'Y': (init_pos[1] - y_range, init_pos[1] + y_range),
            'Z': (init_pos[2] - z_range, init_pos[2] + z_range)
        }
        
        self.final_ranges = {
            'X': (final_pos[0] - 0.6, final_pos[0]),  # -0.6 to 0
            'Y': (final_pos[1] - 0.2, final_pos[1] + 0.1),  # -0.2 to +0.1
            'Z': (final_pos[2], final_pos[2] + 0.4)  # 0 to 0.4
        }
    
    def load_trajectories(self, file_paths):
        """Load all trajectory CSV files"""
        trajectories = {}
        for file_path in file_paths:
            name = file_path.split('/')[-1].replace('.csv', '')
            df = pd.read_csv(file_path)
            trajectories[name] = df
        return trajectories
    
    def estimate_flight_time(self, pos, vel):
        """Estimate flight time from initial position and velocity"""
        # Using quadratic formula for z = z0 + vz*t - 0.5*g*t^2
        # Assuming ball lands at approximately z=0
        a = -0.5 * self.g
        b = vel[2]
        c = pos[2]
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Return positive time
        return max(t1, t2) if max(t1, t2) > 0 else None
    
    def compute_final_position(self, pos, vel, t):
        """Compute final position given initial conditions and time"""
        x_final = pos[0] + vel[0] * t
        y_final = pos[1] + vel[1] * t
        z_final = pos[2] + vel[2] * t - 0.5 * self.g * t**2
        return np.array([x_final, y_final, z_final])
    
    def validate_trajectory(self, pos, vel):
        """Validate if a single trajectory is within expected ranges"""
        results = {
            'initial_in_range': True,
            'final_in_range': True,
            'flight_time_valid': True,
            'details': {}
        }
        
        # Check initial position
        for i, axis in enumerate(['X', 'Y', 'Z']):
            in_range = self.init_ranges[axis][0] <= pos[i] <= self.init_ranges[axis][1]
            results['details'][f'init_{axis}'] = {
                'value': pos[i],
                'range': self.init_ranges[axis],
                'in_range': in_range
            }
            if not in_range:
                results['initial_in_range'] = False
        
        # Estimate flight time
        t_flight = self.estimate_flight_time(pos, vel)
        if t_flight is None or not (self.t_min <= t_flight <= self.t_max):
            results['flight_time_valid'] = False
            results['details']['flight_time'] = {
                'value': t_flight,
                'range': (self.t_min, self.t_max),
                'in_range': False
            }
        else:
            results['details']['flight_time'] = {
                'value': t_flight,
                'range': (self.t_min, self.t_max),
                'in_range': True
            }
            
            # Check final position
            final_pos = self.compute_final_position(pos, vel, t_flight)
            for i, axis in enumerate(['X', 'Y', 'Z']):
                in_range = self.final_ranges[axis][0] <= final_pos[i] <= self.final_ranges[axis][1]
                results['details'][f'final_{axis}'] = {
                    'value': final_pos[i],
                    'range': self.final_ranges[axis],
                    'in_range': in_range
                }
                if not in_range:
                    results['final_in_range'] = False
        
        return results
    
    def analyze_all_trajectories(self, trajectories):
        """Analyze all trajectories and return summary statistics"""
        all_results = {}
        
        for name, df in trajectories.items():
            trajectory_results = []
            for idx, row in df.iterrows():
                pos = np.array([row['X'], row['Y'], row['Z']])
                vel = np.array([row['vx'], row['vy'], row['vz']])
                result = self.validate_trajectory(pos, vel)
                trajectory_results.append(result)
            all_results[name] = trajectory_results
        
        return all_results
    
    def plot_validation_summary(self, trajectories, results):
        """Create comprehensive validation plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: 3D scatter of initial positions with expected final positions
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        
        # Plot initial positions
        for name, df in trajectories.items():
            ax1.scatter(df['X'], df['Y'], df['Z'], label=name, alpha=0.6, s=30)
        
        # Calculate and plot expected final positions
        all_final_positions = []
        for name, df in trajectories.items():
            for idx, row in df.iterrows():
                pos = np.array([row['X'], row['Y'], row['Z']])
                vel = np.array([row['vx'], row['vy'], row['vz']])
                t_flight = self.estimate_flight_time(pos, vel)
                if t_flight and t_flight > 0:
                    final_pos = self.compute_final_position(pos, vel, t_flight)
                    all_final_positions.append(final_pos)
        
        # if all_final_positions:
        #     all_final_positions = np.array(all_final_positions)
        #     ax1.scatter(all_final_positions[:, 0], all_final_positions[:, 1], 
        #                all_final_positions[:, 2], c='orange', marker='^', 
        #                alpha=0.4, s=20, label='Expected Final Pos')
        
        # Draw initial range box (red)
        self._draw_box(ax1, self.init_ranges, 'r', 'Initial Range')
        
        # Draw final range box (blue)
        self._draw_box(ax1, self.final_ranges, 'b', 'Final Range')
        
        # Add reference points
        # ax1.scatter(*self.init_pos, color='red', s=200, marker='o', 
        #            edgecolors='black', linewidth=2, label='Init Center')
        # ax1.scatter(*self.final_pos, color='blue', s=200, marker='*', 
        #            edgecolors='black', linewidth=2, label='Final Center')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Initial Positions and Expected Final Positions vs Ranges')
        ax1.legend(loc='best', fontsize=8)
        
        # Plot 2: Velocity distributions
        # ax2 = fig.add_subplot(2, 3, 2)
        # all_velocities = []
        # labels = []
        # for name, df in trajectories.items():
        #     velocities = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
        #     all_velocities.extend(velocities)
        #     labels.extend([name] * len(velocities))
        
        # ax2.hist(all_velocities, bins=30, alpha=0.7, edgecolor='black')
        # ax2.set_xlabel('Velocity Magnitude (m/s)')
        # ax2.set_ylabel('Count')
        # ax2.set_title('Velocity Magnitude Distribution')
        # ax2.axvline(np.mean(all_velocities), color='red', linestyle='--', 
        #             label=f'Mean: {np.mean(all_velocities):.2f} m/s')
        # ax2.legend()
        
        # Plot 3: Validation summary
        # ax3 = fig.add_subplot(2, 3, 3)
        # summary_data = []
        # for name, traj_results in results.items():
        #     total = len(traj_results)
        #     valid_init = sum(1 for r in traj_results if r['initial_in_range'])
        #     valid_final = sum(1 for r in traj_results if r['final_in_range'])
        #     valid_time = sum(1 for r in traj_results if r['flight_time_valid'])
        #     summary_data.append({
        #         'name': name,
        #         'total': total,
        #         'valid_init': valid_init,
        #         'valid_final': valid_final,
        #         'valid_time': valid_time
        #     })
        
        # summary_df = pd.DataFrame(summary_data)
        # x = np.arange(len(summary_df))
        # width = 0.2
        
        # ax3.bar(x - width, summary_df['valid_init'] / summary_df['total'] * 100, 
        #         width, label='Initial Pos Valid', alpha=0.8)
        # ax3.bar(x, summary_df['valid_final'] / summary_df['total'] * 100, 
        #         width, label='Final Pos Valid', alpha=0.8)
        # ax3.bar(x + width, summary_df['valid_time'] / summary_df['total'] * 100, 
        #         width, label='Flight Time Valid', alpha=0.8)
        
        # ax3.set_xlabel('Trajectory File')
        # ax3.set_ylabel('Percentage Valid (%)')
        # ax3.set_title('Validation Summary by File')
        # ax3.set_xticks(x)
        # ax3.set_xticklabels(summary_df['name'], rotation=45)
        # ax3.legend()
        # ax3.set_ylim(0, 105)
        
        # Plot 4: Component-wise position analysis
        # ax4 = fig.add_subplot(2, 3, 4)
        # all_data = pd.concat(trajectories.values())
        # positions = ['X', 'Y', 'Z']
        # for i, pos in enumerate(positions):
        #     data = all_data[pos]
        #     parts = ax4.violinplot([data], positions=[i], widths=0.7, 
        #                           showmeans=True, showextrema=True)
        #     for pc in parts['bodies']:
        #         pc.set_facecolor(['red', 'green', 'blue'][i])
        #         pc.set_alpha(0.6)
        
        # # Add range lines
        # for i, pos in enumerate(positions):
        #     ax4.hlines(self.init_ranges[pos][0], i-0.4, i+0.4, colors='black', 
        #               linestyles='dashed', alpha=0.5)
        #     ax4.hlines(self.init_ranges[pos][1], i-0.4, i+0.4, colors='black', 
        #               linestyles='dashed', alpha=0.5)
        
        # ax4.set_xticks(range(len(positions)))
        # ax4.set_xticklabels(positions)
        # ax4.set_ylabel('Position (m)')
        # ax4.set_title('Initial Position Distributions vs Expected Ranges')
        
        # Plot 5: Component-wise velocity analysis
        # ax5 = fig.add_subplot(2, 3, 5)
        # velocities = ['vx', 'vy', 'vz']
        # for i, vel in enumerate(velocities):
        #     data = all_data[vel]
        #     parts = ax5.violinplot([data], positions=[i], widths=0.7, 
        #                           showmeans=True, showextrema=True)
        #     for pc in parts['bodies']:
        #         pc.set_facecolor(['red', 'green', 'blue'][i])
        #         pc.set_alpha(0.6)
        
        # ax5.set_xticks(range(len(velocities)))
        # ax5.set_xticklabels(velocities)
        # ax5.set_ylabel('Velocity (m/s)')
        # ax5.set_title('Velocity Component Distributions')
        
        # Plot 6: Trajectory paths
        # ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        # colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
        
        # for (name, df), color in zip(trajectories.items(), colors):
        #     # Sample up to 10 trajectories per file
        #     sample_size = min(10, len(df))
        #     sample_indices = np.random.choice(len(df), sample_size, replace=False)
            
        #     for idx in sample_indices:
        #         row = df.iloc[idx]
        #         pos = np.array([row['X'], row['Y'], row['Z']])
        #         vel = np.array([row['vx'], row['vy'], row['vz']])
                
        #         t_flight = self.estimate_flight_time(pos, vel)
        #         if t_flight and t_flight > 0:
        #             t = np.linspace(0, t_flight, 50)
        #             x = pos[0] + vel[0] * t
        #             y = pos[1] + vel[1] * t
        #             z = pos[2] + vel[2] * t - 0.5 * self.g * t**2
                    
        #             # Only plot if trajectory is reasonable
        #             if np.all(z >= -0.1):  # Allow small negative values
        #                 ax6.plot(x, y, z, color=color, alpha=0.5, linewidth=1)
        
        # Also draw boxes in the trajectory paths plot
        # self._draw_box(ax6, self.init_ranges, 'r', 'Initial Range')
        # self._draw_box(ax6, self.final_ranges, 'b', 'Final Range')
        
        # ax6.set_xlabel('X (m)')
        # ax6.set_ylabel('Y (m)')
        # ax6.set_zlabel('Z (m)')
        # ax6.set_title('Sample Trajectory Paths')
        # ax6.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_box(self, ax, ranges, color, label):
        """Helper to draw 3D box"""
        from itertools import product
        
        x_range = ranges['X']
        y_range = ranges['Y']
        z_range = ranges['Z']
        
        vertices = list(product(x_range, y_range, z_range))
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  # x-direction
            (0, 2), (1, 3), (4, 6), (5, 7),  # y-direction
            (0, 4), (1, 5), (2, 6), (3, 7)   # z-direction
        ]
        
        for i, edge in enumerate(edges):
            points = [vertices[edge[0]], vertices[edge[1]]]
            if i == 0:  # Add label only once
                ax.plot3D(*zip(*points), f'{color}-', linewidth=2, alpha=0.5, label=label)
            else:
                ax.plot3D(*zip(*points), f'{color}-', linewidth=2, alpha=0.5)
    
    def print_detailed_report(self, trajectories, results):
        """Print detailed validation report"""
        print("="*80)
        print("TRAJECTORY VALIDATION REPORT")
        print("="*80)
        
        print(f"\nExpected Initial Position Ranges:")
        for axis in ['X', 'Y', 'Z']:
            print(f"  {axis}: [{self.init_ranges[axis][0]:.3f}, {self.init_ranges[axis][1]:.3f}] m")
        
        print(f"\nExpected Final Position Ranges:")
        for axis in ['X', 'Y', 'Z']:
            print(f"  {axis}: [{self.final_ranges[axis][0]:.3f}, {self.final_ranges[axis][1]:.3f}] m")
        
        print(f"\nExpected Flight Time Range: [{self.t_min:.2f}, {self.t_max:.2f}] s")
        
        print("\n" + "-"*80)
        
        for name, traj_results in results.items():
            df = trajectories[name]
            print(f"\nFile: {name}")
            print(f"Total trajectories: {len(traj_results)}")
            
            # Count valid trajectories
            valid_init = sum(1 for r in traj_results if r['initial_in_range'])
            valid_final = sum(1 for r in traj_results if r['final_in_range'])
            valid_time = sum(1 for r in traj_results if r['flight_time_valid'])
            fully_valid = sum(1 for r in traj_results if 
                            r['initial_in_range'] and r['final_in_range'] and r['flight_time_valid'])
            
            print(f"  Valid initial positions: {valid_init}/{len(traj_results)} ({valid_init/len(traj_results)*100:.1f}%)")
            print(f"  Valid final positions: {valid_final}/{len(traj_results)} ({valid_final/len(traj_results)*100:.1f}%)")
            print(f"  Valid flight times: {valid_time}/{len(traj_results)} ({valid_time/len(traj_results)*100:.1f}%)")
            print(f"  Fully valid trajectories: {fully_valid}/{len(traj_results)} ({fully_valid/len(traj_results)*100:.1f}%)")
            
            # Show statistics
            print(f"\n  Position statistics:")
            for axis in ['X', 'Y', 'Z']:
                print(f"    {axis}: mean={df[axis].mean():.3f}, std={df[axis].std():.3f}, " +
                      f"range=[{df[axis].min():.3f}, {df[axis].max():.3f}]")
            
            print(f"\n  Velocity statistics:")
            for axis in ['vx', 'vy', 'vz']:
                print(f"    {axis}: mean={df[axis].mean():.3f}, std={df[axis].std():.3f}, " +
                      f"range=[{df[axis].min():.3f}, {df[axis].max():.3f}]")
            
            # Show outliers
            outliers = [i for i, r in enumerate(traj_results) if not 
                       (r['initial_in_range'] and r['final_in_range'] and r['flight_time_valid'])]
            if outliers and len(outliers) <= 5:
                print(f"\n  Outlier indices: {outliers}")

# Example usage
if __name__ == "__main__":
    # Initialize validator with your training parameters
    validator = TrajectoryValidator()
    
    # Load your CSV files
    file_paths = ['pos_1.csv', 'pos_3.csv', 'pos_4.csv', 'pos_5.csv', 'pos_7.csv', 'pos_8.csv']
    trajectories = validator.load_trajectories(file_paths)
    
    # Analyze all trajectories
    results = validator.analyze_all_trajectories(trajectories)
    
    # Create visualizations
    validator.plot_validation_summary(trajectories, results)
    
    # Print detailed report
    validator.print_detailed_report(trajectories, results)