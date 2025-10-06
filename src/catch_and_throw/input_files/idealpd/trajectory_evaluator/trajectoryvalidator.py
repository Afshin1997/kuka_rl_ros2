import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class TrajectoryValidator:
    def __init__(self, init_pos=(3.0, -0.45, 0.25), final_pos=(0.51, -0.45, 0.65),
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
        
        # Calculate expected velocity ranges based on position ranges and time
        self._calculate_velocity_ranges()
    
    def _calculate_velocity_ranges(self):
        """Calculate expected velocity ranges from position and time ranges"""
        # Sample many random trajectories to estimate velocity ranges
        n_samples = 10000
        
        # Random initial positions
        x_init = self.init_pos[0] + (np.random.rand(n_samples) * 2 - 1) * self.x_range
        y_init = self.init_pos[1] + (np.random.rand(n_samples) * 2 - 1) * self.y_range
        z_init = self.init_pos[2] + (np.random.rand(n_samples) * 2 - 1) * self.z_range
        
        # Random final positions (following your random_throw logic)
        x_final = self.final_pos[0] - np.random.rand(n_samples) * self.x_range * 2.0
        y_final = self.final_pos[1] - np.random.rand(n_samples) * 0.4 + 0.1
        z_final = self.final_pos[2] + np.random.rand(n_samples) * self.z_range * 2.0 + 0.1
        
        # Random flight times
        t = self.t_min + (self.t_max - self.t_min) * np.random.rand(n_samples)
        
        # Calculate velocities
        vx = (x_final - x_init) / t
        vy = (y_final - y_init) / t
        vz = (z_final - z_init + 0.5 * self.g * t * t) / t
        
        # Store velocity ranges (using 1st and 99th percentiles for robustness)
        self.velocity_ranges = {
            'vx': (np.percentile(vx, 1), np.percentile(vx, 99)),
            'vy': (np.percentile(vy, 1), np.percentile(vy, 99)),
            'vz': (np.percentile(vz, 1), np.percentile(vz, 99))
        }
        
        # Store samples for visualization
        self.velocity_samples = {'vx': vx, 'vy': vy, 'vz': vz}
    
    def load_trajectories(self, file_paths):
        """Load all trajectory CSV files (first 80 rows only)"""
        trajectories = {}
        for file_path in file_paths:
            name = file_path.split('/')[-1].replace('.csv', '')
            df = pd.read_csv(file_path, nrows=100)  # Read only first 80 rows
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
    
    def validate_trajectory(self, pos, vel, final_pos_actual=None):
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
        
        # Check initial velocities
        for i, v_comp in enumerate(['vx', 'vy', 'vz']):
            in_range = self.velocity_ranges[v_comp][0] <= vel[i] <= self.velocity_ranges[v_comp][1]
            results['details'][f'init_{v_comp}'] = {
                'value': vel[i],
                'range': self.velocity_ranges[v_comp],
                'in_range': in_range
            }
        
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
            
            # Check final position - use actual if provided, otherwise compute expected
            if final_pos_actual is not None:
                final_pos = final_pos_actual
            else:
                final_pos = self.compute_final_position(pos, vel, t_flight)
                
            # Fixed: Use correct keys for final_ranges
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
            # Only analyze the first row (initial state) for each trajectory
            # Assuming each CSV file contains one trajectory
            if len(df) > 0:
                # Get initial state (first row)
                first_row = df.iloc[0]
                pos = np.array([first_row['ball_pos_x'], first_row['ball_pos_y'], first_row['ball_pos_z']])
                vel = np.array([first_row['ball_ema_vx'], first_row['ball_ema_vy'], first_row['ball_ema_vz']])
                
                # Get final state (last row) for validation
                last_row = df.iloc[-1]
                final_pos_actual = np.array([last_row['ball_pos_x'], last_row['ball_pos_y'], last_row['ball_pos_z']])
                
                result = self.validate_trajectory(pos, vel, final_pos_actual)
                trajectory_results.append(result)
            all_results[name] = trajectory_results
        
        return all_results
    
    def plot_validation_summary(self, trajectories, results):
        """Create comprehensive validation plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: 3D scatter of initial positions with expected final positions
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # Plot initial positions
        for name, df in trajectories.items():
            ax1.scatter(df['ball_pos_x'], df['ball_pos_y'], df['ball_pos_z'], label=name, alpha=0.6, s=30)
        
        # Calculate and plot expected final positions
        all_final_positions = []
        for name, df in trajectories.items():
            for idx, row in df.iterrows():
                pos = np.array([row['ball_pos_x'], row['ball_pos_y'], row['ball_pos_z']])
                vel = np.array([row['ball_ema_vx'], row['ball_ema_vy'], row['ball_ema_vz']])
                t_flight = self.estimate_flight_time(pos, vel)
                if t_flight and t_flight > 0:
                    final_pos = self.compute_final_position(pos, vel, t_flight)
                    all_final_positions.append(final_pos)
        
        # Draw initial range box (red)
        self._draw_box(ax1, self.init_ranges, 'r', 'Initial Range')
        
        # Draw final range box (blue)
        self._draw_box(ax1, self.final_ranges, 'b', 'Final Range')
        
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Expected Initial Positions and Final Positions vs Real Data')
        ax1.legend(loc='best', fontsize=8)
        
        ax5 = fig.add_subplot(1, 2, 2)
        velocities = ['vx', 'vy', 'vz']
        
        # Collect first row velocity data
        init_vel_data = {'vx': [], 'vy': [], 'vz': []}
        for name, df in trajectories.items():
            if len(df) > 0:
                init_vel_data['vx'].append(df.iloc[0]['ball_ema_vx'])
                init_vel_data['vy'].append(df.iloc[0]['ball_ema_vy'])
                init_vel_data['vz'].append(df.iloc[0]['ball_ema_vz'])
        
        # Plot actual data as violin plots
        positions = np.arange(len(velocities))
        for i, vel in enumerate(velocities):
            data = init_vel_data[vel]
            parts = ax5.violinplot([data], positions=[i], widths=0.7, 
                                  showmeans=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(['red', 'green', 'blue'][i])
                pc.set_alpha(0.6)
            
            # Add range lines for expected velocities
            ax5.hlines(self.velocity_ranges[vel][0], i-0.4, i+0.4, colors='black', 
                      linestyles='dashed', alpha=0.7, linewidth=2)
            ax5.hlines(self.velocity_ranges[vel][1], i-0.4, i+0.4, colors='black', 
                      linestyles='dashed', alpha=0.7, linewidth=2)
            
            # Add text labels for ranges
            ax5.text(i, self.velocity_ranges[vel][1] + 0.5, 
                    f'[{self.velocity_ranges[vel][0]:.1f}, {self.velocity_ranges[vel][1]:.1f}]',
                    ha='center', fontsize=9)
        
        ax5.set_xticks(positions)
        ax5.set_xticklabels(velocities)
        ax5.set_ylabel('Velocity (m/s)')
        ax5.set_title('Initial Velocity Distributions in training vs Real Data')
        
        ax5.set_ylim(bottom=min(min(self.velocity_ranges[v]) for v in velocities) - 2,
                     top=max(max(self.velocity_ranges[v]) for v in velocities) + 3)
        
        
        plt.tight_layout()
        plt.savefig('trajectory_validation_summary.png')
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
        
        print(f"\nExpected Initial Velocity Ranges (from training):")
        for vel in ['vx', 'vy', 'vz']:
            print(f"  {vel}: [{self.velocity_ranges[vel][0]:.3f}, {self.velocity_ranges[vel][1]:.3f}] m/s")
        
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

            # With this:
            print(f"\n  Position statistics:")
            pos_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
            axis_labels = ['X', 'Y', 'Z']
            for col, axis in zip(pos_cols, axis_labels):
                print(f"    {axis}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, " +
                    f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")

            # Similarly for velocity statistics, replace:
            print(f"\n  Velocity statistics:")
            for axis in ['vx', 'vy', 'vz']:
                print(f"    {axis}: mean={df[axis].mean():.3f}, std={df[axis].std():.3f}, " +
                    f"range=[{df[axis].min():.3f}, {df[axis].max():.3f}]")

            # With this:
            print(f"\n  Velocity statistics:")
            vel_cols = ['ball_ema_vx', 'ball_ema_vy', 'ball_ema_vz']
            vel_labels = ['vx', 'vy', 'vz']
            for col, vel in zip(vel_cols, vel_labels):
                print(f"    {vel}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, " +
                    f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")
                # Show velocity statistics          
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
    # file_paths = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv', '11.csv', '12.csv', '13.csv', '14.csv', '15.csv', '16.csv', '17.csv', '18.csv', '19.csv']
    file_paths = ['2.csv', '3.csv']
    trajectories = validator.load_trajectories(file_paths)
    
    # Analyze all trajectories
    results = validator.analyze_all_trajectories(trajectories)
    
    # Create visualizations
    validator.plot_validation_summary(trajectories, results)
    
    # Print detailed report
    # validator.print_detailed_report(trajectories, results)