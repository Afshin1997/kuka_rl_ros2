import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from pathlib import Path
import seaborn as sns

class TrajectoryAnalyzer:
    """Analyze and compare simulation vs real-world ball trajectories"""
    
    def __init__(self, sim_dir=".", real_dir="dataset", g=9.81):
        self.sim_dir = sim_dir
        self.real_dir = real_dir
        self.g = g
        self.sim_data = {}
        self.real_data = {}
        
    def load_simulation_data(self, max_files=100):
        """Load simulation CSV files from env_X_data.csv format"""
        print("Loading simulation data...")
        loaded_count = 0
        
        for i in range(max_files):
            filename = f"env_{i}_data.csv"
            filepath = os.path.join(self.sim_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    # Extract relevant columns
                    sim_cols = {
                        'ball_pos_x': 'tennisball_pos_0',
                        'ball_pos_y': 'tennisball_pos_1', 
                        'ball_pos_z': 'tennisball_pos_2',
                        'ball_vel_x': 'tennisball_lin_vel_0',
                        'ball_vel_y': 'tennisball_lin_vel_1',
                        'ball_vel_z': 'tennisball_lin_vel_2'
                    }
                    
                    extracted_df = pd.DataFrame()
                    for new_col, old_col in sim_cols.items():
                        if old_col in df.columns:
                            extracted_df[new_col] = df[old_col]
                    
                    self.sim_data[f"env_{i}"] = extracted_df
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        print(f"Loaded {loaded_count} simulation files")
        return loaded_count
    
    def load_real_data(self):
        """Load real-world trajectory data from dataset folder"""
        print("Loading real-world data...")
        loaded_count = 0
        
        csv_files = [f for f in os.listdir(self.real_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            filepath = os.path.join(self.real_dir, csv_file)
            try:
                df = pd.read_csv(filepath)
                # Check for required columns
                required_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z']
                if all(col in df.columns for col in required_cols):
                    # Calculate velocities if not present
                    if 'ball_ema_vx' in df.columns:
                        df['ball_vel_x'] = df['ball_ema_vx']
                        df['ball_vel_y'] = df['ball_ema_vy']
                        df['ball_vel_z'] = df['ball_ema_vz']
                    else:
                        # Estimate velocities from position differences
                        dt = 0.01  # Assuming 100 Hz sampling
                        df['ball_vel_x'] = df['ball_pos_x'].diff() / dt
                        df['ball_vel_y'] = df['ball_pos_y'].diff() / dt
                        df['ball_vel_z'] = df['ball_pos_z'].diff() / dt
                        df.iloc[0, df.columns.get_loc('ball_vel_x'):df.columns.get_loc('ball_vel_z')+1] = 0
                    
                    trajectory_name = os.path.splitext(csv_file)[0]
                    self.real_data[trajectory_name] = df
                    loaded_count += 1
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                
        print(f"Loaded {loaded_count} real-world files")
        return loaded_count
    
    def extract_initial_conditions(self, data_dict):
        """Extract initial position and velocity from trajectories"""
        initial_conditions = []
        
        for name, df in data_dict.items():
            if len(df) > 0:
                init_data = {
                    'name': name,
                    'init_pos_x': df.iloc[0]['ball_pos_x'],
                    'init_pos_y': df.iloc[0]['ball_pos_y'],
                    'init_pos_z': df.iloc[0]['ball_pos_z'],
                    'init_vel_x': df.iloc[0]['ball_vel_x'],
                    'init_vel_y': df.iloc[0]['ball_vel_y'],
                    'init_vel_z': df.iloc[0]['ball_vel_z'],
                    'trajectory_length': len(df)
                }
                
                # Calculate initial speed and launch angle
                speed = np.sqrt(init_data['init_vel_x']**2 + 
                               init_data['init_vel_y']**2 + 
                               init_data['init_vel_z']**2)
                
                # Launch angle (from horizontal)
                horizontal_speed = np.sqrt(init_data['init_vel_x']**2 + 
                                          init_data['init_vel_y']**2)
                launch_angle = np.degrees(np.arctan2(init_data['init_vel_z'], 
                                                     horizontal_speed))
                
                init_data['init_speed'] = speed
                init_data['launch_angle'] = launch_angle
                initial_conditions.append(init_data)
                
        return pd.DataFrame(initial_conditions)
    
    def calculate_trajectory_metrics(self, df):
        """Calculate key metrics for a trajectory"""
        metrics = {}
        
        # Max height
        metrics['max_height'] = df['ball_pos_z'].max()
        metrics['max_height_idx'] = df['ball_pos_z'].idxmax()
        
        # Range (horizontal distance)
        metrics['range_x'] = df['ball_pos_x'].iloc[-1] - df['ball_pos_x'].iloc[0]
        metrics['range_y'] = df['ball_pos_y'].iloc[-1] - df['ball_pos_y'].iloc[0]
        metrics['total_range'] = np.sqrt(metrics['range_x']**2 + metrics['range_y']**2)
        
        # Flight time (assuming data is cut at landing)
        metrics['flight_time'] = len(df) * 0.01  # Assuming 100 Hz
        
        # Average velocities
        metrics['avg_vel_x'] = df['ball_vel_x'].mean()
        metrics['avg_vel_y'] = df['ball_vel_y'].mean()
        metrics['avg_vel_z'] = df['ball_vel_z'].mean()
        
        # Velocity at peak
        peak_idx = metrics['max_height_idx']
        if peak_idx < len(df):
            metrics['vel_at_peak_x'] = df.iloc[peak_idx]['ball_vel_x']
            metrics['vel_at_peak_y'] = df.iloc[peak_idx]['ball_vel_y']
            metrics['vel_at_peak_z'] = df.iloc[peak_idx]['ball_vel_z']
        
        return metrics
    
    def compare_multiple_timesteps(self, timesteps=[0, 39, 79, 119]):
        """Compare trajectories at multiple timesteps to see evolution"""
        print(f"\n" + "="*60)
        print(f"MULTI-TIMESTEP COMPARISON")
        print("="*60)
        
        # Create a figure with subplots for each timestep
        fig, axes = plt.subplots(len(timesteps), 6, figsize=(18, 4*len(timesteps)))
        fig.suptitle('Evolution of Ball State Over Time: Simulation vs Reality', fontsize=16)
        
        for idx, timestep in enumerate(timesteps):
            sim_at_timestep = []
            real_at_timestep = []
            
            # Extract data at specific timestep for simulation
            for name, df in self.sim_data.items():
                if len(df) > timestep:
                    data = {
                        'pos_x': df.iloc[timestep]['ball_pos_x'],
                        'pos_y': df.iloc[timestep]['ball_pos_y'],
                        'pos_z': df.iloc[timestep]['ball_pos_z'],
                        'vel_x': df.iloc[timestep]['ball_vel_x'],
                        'vel_y': df.iloc[timestep]['ball_vel_y'],
                        'vel_z': df.iloc[timestep]['ball_vel_z']
                    }
                    sim_at_timestep.append(data)
            
            # Extract data at specific timestep for real data
            for name, df in self.real_data.items():
                if len(df) > timestep:
                    data = {
                        'pos_x': df.iloc[timestep]['ball_pos_x'],
                        'pos_y': df.iloc[timestep]['ball_pos_y'],
                        'pos_z': df.iloc[timestep]['ball_pos_z'],
                        'vel_x': df.iloc[timestep]['ball_vel_x'],
                        'vel_y': df.iloc[timestep]['ball_vel_y'],
                        'vel_z': df.iloc[timestep]['ball_vel_z']
                    }
                    real_at_timestep.append(data)
            
            sim_df = pd.DataFrame(sim_at_timestep)
            real_df = pd.DataFrame(real_at_timestep)
            
            # Plot for this timestep
            for i, (param_type, param) in enumerate([('pos', 'x'), ('pos', 'y'), ('pos', 'z'),
                                                      ('vel', 'x'), ('vel', 'y'), ('vel', 'z')]):
                ax = axes[idx, i]
                col = f'{param_type}_{param}'
                
                if len(sim_df) > 0 and col in sim_df.columns:
                    ax.hist(sim_df[col].values, alpha=0.5, label='Sim', bins=15, color='blue', density=True)
                if len(real_df) > 0 and col in real_df.columns:
                    ax.hist(real_df[col].values, alpha=0.5, label='Real', bins=15, color='red', density=True)
                
                if idx == 0:
                    ax.set_title(f'{param_type.upper()}_{param.upper()}')
                if idx == len(timesteps) - 1:
                    ax.set_xlabel(f'{"Position" if param_type == "pos" else "Velocity"} ({"m" if param_type == "pos" else "m/s"})')
                if i == 0:
                    ax.set_ylabel(f't={timestep*0.01:.2f}s\nDensity', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                if idx == 0 and i == 0:
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig('timestep_evolution_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Print summary table
        print("\nSummary Statistics Table:")
        print("-"*100)
        print(f"{'Timestep':<10} {'Time(s)':<8} {'Parameter':<10} {'Sim Mean':<12} {'Real Mean':<12} {'Difference':<12} {'Sim Range':<20} {'Real Range':<20}")
        print("-"*100)
        
        for timestep in timesteps:
            sim_at_timestep = []
            real_at_timestep = []
            
            for name, df in self.sim_data.items():
                if len(df) > timestep:
                    sim_at_timestep.append({
                        'pos_x': df.iloc[timestep]['ball_pos_x'],
                        'pos_y': df.iloc[timestep]['ball_pos_y'],
                        'pos_z': df.iloc[timestep]['ball_pos_z'],
                        'vel_x': df.iloc[timestep]['ball_vel_x'],
                        'vel_y': df.iloc[timestep]['ball_vel_y'],
                        'vel_z': df.iloc[timestep]['ball_vel_z']
                    })
            
            for name, df in self.real_data.items():
                if len(df) > timestep:
                    real_at_timestep.append({
                        'pos_x': df.iloc[timestep]['ball_pos_x'],
                        'pos_y': df.iloc[timestep]['ball_pos_y'],
                        'pos_z': df.iloc[timestep]['ball_pos_z'],
                        'vel_x': df.iloc[timestep]['ball_vel_x'],
                        'vel_y': df.iloc[timestep]['ball_vel_y'],
                        'vel_z': df.iloc[timestep]['ball_vel_z']
                    })
            
            if sim_at_timestep and real_at_timestep:
                sim_df = pd.DataFrame(sim_at_timestep)
                real_df = pd.DataFrame(real_at_timestep)
                
                for param in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']:
                    sim_mean = sim_df[param].mean()
                    real_mean = real_df[param].mean()
                    sim_range = f"[{sim_df[param].min():.3f}, {sim_df[param].max():.3f}]"
                    real_range = f"[{real_df[param].min():.3f}, {real_df[param].max():.3f}]"
                    
                    print(f"{timestep:<10} {timestep*0.01:<8.2f} {param:<10} {sim_mean:<12.3f} {real_mean:<12.3f} {real_mean-sim_mean:<12.3f} {sim_range:<20} {real_range:<20}")
            
            if timestep != timesteps[-1]:
                print("-"*100)
    
    def compare_at_timestep(self, timestep=79):
        """Compare trajectories at a specific timestep"""
        print(f"\n" + "="*60)
        print(f"COMPARISON AT TIMESTEP {timestep}")
        print("="*60)
        
        sim_at_timestep = []
        real_at_timestep = []
        
        # Extract data at specific timestep for simulation
        for name, df in self.sim_data.items():
            if len(df) > timestep:
                data = {
                    'name': name,
                    'pos_x': df.iloc[timestep]['ball_pos_x'],
                    'pos_y': df.iloc[timestep]['ball_pos_y'],
                    'pos_z': df.iloc[timestep]['ball_pos_z'],
                    'vel_x': df.iloc[timestep]['ball_vel_x'],
                    'vel_y': df.iloc[timestep]['ball_vel_y'],
                    'vel_z': df.iloc[timestep]['ball_vel_z']
                }
                sim_at_timestep.append(data)
        
        # Extract data at specific timestep for real data
        for name, df in self.real_data.items():
            if len(df) > timestep:
                data = {
                    'name': name,
                    'pos_x': df.iloc[timestep]['ball_pos_x'],
                    'pos_y': df.iloc[timestep]['ball_pos_y'],
                    'pos_z': df.iloc[timestep]['ball_pos_z'],
                    'vel_x': df.iloc[timestep]['ball_vel_x'],
                    'vel_y': df.iloc[timestep]['ball_vel_y'],
                    'vel_z': df.iloc[timestep]['ball_vel_z']
                }
                real_at_timestep.append(data)
        
        sim_df = pd.DataFrame(sim_at_timestep)
        real_df = pd.DataFrame(real_at_timestep)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'State at Timestep {timestep} (t={timestep*0.01:.2f}s): Simulation vs Reality', fontsize=16)
        
        # Position comparisons
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[0, i]
            col = f'pos_{axis}'
            
            if len(sim_df) > 0 and col in sim_df.columns:
                ax.hist(sim_df[col].values, alpha=0.5, label='Simulation', bins=20, color='blue')
            if len(real_df) > 0 and col in real_df.columns:
                ax.hist(real_df[col].values, alpha=0.5, label='Reality', bins=20, color='red')
            
            ax.set_xlabel(f'Position {axis.upper()} (m)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Position {axis.upper()} at t={timestep*0.01:.2f}s')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Velocity comparisons
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[1, i]
            col = f'vel_{axis}'
            
            if len(sim_df) > 0 and col in sim_df.columns:
                ax.hist(sim_df[col].values, alpha=0.5, label='Simulation', bins=20, color='blue')
            if len(real_df) > 0 and col in real_df.columns:
                ax.hist(real_df[col].values, alpha=0.5, label='Reality', bins=20, color='red')
            
            ax.set_xlabel(f'Velocity {axis.upper()} (m/s)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Velocity {axis.upper()} at t={timestep*0.01:.2f}s')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'timestep_{timestep}_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Statistical comparison
        print(f"\nStatistical Comparison at Timestep {timestep} (t={timestep*0.01:.2f}s):")
        print("-"*50)
        
        for param in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']:
            if len(sim_df) > 0 and len(real_df) > 0:
                sim_mean = sim_df[param].mean()
                sim_std = sim_df[param].std()
                sim_min = sim_df[param].min()
                sim_max = sim_df[param].max()
                
                real_mean = real_df[param].mean()
                real_std = real_df[param].std()
                real_min = real_df[param].min()
                real_max = real_df[param].max()
                
                print(f"\n{param}:")
                print(f"  Simulation: μ={sim_mean:6.3f} ± σ={sim_std:5.3f}  Range=[{sim_min:6.3f}, {sim_max:6.3f}]")
                print(f"  Reality:    μ={real_mean:6.3f} ± σ={real_std:5.3f}  Range=[{real_min:6.3f}, {real_max:6.3f}]")
                print(f"  Difference: Δμ={real_mean - sim_mean:6.3f}  (Reality - Simulation)")
                
                # Perform t-test if we have enough samples
                if len(sim_df) > 1 and len(real_df) > 1:
                    t_stat, p_value = stats.ttest_ind(sim_df[param], real_df[param])
                    print(f"  T-test: t={t_stat:6.3f}, p={p_value:.4f}", end="")
                    if p_value < 0.05:
                        print(" *** Significant difference ***")
                    else:
                        print()
        
        print(f"\nNumber of trajectories compared:")
        print(f"  Simulation: {len(sim_df)} trajectories")
        print(f"  Reality: {len(real_df)} trajectories")
        
        return sim_df, real_df
    
    def compare_initial_conditions(self):
        """Compare initial conditions between simulation and reality"""
        sim_init = self.extract_initial_conditions(self.sim_data)
        real_init = self.extract_initial_conditions(self.real_data)
        
        if len(sim_init) == 0 and len(real_init) == 0:
            print("No data available for comparison")
            return sim_init, real_init
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Initial Conditions: Simulation vs Reality', fontsize=16)
        
        # Position comparisons
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[0, i]
            col = f'init_pos_{axis}'
            
            if len(sim_init) > 0 and col in sim_init.columns:
                ax.hist(sim_init[col].values, alpha=0.5, label='Simulation', bins=20, color='blue')
            if len(real_init) > 0 and col in real_init.columns:
                ax.hist(real_init[col].values, alpha=0.5, label='Reality', bins=20, color='red')
            
            ax.set_xlabel(f'Initial Position {axis.upper()} (m)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Initial Position {axis.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Velocity comparisons
        for i, axis in enumerate(['x', 'y', 'z']):
            ax = axes[1, i]
            col = f'init_vel_{axis}'
            
            if len(sim_init) > 0 and col in sim_init.columns:
                ax.hist(sim_init[col].values, alpha=0.5, label='Simulation', bins=20, color='blue')
            if len(real_init) > 0 and col in real_init.columns:
                ax.hist(real_init[col].values, alpha=0.5, label='Reality', bins=20, color='red')
            
            ax.set_xlabel(f'Initial Velocity {axis.upper()} (m/s)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Initial Velocity {axis.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('initial_conditions_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Statistical comparison
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON OF INITIAL CONDITIONS")
        print("="*60)
        
        for param in ['init_pos_x', 'init_pos_y', 'init_pos_z', 
                     'init_vel_x', 'init_vel_y', 'init_vel_z']:
            if len(sim_init) > 0 and len(real_init) > 0:
                sim_mean = sim_init[param].mean()
                sim_std = sim_init[param].std()
                real_mean = real_init[param].mean()
                real_std = real_init[param].std()
                
                print(f"\n{param}:")
                print(f"  Simulation: μ = {sim_mean:.3f} ± σ = {sim_std:.3f}")
                print(f"  Reality:    μ = {real_mean:.3f} ± σ = {real_std:.3f}")
                print(f"  Difference: Δμ = {real_mean - sim_mean:.3f}")
                
                # Perform t-test
                if len(sim_init) > 1 and len(real_init) > 1:
                    t_stat, p_value = stats.ttest_ind(sim_init[param], real_init[param])
                    print(f"  T-test: t = {t_stat:.3f}, p = {p_value:.4f}")
                    if p_value < 0.05:
                        print(f"  *** Significant difference detected ***")
        
        return sim_init, real_init
    
    def plot_trajectory_comparison(self, num_trajectories=5):
        """Plot side-by-side trajectory comparisons"""
        fig = plt.figure(figsize=(16, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        
        # 2D projections
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # Plot simulation trajectories
        sim_keys = list(self.sim_data.keys())[:num_trajectories]
        for i, key in enumerate(sim_keys):
            df = self.sim_data[key]
            color = plt.cm.Blues(i / max(num_trajectories, 1))
            
            # Convert to numpy arrays for plotting
            x = df['ball_pos_x'].values
            y = df['ball_pos_y'].values
            z = df['ball_pos_z'].values
            
            ax1.plot(x, y, z, 
                    label=f'Sim {i+1}', color=color, alpha=0.7)
            ax3.plot(x, z, 
                    color=color, alpha=0.7)
        
        # Plot real trajectories
        real_keys = list(self.real_data.keys())[:num_trajectories]
        for i, key in enumerate(real_keys):
            df = self.real_data[key]
            color = plt.cm.Reds(i / max(num_trajectories, 1))
            
            # Convert to numpy arrays for plotting
            x = df['ball_pos_x'].values
            y = df['ball_pos_y'].values
            z = df['ball_pos_z'].values
            
            ax2.plot(x, y, z, 
                    label=f'Real {i+1}', color=color, alpha=0.7)
            ax4.plot(x, z, 
                    color=color, alpha=0.7)
        
        # Format 3D plots
        for ax, title in [(ax1, 'Simulation Trajectories'), 
                          (ax2, 'Real Trajectories')]:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Format 2D plots
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Simulation: Side View (X-Z)')
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('Reality: Side View (X-Z)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Trajectory Comparison: Simulation vs Reality', fontsize=14)
        plt.tight_layout()
        plt.savefig('trajectory_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def analyze_deviations(self):
        """Analyze deviations and patterns in trajectories"""
        print("\n" + "="*60)
        print("TRAJECTORY METRICS ANALYSIS")
        print("="*60)
        
        sim_metrics_list = []
        real_metrics_list = []
        
        # Calculate metrics for all trajectories
        for name, df in self.sim_data.items():
            metrics = self.calculate_trajectory_metrics(df)
            metrics['type'] = 'simulation'
            metrics['name'] = name
            sim_metrics_list.append(metrics)
        
        for name, df in self.real_data.items():
            metrics = self.calculate_trajectory_metrics(df)
            metrics['type'] = 'reality'
            metrics['name'] = name
            real_metrics_list.append(metrics)
        
        # Convert to DataFrames
        sim_metrics_df = pd.DataFrame(sim_metrics_list)
        real_metrics_df = pd.DataFrame(real_metrics_list)
        
        # Key metrics comparison
        metrics_to_compare = ['max_height', 'total_range', 'flight_time']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Key Trajectory Metrics Comparison', fontsize=14)
        
        for i, metric in enumerate(metrics_to_compare):
            ax = axes[i]
            
            data_to_plot = []
            labels = []
            
            if len(sim_metrics_df) > 0 and metric in sim_metrics_df.columns:
                sim_data = sim_metrics_df[metric].dropna().values
                if len(sim_data) > 0:
                    data_to_plot.append(sim_data)
                    labels.append('Simulation')
            
            if len(real_metrics_df) > 0 and metric in real_metrics_df.columns:
                real_data = real_metrics_df[metric].dropna().values
                if len(real_data) > 0:
                    data_to_plot.append(real_data)
                    labels.append('Reality')
            
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trajectory_metrics_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        for metric in metrics_to_compare:
            print(f"\n{metric.replace('_', ' ').title()}:")
            if len(sim_metrics_df) > 0 and metric in sim_metrics_df.columns:
                print(f"  Simulation: μ = {sim_metrics_df[metric].mean():.3f} ± σ = {sim_metrics_df[metric].std():.3f}")
            if len(real_metrics_df) > 0 and metric in real_metrics_df.columns:
                print(f"  Reality:    μ = {real_metrics_df[metric].mean():.3f} ± σ = {real_metrics_df[metric].std():.3f}")
        
        return sim_metrics_df, real_metrics_df
    
    def calculate_physics_parameters(self):
        """Estimate physics parameters from trajectories"""
        print("\n" + "="*60)
        print("PHYSICS PARAMETERS ESTIMATION")
        print("="*60)
        
        def estimate_g_from_trajectory(df):
            """Estimate gravity from parabolic trajectory"""
            # Use the vertical motion equation: z = z0 + vz0*t - 0.5*g*t^2
            # At the peak, vz = 0, so we can estimate g
            
            max_idx = df['ball_pos_z'].idxmax()
            if max_idx > 0 and max_idx < len(df) - 1:
                # Time to peak (assuming constant dt)
                t_peak = max_idx * 0.01
                
                # Initial vertical velocity
                vz0 = df.iloc[0]['ball_vel_z']
                
                # At peak, vz = vz0 - g*t = 0
                if t_peak > 0:
                    g_est = vz0 / t_peak
                    return g_est
            return None
        
        # Estimate gravity for each trajectory
        sim_g_estimates = []
        real_g_estimates = []
        
        for name, df in self.sim_data.items():
            g_est = estimate_g_from_trajectory(df)
            if g_est is not None and g_est > 0:
                sim_g_estimates.append(g_est)
        
        for name, df in self.real_data.items():
            g_est = estimate_g_from_trajectory(df)
            if g_est is not None and g_est > 0:
                real_g_estimates.append(g_est)
        
        if sim_g_estimates:
            print(f"\nEstimated gravity (simulation): {np.mean(sim_g_estimates):.2f} ± {np.std(sim_g_estimates):.2f} m/s²")
        if real_g_estimates:
            print(f"Estimated gravity (reality):    {np.mean(real_g_estimates):.2f} ± {np.std(real_g_estimates):.2f} m/s²")
        print(f"Expected gravity:                9.81 m/s²")
        
        # Energy analysis
        print("\n" + "-"*40)
        print("ENERGY ANALYSIS")
        print("-"*40)
        
        def calculate_energy(df, mass=0.052):
            """Calculate kinetic and potential energy"""
            # Kinetic energy
            v_squared = df['ball_vel_x']**2 + df['ball_vel_y']**2 + df['ball_vel_z']**2
            KE = 0.5 * mass * v_squared
            
            # Potential energy (relative to initial height)
            PE = mass * 9.81 * (df['ball_pos_z'] - df.iloc[0]['ball_pos_z'])
            
            # Total energy
            E_total = KE + PE
            
            return KE, PE, E_total
        
        # Calculate energy dissipation
        sim_energy_loss = []
        real_energy_loss = []
        
        for name, df in list(self.sim_data.items())[:10]:  # Sample first 10
            KE, PE, E_total = calculate_energy(df)
            if len(E_total) > 1:
                energy_loss = (E_total.iloc[0] - E_total.iloc[-1]) / E_total.iloc[0] * 100
                sim_energy_loss.append(energy_loss)
        
        for name, df in list(self.real_data.items())[:10]:
            KE, PE, E_total = calculate_energy(df)
            if len(E_total) > 1:
                energy_loss = (E_total.iloc[0] - E_total.iloc[-1]) / E_total.iloc[0] * 100
                real_energy_loss.append(energy_loss)
        
        if sim_energy_loss:
            print(f"Average energy loss (simulation): {np.mean(sim_energy_loss):.1f}%")
        if real_energy_loss:
            print(f"Average energy loss (reality):    {np.mean(real_energy_loss):.1f}%")
    
    def generate_report(self, output_file="trajectory_analysis_report.txt"):
        """Generate a comprehensive analysis report"""
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAJECTORY ANALYSIS REPORT: SIMULATION VS REALITY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Simulation files analyzed: {len(self.sim_data)}\n")
            f.write(f"Real-world files analyzed: {len(self.real_data)}\n\n")
            
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            # Run analyses
            self.compare_initial_conditions()
            self.analyze_deviations()
            self.calculate_physics_parameters()
            
            # Restore stdout
            sys.stdout = original_stdout
            
        print(f"\nReport saved to {output_file}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("="*60)
        print("STARTING FULL TRAJECTORY ANALYSIS")
        print("="*60)
        
        # Load data
        sim_count = self.load_simulation_data()
        real_count = self.load_real_data()
        
        if sim_count == 0 and real_count == 0:
            print("No data files found! Please check your directories.")
            return
        
        # Run analyses
        self.compare_initial_conditions()
        self.compare_at_timestep(timestep=79)  # Compare at 79th timestep
        self.plot_trajectory_comparison()
        self.analyze_deviations()
        self.calculate_physics_parameters()
        self.generate_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  - initial_conditions_comparison.png")
        print("  - timestep_79_comparison.png")
        print("  - trajectory_comparison.png")
        print("  - trajectory_metrics_comparison.png")
        print("  - trajectory_analysis_report.txt")


def main():
    """Main function to run the analysis"""
    # Configure paths
    SIMULATION_DIR = "recorded_trajectories/."  # Directory with env_X_data.csv files
    REAL_DATA_DIR = "dataset_realistic"  # Directory with real trajectory CSV files
    
    # Create analyzer
    analyzer = TrajectoryAnalyzer(
        sim_dir=SIMULATION_DIR,
        real_dir=REAL_DATA_DIR
    )
    
    # Load data first
    sim_count = analyzer.load_simulation_data()
    real_count = analyzer.load_real_data()
    
    if sim_count == 0 and real_count == 0:
        print("No data files found! Please check your directories.")
        return
    
    # Run specific analyses
    print("\n" + "="*60)
    print("ANALYSIS OPTIONS")
    print("="*60)
    print("1. Full analysis (all comparisons)")
    print("2. Compare at timestep 79 only")
    print("3. Compare multiple timesteps")
    print("4. Initial conditions only")
    
    choice = input("\nEnter choice (1-4, or press Enter for option 2): ").strip()
    
    if choice == '1' or choice == '':
        analyzer.compare_initial_conditions()
        analyzer.compare_at_timestep(timestep=79)
        analyzer.compare_multiple_timesteps([0, 39, 79, 119])
        analyzer.plot_trajectory_comparison()
        analyzer.analyze_deviations()
        analyzer.calculate_physics_parameters()
        analyzer.generate_report()
    elif choice == '2':
        analyzer.compare_at_timestep(timestep=79)
    elif choice == '3':
        timesteps_input = input("Enter timesteps separated by commas (default: 0,39,79,119): ").strip()
        if timesteps_input:
            try:
                timesteps = [int(t.strip()) for t in timesteps_input.split(',')]
            except:
                timesteps = [0, 39, 79, 119]
        else:
            timesteps = [0, 39, 79, 119]
        analyzer.compare_multiple_timesteps(timesteps)
    elif choice == '4':
        analyzer.compare_initial_conditions()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()