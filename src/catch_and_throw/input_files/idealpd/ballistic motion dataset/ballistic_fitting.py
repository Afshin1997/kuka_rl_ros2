import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import least_squares, differential_evolution
from scipy.signal import savgol_filter
import glob
from mpl_toolkits.mplot3d import Axes3D

class BallisticParameterFitter:
    def __init__(self, mass=0.052, diameter=0.072, temperature=25, altitude=0):
        """
        Initialize the ballistic parameter fitter
        
        Parameters:
        - mass: ball mass in kg (0.052 kg)
        - diameter: ball diameter in meters (0.072 m)
        - temperature: air temperature in Celsius (27°C)
        - altitude: altitude above sea level in meters (0 m)
        """
        self.mb = mass  # kg
        self.db = diameter  # m
        self.g = 9.81  # m/s^2
        
        # Calculate air density based on temperature and altitude
        # Standard air density at sea level and 15°C is 1.225 kg/m³
        # Adjust for temperature: ρ = ρ0 * (T0/T)
        T_kelvin = temperature + 273.15
        T0_kelvin = 288.15  # Standard temperature (15°C)
        rho_0 = 1.225  # kg/m³ at sea level, 15°C
        
        # Temperature correction
        self.rho_a = rho_0 * (T0_kelvin / T_kelvin)
        
        # Altitude correction (if needed)
        if altitude > 0:
            # Barometric formula
            self.rho_a *= np.exp(-altitude * 0.00012)  # Approximate
        
        print(f"Calculated air density: {self.rho_a:.4f} kg/m³")
        
        # Initial guess for drag coefficient
        self.cw_initial = 0.45  # Standard for sphere
        
        # Storage for results
        self.trajectories = {}
        self.fitted_parameters = {}
        self.validation_results = {}
        
    # def ballistic_ode(self, t, state, cw):
    #     """
    #     ODE for ballistic motion with air drag (Equation 19 from paper)
        
    #     state = [x, y, z, vx, vy, vz]
    #     """
    #     pos = state[:3]
    #     vel = state[3:]
        
    #     # Gravity vector
    #     g_vec = np.array([0, 0, -self.g])  # Note: Z is up in your data
        
    #     # Air drag
    #     v_mag = np.linalg.norm(vel)
    #     if v_mag > 0:
    #         drag_coeff = (cw * np.pi * self.db**2 * self.rho_a) / (2 * self.mb)
    #         drag = -drag_coeff * v_mag * vel
    #     else:
    #         drag = np.zeros(3)

        
    #     # Acceleration
    #     acc = g_vec + drag
        
    #     # Return derivatives
    #     return np.concatenate([vel, acc])

    def ballistic_ode(self, t, state, cw, include_magnus=True):
        """
        Enhanced ODE with additional effects
        """
        pos = state[:3]
        vel = state[3:]
        
        # Gravity vector
        g_vec = np.array([0, 0, -self.g])
        
        # Air drag
        v_mag = np.linalg.norm(vel)
        if v_mag > 0:
            drag_coeff = (cw * np.pi * self.db**2 * self.rho_a) / (2 * self.mb)
            drag = -drag_coeff * v_mag * vel
            
            # Magnus effect (if ball is spinning)
            if include_magnus:
                # Assume some backspin (common in throws)
                omega = np.array([0, 0, 200.0])  # rad/s, adjust based on your setup
                magnus_coeff = 0.5 * self.rho_a * np.pi * (self.db/2)**3
                magnus = magnus_coeff * np.cross(omega, vel)
                # print(f"Magnus effect: {magnus}")
                # print(f"Drag: {drag}")
                drag += magnus
        else:
            drag = np.zeros(3)
        
        # Acceleration
        acc = g_vec + drag
        
        return np.concatenate([vel, acc])
    
    def load_trajectory_data(self, file_paths):
        """Load and preprocess trajectory data from CSV files"""
        print(f"\nLoading {len(file_paths)} trajectory files...")
        
        for file_path in file_paths:
            name = file_path.split('/')[-1].replace('.csv', '')
            
            try:
                df = pd.read_csv(file_path)

                # Check if required columns exist
                required_cols = ['ball_raw_vx', 'ball_pos_z', 'ball_pos_y']
                
                # Find stopping conditions
                # 1. ball_raw_vx is zero (or very close to zero)
                zero_vx_indices = df[abs(df['ball_raw_vx']) < 1e-6].index.tolist()
                
                # 2. ball_pos_z is zero or negative (ball hits ground/table)
                zero_pz_indices = df[df['ball_pos_z'] <= 0.01].index.tolist()  # Small tolerance above ground
                
                # 3. ball_pos_y is in the "bad" range [-0.3, -0.2] (you want to EXCLUDE this)
                # So we want to stop BEFORE it gets into this range
                bad_py_indices = df[(df['ball_pos_y'] >= -0.3)].index.tolist()
                                
                # Determine cutoff index
                cutoff_candidates = []
                
                # Add first occurrence of each condition
                if zero_vx_indices:
                    cutoff_candidates.append(zero_vx_indices[0])
                if zero_pz_indices:
                    cutoff_candidates.append(zero_pz_indices[0])
                if bad_py_indices:
                    cutoff_candidates.append(bad_py_indices[0])
                
                if cutoff_candidates:
                    # Use the earliest stopping condition, minus 2 rows as buffer
                    cutoff_index = min(cutoff_candidates) - 2
                    cutoff_index = max(0, cutoff_index)  # Don't go negative
                    
                    print(f"File {name}: Cutting at index {cutoff_index} (original data had {len(df)} rows)")
                    df = df.iloc[:cutoff_index]
                
                # Check required columns
                required_cols = ['ball_pos_x', 'ball_pos_y', 'ball_pos_z',
                               'ball_raw_vx', 'ball_raw_vy', 'ball_raw_vz']
                
                if all(col in df.columns for col in required_cols):
                    # Store trajectory data
                    self.trajectories[name] = {
                        'data': df,
                        'positions': df[['ball_pos_x', 'ball_pos_y', 'ball_pos_z']].values,
                        'velocities': df[['ball_raw_vx', 'ball_raw_vy', 'ball_raw_vz']].values,
                        'times': np.arange(len(df)) * 0.01  # Assuming 100Hz sampling
                    }
                    print(f"  Loaded {name}: {len(df)} samples")
                else:
                    print(f"  Skipping {name}: missing required columns")
                    
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    def estimate_initial_conditions(self, trajectory_data, n_points=5):
        """
        Estimate initial velocity using first few points
        More robust than using single differentiation
        """
        positions = trajectory_data['positions'][:n_points]
        times = trajectory_data['times'][:n_points]
        
        # Fit polynomial to first few points for each dimension
        velocities = []
        for dim in range(3):
            # Fit 2nd order polynomial
            coeffs = np.polyfit(times, positions[:, dim], 2)
            # Velocity is derivative of position
            vel_coeffs = np.polyder(coeffs)
            # Evaluate at t=0
            v0 = np.polyval(vel_coeffs, times[0])
            velocities.append(v0)
        
        return positions[0], np.array(velocities)
    
    def simulate_trajectory(self, p0, v0, t_span, cw):
        """Simulate trajectory using ballistic model"""
        # Initial state
        initial_state = np.concatenate([p0, v0])
        
        # Solve ODE
        sol = solve_ivp(self.ballistic_ode, 
                       [t_span[0], t_span[-1]], 
                       initial_state, 
                       args=(cw,),
                       t_eval=t_span,
                       method='RK45',
                       rtol=1e-8)
        
        if sol.success:
            return sol.y[:3, :].T  # Return positions only
        else:
            return None
    
    def objective_function(self, cw, trajectory_name):
        """Objective function for single trajectory"""
        traj_data = self.trajectories[trajectory_name]
        
        # Get initial conditions
        p0, v0 = self.estimate_initial_conditions(traj_data)
        
        # Simulate with current cw
        t_span = traj_data['times']
        predicted_pos = self.simulate_trajectory(p0, v0, t_span, cw)
        
        if predicted_pos is None:
            return np.ones(len(t_span) * 3) * 1e6  # Large penalty
        
        # Compute residuals
        measured_pos = traj_data['positions']
        residuals = (measured_pos - predicted_pos).flatten()
        
        return residuals
    

    def fit_single_trajectory(self, trajectory_name, method='lm'):
        """Fit drag coefficient for a single trajectory"""
        print(f"\nFitting trajectory: {trajectory_name}")
        
        if method == 'lm':
            # Levenberg-Marquardt (no bounds)
            result = least_squares(
                lambda cw: self.objective_function(cw, trajectory_name),
                [self.cw_initial],
                method='lm'  # Removed bounds
            )
        else:
            # Differential Evolution (global optimization)
            result = differential_evolution(
                lambda cw: np.sum(self.objective_function(cw, trajectory_name)**2),
                bounds=[(0.1, 1.0)],
                seed=42
            )
        
        cw_fitted = result.x[0]
        
        # Calculate RMSE
        residuals = self.objective_function(cw_fitted, trajectory_name)
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Store results
        self.fitted_parameters[trajectory_name] = {
            'cw': cw_fitted,
            'rmse': rmse,
            'success': result.success,
            'result': result
        }
        
        print(f"  Fitted cw: {cw_fitted:.4f}")
        print(f"  RMSE: {rmse:.4f} m")
        
        return cw_fitted, rmse
    
    def fit_all_trajectories(self, method='lm'):
        """Fit drag coefficient for all trajectories"""
        all_cw = []
        all_rmse = []
        
        for name in self.trajectories.keys():
            cw, rmse = self.fit_single_trajectory(name, method)
            all_cw.append(cw)
            all_rmse.append(rmse)
        
        # Calculate global statistics
        self.global_cw = np.mean(all_cw)
        self.global_cw_std = np.std(all_cw)
        
        print(f"\n{'='*50}")
        print(f"GLOBAL RESULTS:")
        print(f"Mean cw: {self.global_cw:.4f} ± {self.global_cw_std:.4f}")
        print(f"Range: [{np.min(all_cw):.4f}, {np.max(all_cw):.4f}]")
        print(f"Mean RMSE: {np.mean(all_rmse):.4f} m")
        print(f"{'='*50}")
        
        return self.global_cw
    
    def validate_fit(self, trajectory_name, cw=None):
        """Validate the fit for a trajectory"""
        if cw is None:
            cw = self.fitted_parameters[trajectory_name]['cw']
        
        traj_data = self.trajectories[trajectory_name]
        p0, v0 = self.estimate_initial_conditions(traj_data)
        
        # Simulate with fitted parameters
        predicted_pos = self.simulate_trajectory(p0, v0, traj_data['times'], cw)
        
        # Calculate errors
        measured_pos = traj_data['positions']
        errors = measured_pos - predicted_pos
        
        # Store validation results
        self.validation_results[trajectory_name] = {
            'predicted': predicted_pos,
            'measured': measured_pos,
            'errors': errors,
            'rmse_per_axis': np.sqrt(np.mean(errors**2, axis=0)),
            'max_error_per_axis': np.max(np.abs(errors), axis=0)
        }
        
        return errors
    
    def plot_results(self, trajectory_names=None, save_plots=False):
        """Comprehensive plotting of results"""
        if trajectory_names is None:
            trajectory_names = list(self.trajectories.keys())[:6]  # First 3
        
        # Plot 1: 3D trajectories comparison
        fig1 = plt.figure(figsize=(15, 10))
        
        for i, name in enumerate(trajectory_names):
            ax = fig1.add_subplot(2, 3, i+1, projection='3d')
            
            # Validate if not done
            if name not in self.validation_results:
                self.validate_fit(name)
            
            val_results = self.validation_results[name]
            measured = val_results['measured']
            predicted = val_results['predicted']
            
            # Plot trajectories
            ax.plot(measured[:, 0], measured[:, 1], measured[:, 2], 
                   'b-', label='Measured', linewidth=2)
            ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], 
                   'r--', label='Predicted', linewidth=2)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{name}\ncw={self.fitted_parameters[name]["cw"]:.3f}')
            ax.legend()
        
        plt.suptitle('Measured vs Predicted Trajectories', fontsize=16)
        plt.tight_layout()
        
        # Plot 2: Error analysis
        fig2, axes = plt.subplots(3, 1, figsize=(12, 10))
        axis_labels = ['X', 'Y', 'Z']
        
        for name in trajectory_names:
            val_results = self.validation_results[name]
            errors = val_results['errors']
            times = self.trajectories[name]['times']
            
            for i, ax in enumerate(axes):
                ax.plot(times, errors[:, i] * 1000, label=name)  # Convert to mm
                ax.set_ylabel(f'{axis_labels[i]} Error (mm)')
                ax.grid(True, alpha=0.3)
        
        axes[0].set_title('Position Errors Over Time')
        axes[0].legend()
        axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()
                
        if save_plots:
            fig1.savefig('trajectory_comparison.png', dpi=300)
            fig2.savefig('error_analysis.png', dpi=300)
            # fig3.savefig('parameter_distribution.png', dpi=300)
        
        plt.show()
    
    def save_results(self, filename='ballistic_parameters.txt'):
        """Save fitting results to file"""
        with open(filename, 'w') as f:
            f.write("BALLISTIC MODEL PARAMETER FITTING RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Ball mass: {self.mb} kg\n")
            f.write(f"Ball diameter: {self.db} m\n")
            f.write(f"Air density: {self.rho_a:.4f} kg/m³\n")
            f.write(f"Gravity: {self.g} m/s²\n\n")
            
            f.write(f"GLOBAL DRAG COEFFICIENT: {self.global_cw:.4f} ± {self.global_cw_std:.4f}\n\n")
            
            f.write("Individual trajectory results:\n")
            f.write("-"*60 + "\n")
            
            for name, params in self.fitted_parameters.items():
                f.write(f"\nTrajectory: {name}\n")
                f.write(f"  cw: {params['cw']:.4f}\n")
                f.write(f"  RMSE: {params['rmse']*1000:.2f} mm\n")
                
                if name in self.validation_results:
                    val = self.validation_results[name]
                    f.write(f"  RMSE per axis (mm): X={val['rmse_per_axis'][0]*1000:.2f}, ")
                    f.write(f"Y={val['rmse_per_axis'][1]*1000:.2f}, ")
                    f.write(f"Z={val['rmse_per_axis'][2]*1000:.2f}\n")
        
        print(f"\nResults saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Initialize the fitter with your parameters
    fitter = BallisticParameterFitter(
        mass=0.052,      # kg
        diameter=0.072,  # m
        temperature=25,  # Celsius
        altitude=0       # m above sea level
    )
    
    # Load all CSV files
    file_paths = glob.glob("*.csv")
    file_paths.sort()
    
    if not file_paths:
        print("No CSV files found!")
    else:
        # Load trajectory data
        fitter.load_trajectory_data(file_paths)
        
        # Fit all trajectories
        optimal_cw = fitter.fit_all_trajectories(method='lm')
        # optimal_cw = fitter.fit_all_trajectories()
        
        # Validate and plot results
        fitter.plot_results(save_plots=True)
        
        # Save results
        fitter.save_results()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Optimal drag coefficient: {optimal_cw:.4f}")
        print(f"Expected range: 0.40-0.50 for smooth sphere")
        print(f"Your value is {'within' if 0.40 <= optimal_cw <= 0.50 else 'outside'} the expected range")
        print(f"{'='*60}")