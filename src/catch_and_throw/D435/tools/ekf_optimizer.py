import numpy as np
from scipy.optimize import minimize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

# Load the CSV file
file_path = '../tracking/recorded_data/ball_tracking_data_2_tracking.csv'
data = pd.read_csv(file_path)
# data = data.iloc[:-4]
file_path_predictions = '../tracking/recorded_data/ball_tracking_data_2_predictions.csv'
predictions_data = pd.read_csv(file_path_predictions)

# Extract the relevant columns for plotting
x_world = predictions_data['x_world'].to_numpy()
y_world = predictions_data['y_world'].to_numpy()
z_world = predictions_data['z_world'].to_numpy()


time = predictions_data['dt'].cumsum().to_numpy()

e_x = predictions_data['e_x'].to_numpy()

e_y = predictions_data['e_y'].to_numpy()

e_z = predictions_data['e_z'].to_numpy()

time_predictions = predictions_data['dt'].cumsum().to_numpy()



# Adjust the tracking data's time array to start from zero
adjusted_time = time - time[0]

class EKF:
    def __init__(self, device='cpu'):
        import torch
        self.torch = torch
        self.device = torch.device(device)

        # Initialize state [x, y, z, vx, vy, vz, ax, ay, az]
        initial_position = [3600.0, -500.0, 1000.0]
        initial_velocity = [-3600.0, 0.0, 5000.0]
        initial_acc = [0.0, 0.0, 0.0]
        initial_state = initial_position + initial_velocity + initial_acc
        self.x = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Covariance: Use different values for x, y, z
        pos_var_x = 1e4
        pos_var_y = 1e5
        pos_var_z = 5e4
        vel_var_x = 1e6
        vel_var_y = 2e6
        vel_var_z = 1.5e6
        acc_var_x = 2.5e4
        acc_var_y = 3.0e4
        acc_var_z = 1.0e4

        self.P = torch.zeros((9, 9), dtype=torch.float32, device=self.device)
        self.P[0, 0] = pos_var_x  # Position variance in x
        self.P[1, 1] = pos_var_y  # Position variance in y
        self.P[2, 2] = pos_var_z  # Position variance in z
        self.P[3, 3] = vel_var_x  # Velocity variance in x
        self.P[4, 4] = vel_var_y  # Velocity variance in y
        self.P[5, 5] = vel_var_z  # Velocity variance in z
        self.P[6, 6] = acc_var_x  # Acceleration variance in x
        self.P[7, 7] = acc_var_y  # Acceleration variance in y
        self.P[8, 8] = acc_var_z  # Acceleration variance in z

        # Measurement noise
        measurement_noise_std = 20  # 2 cm
        self.R = torch.eye(3, dtype=torch.float32) * (measurement_noise_std**2)

        # Measurement matrix
        self.H = torch.zeros((3, 9), dtype=torch.float32, device=self.device)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z

        # Process noise parameters (jerk)
        self.q_jerk_x = 10
        self.q_jerk_y = 10
        self.q_jerk_z = 10

    def predict(self, dt):
        """ Prediction step for time increment dt. """
        F = self.torch.eye(9, device=self.device)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[0, 6] = 0.5*(dt**2)
        F[1, 7] = 0.5*(dt**2)
        F[2, 8] = 0.5*(dt**2)
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        Q = self.compute_process_noise(dt)

        # Predict state
        self.x = F @ self.x
        # Predict covariance
        self.P = F @ self.P @ F.T + Q

        return self.x

    def compute_process_noise(self, dt):
        Q = self.torch.zeros((9, 9), dtype=self.torch.float32, device=self.device)
        # Jerk variances
        qx = self.q_jerk_x**2
        qy = self.q_jerk_y**2
        qz = self.q_jerk_z**2

        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        # x-axis submatrix
        Q[0, 0] = qx*(dt4/4)
        Q[0, 3] = qx*(dt3/2)
        Q[0, 6] = qx*(dt2/2)
        Q[3, 0] = Q[0, 3]
        Q[3, 3] = qx*(dt2)
        Q[3, 6] = qx*dt
        Q[6, 0] = Q[0, 6]
        Q[6, 3] = Q[3, 6]
        Q[6, 6] = qx

        # y-axis submatrix
        Q[1, 1] = qy*(dt4/4)
        Q[1, 4] = qy*(dt3/2)
        Q[1, 7] = qy*(dt2/2)
        Q[4, 1] = Q[1, 4]
        Q[4, 4] = qy*(dt2)
        Q[4, 7] = qy*dt
        Q[7, 1] = Q[1, 7]
        Q[7, 4] = Q[4, 7]
        Q[7, 7] = qy

        # z-axis submatrix
        Q[2, 2] = qz*(dt4/4)
        Q[2, 5] = qz*(dt3/2)
        Q[2, 8] = qz*(dt2/2)
        Q[5, 2] = Q[2, 5]
        Q[5, 5] = qz*(dt2)
        Q[5, 8] = qz*dt
        Q[8, 2] = Q[2, 8]
        Q[8, 5] = Q[5, 8]
        Q[8, 8] = qz

        return Q

    def update(self, z):
        """ 
        Update step with measurement z = [X_meas, Y_meas, Z_meas].
        Only call this when a new measurement is actually available.
        """
        z_t = self.torch.tensor(z, dtype=self.torch.float32, device=self.device).unsqueeze(1)
        y = z_t - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ self.torch.inverse(S)

        self.x = self.x + K @ y
        I = self.torch.eye(9, device=self.device)
        self.P = (I - K @ self.H) @ self.P

        return self.x

    def get_state(self):
        return self.x.squeeze().cpu().detach().numpy()

    


def interpolate_tracking(time_ekf, time_tracking, tracking_data):
    return np.interp(time_ekf, time_tracking, tracking_data)

def loss_function(params, ekf, time_ekf, interpolated_data):
    qx, qy, qz, r_std, pos_var_x, pos_var_y, pos_var_z, vel_var_x, vel_var_y, vel_var_z, acc_var_x, acc_var_y, acc_var_z = params

    # Update EKF parameters
    ekf.q_jerk_x = qx
    ekf.q_jerk_y = qy
    ekf.q_jerk_z = qz
    ekf.R = ekf.torch.eye(3, dtype=ekf.torch.float32) * (r_std**2)
    ekf.P[0, 0] = pos_var_x
    ekf.P[1, 1] = pos_var_y
    ekf.P[2, 2] = pos_var_z
    ekf.P[3, 3] = vel_var_x
    ekf.P[4, 4] = vel_var_y
    ekf.P[5, 5] = vel_var_z
    ekf.P[6, 6] = acc_var_x
    ekf.P[7, 7] = acc_var_y
    ekf.P[8, 8] = acc_var_z

    predictions = []
    for t, z in zip(time_ekf, interpolated_data):
        ekf.predict(dt=0.004)  # EKF frequency (250 Hz)
        ekf.update(z)
        predictions.append(ekf.get_state()[:3])  # Only positions

    predictions = np.array(predictions)
    loss = np.mean(np.linalg.norm(predictions - interpolated_data, axis=1))
    print(f"Loss: {loss:.6f} for parameters: {params}")  # Debugging
    return loss



# Initialize EKF
ekf = EKF(device='cpu')

# Time and data
time_tracking = adjusted_time
data_tracking = np.vstack((x_world, y_world, z_world)).T
time_ekf = np.linspace(0, max(time_tracking), len(time_predictions))

# Interpolate tracking data to EKF time
interpolated_data = np.vstack([
    interpolate_tracking(time_ekf, time_tracking, x_world),
    interpolate_tracking(time_ekf, time_tracking, y_world),
    interpolate_tracking(time_ekf, time_tracking, z_world)
]).T

initial_params = [
    1, 1, 1, 20,  # q_jerk_x, q_jerk_y, q_jerk_z, measurement_noise_std
    1e4, 1e4, 1e4,  # pos_var_x, pos_var_y, pos_var_z
    1e6, 1e6, 1e6,  # vel_var_x, vel_var_y, vel_var_z
    2.5e4, 2.5e4, 2.5e4  # acc_var_x, acc_var_y, acc_var_z
]

bounds = [
    (0.1, 100),  # q_jerk_x
    (0.1, 100),  # q_jerk_y
    (0.1, 100),  # q_jerk_z
    (1, 50),     # measurement_noise_std
    (1e1, 1e6),  # pos_var_x
    (1e1, 1e6),  # pos_var_y
    (1e1, 1e6),  # pos_var_z
    (1e1, 1e8),  # vel_var_x
    (1e1, 1e8),  # vel_var_y
    (1e1, 1e8),  # vel_var_z
    (1e1, 1e6),  # acc_var_x
    (1e1, 1e6),  # acc_var_y
    (1e1, 1e6)   # acc_var_z
]


strategies = ['rand1bin']
results = []

for strategy in strategies:
    print(f"Testing strategy: {strategy}")
    result = differential_evolution(
        lambda params: loss_function(params, ekf, time_ekf, interpolated_data),
        bounds=bounds,
        strategy=strategy,
        maxiter=500,
        disp=True
    )
    results.append((strategy, result.fun, result.x))

# Print all results after the loop
print("\nSummary of Results:")
for strategy, best_loss, best_params in results:
    print(f"Strategy: {strategy} -> Best Loss: {best_loss}, Parameters: {best_params}")
