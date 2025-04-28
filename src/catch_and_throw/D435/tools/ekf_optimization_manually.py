import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from ekf_class import EKF  # Assuming EKF is defined in ekf_class.py


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
        pos_var_x = 6.7e5
        pos_var_y = 5.7e5
        pos_var_z = 3.45e5
        vel_var_x = 9.2e7
        vel_var_y = 2.16e7
        vel_var_z = 6.5e7
        acc_var_x = 1.3e4
        acc_var_y = 1.67e5
        acc_var_z = 1.5e5

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
        measurement_noise_std = 1  # 2 cm
        self.R = torch.eye(3, dtype=torch.float32) * (measurement_noise_std**2)

        # Measurement matrix
        self.H = torch.zeros((3, 9), dtype=torch.float32, device=self.device)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z

        # Process noise parameters (jerk)
        self.q_jerk_x = 99
        self.q_jerk_y = 80
        self.q_jerk_z = 99

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




# Load the CSV files
file_path = '../tracking/recorded_data/ball_tracking_data_1_tracking.csv'
data = pd.read_csv(file_path)
file_path_predictions = '../tracking/recorded_data/ball_tracking_data_1_predictions.csv'
predictions_data = pd.read_csv(file_path_predictions)

# Extract relevant columns
x_world = data['x_world'].to_numpy()
y_world = data['y_world'].to_numpy()
z_world = data['z_world'].to_numpy()
time = data['dt'].cumsum().to_numpy()

e_x = predictions_data['e_x'].to_numpy()
e_y = predictions_data['e_y'].to_numpy()
e_z = predictions_data['e_z'].to_numpy()
time_predictions = predictions_data['dt'].cumsum().to_numpy()

# Adjust the tracking data's time array to start from zero
adjusted_time = time - time[0]

# Interpolate tracking data to prediction timestamps
interpolated_x = np.interp(time_predictions, adjusted_time, x_world)
interpolated_y = np.interp(time_predictions, adjusted_time, y_world)
interpolated_z = np.interp(time_predictions, adjusted_time, z_world)

# Initialize EKF with manual parameters
ekf = EKF(device='cpu')

# Process the data with EKF
predictions = []
for i in range(len(time_predictions)):
    dt = 0.004  # Fixed timestep for EKF updates
    ekf.predict(dt)
    z = [interpolated_x[i], interpolated_y[i], interpolated_z[i]]  # Measurement input
    ekf.update(z)
    predictions.append(ekf.get_state()[:3])  # Extract positions only

predictions = np.array(predictions)

# Plot x_world, e_x, interpolated data, and EKF predictions
plt.figure(figsize=(8, 6))
plt.plot(adjusted_time, x_world, label='x_world', linewidth=2)
plt.plot(time_predictions, e_x, label='e_x (Error)', linestyle='--', linewidth=2)
plt.plot(time_predictions, interpolated_x, label='Interpolated x', linestyle='-.', linewidth=2)
plt.plot(time_predictions, predictions[:, 0], label='EKF Prediction x', linestyle=':', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('X_world / e_x / Interpolated x (mm)')
plt.title('X_world, e_x, Interpolated x, and EKF Prediction vs Time')
plt.grid(True)
plt.legend()
plt.show()

## Repeat the same for y_world
plt.figure(figsize=(8, 6))
plt.plot(adjusted_time, y_world, label='y_world', linewidth=2)
plt.plot(time_predictions, e_y, label='e_y (Error)', linestyle='--', linewidth=2)
plt.plot(time_predictions, interpolated_y, label='Interpolated y', linestyle='-.', linewidth=2)
plt.plot(time_predictions, predictions[:, 1], label='EKF Prediction y', linestyle=':', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Y_world / e_y / Interpolated y (mm)')
plt.title('Y_world, e_y, Interpolated y, and EKF Prediction vs Time')
plt.grid(True)
plt.legend()
plt.show()

# Repeat the same for z_world
plt.figure(figsize=(8, 6))
plt.plot(adjusted_time, z_world, label='z_world', linewidth=2)
plt.plot(time_predictions, e_z, label='e_z (Error)', linestyle='--', linewidth=2)
plt.plot(time_predictions, interpolated_z, label='Interpolated z', linestyle='-.', linewidth=2)
plt.plot(time_predictions, predictions[:, 2], label='EKF Prediction z', linestyle=':', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Z_world / e_z / Interpolated z (mm)')
plt.title('Z_world, e_z, Interpolated z, and EKF Prediction vs Time')
plt.grid(True)
plt.legend()
plt.show()