class KalmanFilter3D:
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

        # Covariance
        self.P = torch.zeros((9, 9), dtype=torch.float32, device=self.device)
        pos_var = 1e3
        vel_var = 1e4
        acc_var = 1e2
        self.P[0:3, 0:3] = torch.eye(3)*pos_var
        self.P[3:6, 3:6] = torch.eye(3)*vel_var
        self.P[6:9, 6:9] = torch.eye(3)*acc_var

        # Measurement noise
        self.R = torch.eye(3, dtype=torch.float32, device=self.device)*(10.0**2)

        # Measurement matrix
        self.H = torch.zeros((3, 9), dtype=torch.float32, device=self.device)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z

        # Process noise parameters (jerk)
        self.q_jerk_x = 0.2
        self.q_jerk_y = 0.2
        self.q_jerk_z = 0.2

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

# Usage:
# 1) Loop at 250 Hz => call kf.predict(dt=1/250) each cycle.
# 2) If a measurement arrives (e.g. every 1/40 s), do kf.update(z).

