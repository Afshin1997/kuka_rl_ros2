import os
import time
import csv
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import threading
from threading import Lock
from queue import Queue, Empty

# ---------------------------
# EKF Class
# ---------------------------
class EKF:
    def __init__(self, device='cpu'):
        import torch
        self.torch = torch
        self.device = torch.device(device)

        # Initialize state [x, y, z, vx, vy, vz, ax, ay, az]
        initial_position = [3600.0, -300.0, 700.0]
        initial_velocity = [-3600.0, 0.0, 3000.0]
        initial_acc = [0.0, 0.0, 0.0]
        initial_state = initial_position + initial_velocity + initial_acc
        self.x = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Covariance
        pos_var_x = 1e4
        pos_var_y = 1e4
        pos_var_z = 1e4

        vel_var_x = 1e6   # => std dev of 1000 mm/s = 1 m/s
        vel_var_y = 1e6
        vel_var_z = 1e6

        acc_var_x = 1e6   # => std dev of 1000 mm/s^2
        acc_var_y = 1e6
        acc_var_z = 1e6

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
        measurement_noise_std = 10  # 2 cm
        self.R = torch.eye(3, dtype=torch.float32) * (measurement_noise_std**2)

        # Measurement matrix
        self.H = torch.zeros((3, 9), dtype=torch.float32, device=self.device)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z

        # Process noise parameters (jerk)
        self.q_jerk_x = 5
        self.q_jerk_y = 5
        self.q_jerk_z = 5

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

        return self.x.squeeze().cpu().detach().numpy()

    def get_state(self):
        return self.x.squeeze().cpu().detach().numpy()

# ---------------------------
# Depth and Color Camera Class
# ---------------------------
# class DepthColorCamera:
#     def __init__(self):
#         self.pipeline = rs.pipeline()
#         config = rs.config()

#         # Enable depth and color streams at desired resolutions and frame rates
#         config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)    # Depth at 60 FPS
#         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)   # Color at 60 FPS

#         try:
#             self.pipeline.start(config)
#         except RuntimeError as e:
#             print("Failed to start pipeline:", e)
#             print("Please check supported stream configurations in RealSense Viewer.")
#             exit(1)

#         self.align = rs.align(rs.stream.color)

#         # RealSense filters
#         self.depth_to_disparity = rs.disparity_transform(True)
#         self.disparity_to_depth = rs.disparity_transform(False)

#         self.spatial_filter = rs.spatial_filter()
#         self.temporal_filter = rs.temporal_filter()
#         self.hole_filling = rs.hole_filling_filter()

#         # Access the device and sensors to reset exposure settings
#         profile = self.pipeline.get_active_profile()
#         depth_sensor = profile.get_device().first_depth_sensor()
#         color_sensor = profile.get_device().first_color_sensor()

#         print("Auto-exposure re-enabled for depth and color sensors.")

#     def get_frames(self):
#         frames = self.pipeline.wait_for_frames()
#         aligned_frames = self.align.process(frames)

#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         # Apply filters to depth frame
#         if depth_frame:
#             depth_frame = self.depth_to_disparity.process(depth_frame)
#             depth_frame = self.spatial_filter.process(depth_frame)
#             depth_frame = self.temporal_filter.process(depth_frame)
#             depth_frame = self.disparity_to_depth.process(depth_frame)
#             depth_frame = self.hole_filling.process(depth_frame)

#         depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
#         color_image = np.asanyarray(color_frame.get_data()) if color_frame else None

#         return depth_image, color_image

#     def release(self):
#         self.pipeline.stop()

class DepthColorCamera:
    def __init__(self, ir_emitter_enabled=True, ir_emitter_always_on=True, ir_emitter_power=150, auto_exposure_enabled=False):
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Enable depth and color streams at desired resolutions and frame rates
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)    # Depth at 60 FPS
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)   # Color at 60 FPS

        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            print("Failed to start pipeline:", e)
            print("Please check supported stream configurations in RealSense Viewer.")
            exit(1)

        self.align = rs.align(rs.stream.color)

        # RealSense filters
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # Access the device and sensors to adjust settings
        profile = self.pipeline.get_active_profile()
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        color_sensor = device.first_color_sensor()

        # Configure IR Emitter settings
        self.configure_ir_emitter(depth_sensor, ir_emitter_enabled, ir_emitter_always_on, ir_emitter_power)

        # Configure Auto Exposure settings
        self.configure_auto_exposure(color_sensor, auto_exposure_enabled)

        print("Camera initialized with IR emitter settings applied.")

    def configure_ir_emitter(self, depth_sensor, enabled, always_on, power_level):
        # Enable or disable the IR emitter
        try:
            depth_sensor.set_option(rs.option.emitter_enabled, 1 if enabled else 0)
            status = depth_sensor.get_option(rs.option.emitter_enabled)
            print(f"IR Emitter Enabled: {bool(status)}")
        except RuntimeError as e:
            print(f"Failed to set IR emitter enabled state: {e}")

        # Set the IR emitter to always on
        try:
            depth_sensor.set_option(rs.option.emitter_always_on, 1 if always_on else 0)
            status = depth_sensor.get_option(rs.option.emitter_always_on)
            print(f"IR Emitter Always On: {bool(status)}")
        except RuntimeError as e:
            print(f"Failed to set IR emitter always on: {e}")

        # Adjust the laser power
        try:
            depth_sensor.set_option(rs.option.laser_power, power_level)
            current_power = depth_sensor.get_option(rs.option.laser_power)
            print(f"IR Emitter Laser Power set to: {current_power}")
        except RuntimeError as e:
            print(f"Failed to set IR emitter laser power: {e}")

    def configure_auto_exposure(self, color_sensor, enabled):
        try:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1 if enabled else 0)
            status = color_sensor.get_option(rs.option.enable_auto_exposure)
            print(f"Auto Exposure Enabled: {bool(status)}")
        except RuntimeError as e:
            print(f"Failed to set auto exposure: {e}")

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Apply filters to depth frame
        if depth_frame:
            depth_frame = self.depth_to_disparity.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        color_image = np.asanyarray(color_frame.get_data()) if color_frame else None

        return depth_image, color_image

    def release(self):
        self.pipeline.stop()


# ---------------------------
# CSV Recording Function
# ---------------------------
def create_new_csv_file(counter, directory, filename_suffix):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f'ball_tracking_data_{counter}_{filename_suffix}.csv')
    file = open(filename, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z', 'vx', 'vy', 'vz', 'e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz', 'x_world', 'y_world', 'z_world', 'dt'])
    return file, writer

# ---------------------------
# EKF Prediction Thread
# ---------------------------
def ekf_prediction_loop(
    ekf,
    stop_event,
    prediction_writer,    # Only used after the loop ends
    prediction_interval,  # e.g. 1/250 = 0.004
    ekf_lock,
    measurement_queue
):
    FIXED_DT = 0.004  # 250 Hz
    last_measurement = None
    last_prediction_time = time.perf_counter()
    data_buffer = []  # List to store all data rows

    while not stop_event.is_set():
        elapsed = time.perf_counter() - last_prediction_time

        # 1) Check if it's time to do the next predict/update cycle
        if elapsed >= FIXED_DT and last_measurement is not None:
            # with ekf_lock:
            # Predict with fixed dt
            ekf.predict(FIXED_DT)

            state = ekf.update(last_measurement)

            # Make sure last_measurement is not None before using it in CSV
            mx, my, mz = (last_measurement if last_measurement is not None
                            else (0.0, 0.0, 0.0))

            # Append the data row to the buffer
            data_buffer.append([
                "", "", "",
                "", "", "",
                f"{state[0]:.4f}",
                f"{state[1]:.4f}",
                f"{state[2]:.4f}",
                f"{state[3]:.4f}",
                f"{state[4]:.4f}",
                f"{state[5]:.4f}",
                f"{mx:.4f}",
                f"{my:.4f}",
                f"{mz:.4f}",
                f"{time.perf_counter() - last_prediction_time:.4f}",
            ])

            # Advance the "last_prediction_time" by exactly one cycle
            last_prediction_time = time.perf_counter()

        # 2) Grab any new measurements that arrived
        try:
            # Drain the queue, keeping only the most recent measurement
            while True:
                last_measurement = measurement_queue.get_nowait()
                measurement_queue.task_done()
        except Empty:
            pass

    prediction_writer.writerows(data_buffer)



# ---------------------------
# Helper Function: Process Image for Ball Tracking
# ---------------------------
# def process_image_for_ball_tracking(color_image, depth_image, camera_matrix, dist_coeffs):
#     # Undistort the image
#     undistorted_image = cv2.undistort(color_image, camera_matrix, dist_coeffs)
    
#     # Convert to HSV color space
#     hsv = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)
    
#     # Improve lighting conditions using histogram equalization on the V channel
#     h, s, v = cv2.split(hsv)
#     v_eq = cv2.equalizeHist(v)
#     hsv_eq = cv2.merge([h, s, v_eq])

#     # # Define HSV color range for the yellow ball
#     lower_pink = np.array([20, 100, 100])
#     upper_pink = np.array([35, 255, 255])


#     # Define HSV color range for the pink ball
#     # lower_pink = np.array([150, 80, 150])
#     # upper_pink = np.array([180, 255, 255])

#     # Create a mask using the adjusted HSV image
#     mask = cv2.inRange(hsv_eq, lower_pink, upper_pink)
    
#     # Apply morphological operations to clean up the mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

#     mask_smooth = cv2.GaussianBlur(mask, (3, 3), 0)

#     # Find contours in the cleaned mask
#     contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     x_3d, y_3d, z_3d = 0, 0, 0
#     ball_detected = False
    
#     if len(contours) > 0:
#         # Filter contours based on area and shape
#         possible_balls = []
#         for c in contours:
#             area = cv2.contourArea(c)
#             if area < 10:  # Minimum area threshold
#                 continue

#             # Calculate circularity
#             perimeter = cv2.arcLength(c, True)
#             if perimeter == 0:
#                 continue
#             circularity = 4 * np.pi * (area / (perimeter * perimeter))
#             if circularity < 0.5:  # Adjust threshold as needed
#                 continue

#             possible_balls.append(c)

#         if possible_balls:
#             # Choose the contour with the largest area
#             c = max(possible_balls, key=cv2.contourArea)
#             ((x_c, y_c), radius) = cv2.minEnclosingCircle(c)
#             if radius > 3:
#                 # Depth extraction with validation
#                 window_size = 3
#                 half_window = window_size // 2
#                 x_start = max(int(x_c) - half_window, 0)
#                 x_end = min(int(x_c) + half_window + 1, depth_image.shape[1])
#                 y_start = max(int(y_c) - half_window, 0)
#                 y_end = min(int(y_c) + half_window + 1, depth_image.shape[0])
#                 depth_window = depth_image[y_start:y_end, x_start:x_end]
#                 depth_window_masked = np.ma.masked_equal(depth_window, 0)
#                 if depth_window_masked.count() > 0:
#                     depth = depth_window_masked.mean()
#                     # 3D coordinate calculation
#                     x_3d = (x_c - camera_matrix[0, 2]) / camera_matrix[0, 0] * depth
#                     y_3d = (y_c - camera_matrix[1, 2]) / camera_matrix[1, 1] * depth
#                     z_3d = depth
#                     cv2.circle(undistorted_image, (int(x_c), int(y_c)), int(radius), (0, 255, 255), 2)
#                     ball_detected = True
#                 else:
#                     # No valid depth data
#                     ball_detected = False

#     return x_3d, y_3d, z_3d, ball_detected, undistorted_image


def process_image_for_ball_tracking(color_image, depth_image,
                                    camera_matrix, dist_coeffs,
                                    map1, map2):

    # Remap instead of undistort
    undistorted_image = cv2.remap(color_image, map1, map2, interpolation=cv2.INTER_LINEAR)

    
    # 2) Convert to HSV color space
    hsv = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)
    
    # 3) Improve lighting conditions (histogram equalization on the Value channel)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    # 4) Define HSV color range
    #    Adjust these ranges as necessary (example for yellow-like color).
    #    If your ball is pink, set them to the pink range.
    # lower_ball = np.array([20, 100, 100])
    # upper_ball = np.array([35, 255, 255])

    lower_ball = np.array([150, 80, 150])
    upper_ball = np.array([180, 255, 255])

    # 5) Create a mask based on the refined HSV image
    mask = cv2.inRange(hsv_eq, lower_ball, upper_ball)
    
    # 6) Morphological operations to reduce noise
    #    Increased kernel size + possible iterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 7) Smooth the mask before contour detection (larger blur kernel if needed)
    mask_smooth = cv2.GaussianBlur(mask_clean, (3, 3), 0)

    # 8) Find contours in the cleaned/smoothed mask
    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x_3d, y_3d, z_3d = 0, 0, 0
    ball_detected = False
    
    if len(contours) > 0:
        # Filter contours based on area and shape
        possible_balls = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 5:  # Minimum area threshold (may adjust downward if ball is very small/far)
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.7:  # Adjust threshold as needed
                continue

            possible_balls.append(c)

        if possible_balls:
            # Pick the largest contour (by area)
            c = max(possible_balls, key=cv2.contourArea)
            ((x_c, y_c), radius) = cv2.minEnclosingCircle(c)

            if radius > 5:  # Adjust as needed for small/far objects
                # Depth extraction with a larger window
                window_size = 3
                half_window = window_size // 2

                # Clip coordinates so we don't go out of image bounds
                x_start = max(int(x_c) - half_window, 0)
                x_end   = min(int(x_c) + half_window + 1, depth_image.shape[1])
                y_start = max(int(y_c) - half_window, 0)
                y_end   = min(int(y_c) + half_window + 1, depth_image.shape[0])

                depth_window = depth_image[y_start:y_end, x_start:x_end]

                # Exclude invalid depth values (e.g., 0 => no depth)
                valid_depths = depth_window[depth_window > 0]

                if len(valid_depths) > 0:
                    # Use median or mean. Median is more robust to outliers.
                    # print("valid_depths", valid_depths)
                    depth = np.median(valid_depths)
                    # print("depth", depth)

                    # 3D coordinate calculation
                    x_3d = (x_c - camera_matrix[0, 2]) / camera_matrix[0, 0] * depth
                    y_3d = (y_c - camera_matrix[1, 2]) / camera_matrix[1, 1] * depth
                    z_3d = depth

                    cv2.circle(undistorted_image, (int(x_c), int(y_c)), int(radius), (0, 255, 255), 2)
                    ball_detected = True
                else:
                    ball_detected = False

    return x_3d, y_3d, z_3d, ball_detected, undistorted_image


# ---------------------------
# Convert Coordinates to World Frame
# ---------------------------
def convert_to_world_coordinates(x_camera, y_camera, z_camera, R_camera_to_world, T_camera_to_world):
    camera_coords = np.array([[x_camera], [y_camera], [z_camera]])
    world_coords = R_camera_to_world @ camera_coords + T_camera_to_world.reshape(3, 1)
    return world_coords.flatten()

# ---------------------------
# Main Script
# ---------------------------
def main():
    FIXED_DT = 1.0 / 250.0  # 250 Hz for EKF
    MEASUREMENT_FREQ = 30.0  # 30 Hz from camera
    EKF_PREDICTION_FREQ = 250.0  # 250 Hz for EKF predictions
    prediction_interval = 1.0 / EKF_PREDICTION_FREQ

    recording = False
    recording_counter = 0
    directory = 'recorded_data'

    dc = DepthColorCamera()

    camera_matrix = np.load('../camera_params/camera_matrix.npy')
    dist_coeffs = np.load('../camera_params/dist_coeffs.npy')
    R_camera_to_world = np.load('../camera_params/camera_pose_base.npz')['R']
    T_camera_to_world = np.load('../camera_params/camera_pose_base.npz')['tvec']

    #############
    width, height = 640, 480
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs,
        None, 
        newCameraMatrix,
        (width, height),
        cv2.CV_16SC2
    )


    ######
    ekf = None
    ekf_lock = Lock()
    stop_event = threading.Event()
    ekf_thread = None

    frames_without_ball = 0
    max_frames_without_ball = 3  # Stop recording when the ball is not detected

    # Initialize variables to None
    tracking_file = None
    prediction_file = None
    video_writer = None
    prediction_writer = None
    tracking_buffer = []

    # Initialize the measurement queue
    measurement_queue = Queue()

    try:
        while True:
            last_frame_timer = time.perf_counter()
            depth_image, color_image = dc.get_frames()

            if color_image is not None and depth_image is not None:
                # Process the image to detect the ball
                x_3d, y_3d, z_3d, ball_detected, undistorted_image = process_image_for_ball_tracking(
                    color_image,
                    depth_image,
                    camera_matrix,     # used only for the depth -> real world calc
                    dist_coeffs,       # or skip if not needed for anything else
                    map1, map2         # newly added parameters
                )

                if ball_detected:
                    # Convert to world coordinates
                    # start_time_tracking = time.perf_counter()
                    x_world, y_world, z_world = convert_to_world_coordinates(
                        x_3d, y_3d, z_3d, R_camera_to_world, T_camera_to_world
                    )
                    print(f"World Coordinates: X={x_world:.2f}, Y={y_world:.2f}, Z={z_world:.2f}")

                    if np.abs(x_world) < 3600:
                        # Ball is within range
                        if not recording:
                            recording = True
                            recording_counter += 1
                            tracking_file, tracking_writer = create_new_csv_file(
                                recording_counter, directory, "tracking"
                            )
                            print(f"Recording {recording_counter} started...")
                            video_filename = os.path.join(directory, f'ball_tracking_video_{recording_counter}.avi')
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            fps = 30
                            frame_size = (undistorted_image.shape[1], undistorted_image.shape[0])
                            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

                            # Initialize EKF
                            ekf = EKF()

                            # Start ekf_thread
                            if ekf_thread is None:
                                prediction_file, prediction_writer = create_new_csv_file(
                                    recording_counter, directory, "predictions"
                                )
                                ekf_thread = threading.Thread(
                                    target=ekf_prediction_loop,
                                    args=(
                                        ekf,
                                        stop_event,
                                        prediction_writer,
                                        prediction_interval,
                                        ekf_lock,
                                        measurement_queue,
                                    )
                                )
                                ekf_thread.start()

                        # Prepare the measurement
                        z_state = torch.tensor([x_world, y_world, z_world], dtype=torch.float32)

                        # Enqueue the measurement for the EKF thread to process
                        measurement_queue.put(z_state)

                        

                        frames_without_ball = 0  # Reset counter when ball is detected within range

                        if recording and video_writer is not None:
                            video_writer.write(undistorted_image)

                        # tracking_period = time.perf_counter() - last_frame_timer
                        # Record tracking data (without EKF state since it's handled in the EKF thread)
                        # tracking_writer.writerow([
                        #     f"{x_3d:.4f}", f"{y_3d:.4f}", f"{z_3d:.4f}",
                        #     "", "", "", # 3d velocities
                        #     "", "", "",  # Velocities not measured directly
                        #     "", "", "",  # EKF state will be recorded by the EKF thread
                        #     f"{x_world:.4f}", f"{y_world:.4f}", f"{z_world:.4f}",
                        #     f"{tracking_period:.4f}"
                        # ])

                    else:
                        # Ball is beyond 5 meters, treat as not detected
                        frames_without_ball += 1

                else:
                    # Ball is not detected
                    frames_without_ball += 1

                # Check if we should stop recording
                if frames_without_ball > max_frames_without_ball:
                    if recording:
                        recording = False
                        print("Recording stopped due to ball loss.")
                        frames_without_ball = 0
                        if tracking_file is not None:
                            tracking_file.close()
                            print(f"Tracking recording {recording_counter} stopped.")
                            tracking_file = None
                        if video_writer is not None:
                            video_writer.release()
                            print(f"Video recording {recording_counter} stopped.")
                            video_writer = None
                        if ekf_thread is not None:
                            stop_event.set()
                            ekf_thread.join()
                            ekf_thread = None
                            stop_event.clear()
                        if prediction_file is not None:
                            prediction_file.close()
                            print(f"Prediction recording {recording_counter} stopped.")
                            prediction_file = None
                            prediction_writer = None
                        ekf = None

                # Display status text on the image
                # status_text = f"Recording: {'Yes' if recording else 'No'}"
                # cv2.putText(undistorted_image, status_text, (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                # Show the image
                cv2.imshow('Color Image', undistorted_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")

    finally:
        if tracking_file is not None:
            tracking_file.close()
            print(f"Tracking recording {recording_counter} stopped.")
        if prediction_file is not None:
            prediction_file.close()
            print(f"Prediction recording {recording_counter} stopped.")
        if video_writer is not None:
            video_writer.release()
            print(f"Video recording {recording_counter} stopped.")
        if ekf_thread is not None:
            stop_event.set()
            ekf_thread.join()
        dc.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
