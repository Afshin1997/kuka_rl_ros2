import os
import time
import csv
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
import threading
import numpy.ma as ma

# ---------------------------
# Extended Kalman Filter Class
# ---------------------------
class EKFWithKnownGravity:
    def __init__(self, dt):
        """
        EKF with known gravity components in the camera coordinate system.

        :param dt: Time step (seconds).
        :param initial_state: Initial state vector [X, Y, Z, Vx, Vy, Vz].
        :param process_noise_std: Standard deviation of process noise for position and velocity.
                                  Should be a tuple or list: (pos_noise_std, vel_noise_std).
        :param measurement_noise_std: Standard deviation of measurement noise in position.
        :param gravity_camera_frame: Gravity vector components in the camera coordinate system [g_x, g_y, g_z].
        :param device: PyTorch device ('cpu' or 'cuda').
        """

        g_world = np.array([0, 0, -9800])

        initial_position = [4000, -200, 500]
        initial_velocity = [-6000, -2000, 2000]
        initial_state = np.concatenate((initial_position, initial_velocity))

        process_noise_std = (20.0, 10.0)
        measurement_noise_std = (20.0, 10.0)

        self.device = torch.device('cpu')
        self.dt = dt

        # Gravity vector in camera frame
        self.g = torch.tensor(g_world, dtype=torch.float32, device=self.device).unsqueeze(1)

        # State vector: [X, Y, Z, Vx, Vy, Vz]
        self.x = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(1)

        # State transition matrix (F)
        self.F = torch.eye(6, device=self.device)
        self.F[0, 3] = dt  # dX/dVx
        self.F[1, 4] = dt  # dY/dVy
        self.F[2, 5] = dt  # dZ/dVz

        # Control-input matrix (B) for gravity
        self.B = torch.zeros((6, 3), dtype=torch.float32, device=self.device)
        self.B[:3, :] = 0.5 * dt**2 * torch.eye(3, dtype=torch.float32, device=self.device)
        self.B[3:, :] = dt * torch.eye(3, dtype=torch.float32, device=self.device)

        # Process noise covariance (Q)
        pos_noise_std_q, vel_noise_std_q = process_noise_std
        q_pos = pos_noise_std_q ** 2
        q_vel = vel_noise_std_q ** 2

        self.Q = torch.zeros((6, 6), dtype=torch.float32, device=self.device)
        self.Q[0:3, 0:3] = torch.eye(3, device=self.device) * q_pos
        self.Q[3:6, 3:6] = torch.eye(3, device=self.device) * q_vel

        # Measurement matrix (H)
        self.H = torch.eye(6, device=self.device)

        # Measurement noise covariance (R)
        pos_noise_std_r, vel_noise_std_r = measurement_noise_std  # if measurement noise std is given for both
        r_pos = pos_noise_std_r ** 2
        r_vel = vel_noise_std_r ** 2
        self.R = torch.zeros((6, 6), dtype=torch.float32, device=self.device)
        self.R[0:3, 0:3] = torch.eye(3, device=self.device) * r_pos
        self.R[3:6, 3:6] = torch.eye(3, device=self.device) * r_vel

        # State covariance matrix (P)
        self.P = torch.eye(6, dtype=torch.float32, device=self.device) * 1.0  # Initial uncertainty

    def predict(self):
        self.x = self.F.mm(self.x) + self.B.mm(self.g)
        self.P = self.F.mm(self.P).mm(self.F.t()) + self.Q
        return self.x

    def update(self, z):
        z = z.to(self.device).unsqueeze(1)
        y = z - self.H.mm(self.x)  # Measurement residual
        S = self.H.mm(self.P).mm(self.H.t()) + self.R  # Innovation covariance
        K = self.P.mm(self.H.t()).mm(torch.inverse(S))  # Kalman gain
        self.x = self.x + K.mm(y)
        I = torch.eye(6, device=self.device)
        self.P = (I - K.mm(self.H)).mm(self.P)
        return self.x

    def get_state(self):
        return self.x.cpu().detach().numpy().flatten()


# ---------------------------
# Depth and Color Camera Class
# ---------------------------
class DepthColorCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # RealSense filters
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

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
    writer.writerow(['x', 'y', 'z', 'vx', 'vy', 'vz', 'e_x', 'e_y', 'e_z', 'e_vx', 'e_vy', 'e_vz', 'x_world', 'y_world', 'z_world', 'e_gx', 'e_gy', 'e_gz', 'dt'])
    return file, writer


# ---------------------------
# EKF Prediction Thread
# ---------------------------
def ekf_prediction_loop(ekf, stop_event, prediction_writer, prediction_interval=1.0 / 120.0, lock=None):
    while not stop_event.is_set():
        if ekf is not None:
            ekf.predict()  # Predict step
            state = ekf.get_state()
            g_vector = ekf.g.cpu().numpy().flatten()
            prediction_writer.writerow([
                "", "", "",
                "", "", "",
                f"{state[0]:.4f}", f"{state[1]:.4f}", f"{state[2]:.4f}",
                f"{state[3]:.4f}", f"{state[4]:.4f}", f"{state[5]:.4f}",
                "", "", "",
                f"{g_vector[0]:.4f}", f"{g_vector[1]:.4f}", f"{g_vector[2]:.4f}",
                f"{prediction_interval:.4f}"
            ])
        time.sleep(prediction_interval)


# ---------------------------
# Helper Function: Process Image for Ball Tracking
# ---------------------------
import cv2
import numpy as np

def process_image_for_ball_tracking(color_image, depth_image, camera_matrix, dist_coeffs):
    # Undistort the image
    undistorted_image = cv2.undistort(color_image, camera_matrix, dist_coeffs)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)
    
    # Improve lighting conditions using histogram equalization on the V channel
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    # # Define HSV color range for the yellow ball
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([35, 255, 255])

    # Define HSV color range for the pink ball
    lower_yellow = np.array([150, 80, 150])
    upper_yellow = np.array([180, 255, 255])

    # Create a mask using the adjusted HSV image
    mask = cv2.inRange(hsv_eq, lower_yellow, upper_yellow)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    mask_smooth = cv2.GaussianBlur(mask_clean, (3, 3), 0)
    
    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x_3d, y_3d, z_3d = 0, 0, 0
    ball_detected = False
    
    if len(contours) > 0:
        # Filter contours based on area and shape
        possible_balls = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 10:  # Minimum area threshold
                continue

            # Calculate circularity
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.7:  # Adjust threshold as needed
                continue

            possible_balls.append(c)

        if possible_balls:
            # Choose the contour with the largest area
            c = max(possible_balls, key=cv2.contourArea)
            ((x_c, y_c), radius) = cv2.minEnclosingCircle(c)
            if radius > 3:
                # Depth extraction with validation
                window_size = 3
                half_window = window_size // 2
                x_start = max(int(x_c) - half_window, 0)
                x_end = min(int(x_c) + half_window + 1, depth_image.shape[1])
                y_start = max(int(y_c) - half_window, 0)
                y_end = min(int(y_c) + half_window + 1, depth_image.shape[0])
                depth_window = depth_image[y_start:y_end, x_start:x_end]
                depth_window_masked = np.ma.masked_equal(depth_window, 0)
                if depth_window_masked.count() > 0:
                    depth = depth_window_masked.mean()
                    # 3D coordinate calculation
                    x_3d = (x_c - camera_matrix[0, 2]) / camera_matrix[0, 0] * depth
                    y_3d = (y_c - camera_matrix[1, 2]) / camera_matrix[1, 1] * depth
                    z_3d = depth
                    cv2.circle(undistorted_image, (int(x_c), int(y_c)), int(radius), (0, 255, 255), 2)
                    ball_detected = True
                else:
                    # No valid depth data
                    ball_detected = False

    return x_3d, y_3d, z_3d, ball_detected, undistorted_image


# ---------------------------
# Convert Coordinates to World Frame
# ---------------------------
def convert_to_world_coordinates(x_camera, y_camera, z_camera, R_camera_to_world, T_camera_to_world):
    camera_coords = np.array([[x_camera], [y_camera], [z_camera]])
    world_coords = R_camera_to_world @ camera_coords + T_camera_to_world

    # print("camera_coords", camera_coords)
    # print("world_coords", world_coords)

    return world_coords.flatten()

# ---------------------------
# Main Script
# ---------------------------
def main():

    recording = False
    recording_counter = 0
    directory = 'recorded_data'

    dc = DepthColorCamera()
    camera_matrix = np.load('required_files/camera_matrix.npy')
    dist_coeffs = np.load('required_files/dist_coeffs.npy')
    R_camera_to_world = np.load('required_files/camera_pose_base.npz')['R_inv']
    T_camera_to_world = np.load('required_files/camera_pose_base.npz')['tvec_inv']

    prev_position = None
    prev_time = None
    ekf = None
    frames_without_ball = 0
    ekf_thread = None

    # Initialize variables to None
    file = None
    prediction_file = None
    video_writer = None
    stop_event = threading.Event()
    lock = threading.Lock()
    prediction_writer = None

    max_frames_without_ball = 5  # Stop recording immediately when the ball is not detected

    try:
        while True:
            depth_image, color_image = dc.get_frames()

            # ekf = EKFWithKnownGravity(dt=1.0 / 120.0)
            # lock = threading.Lock()

            if color_image is not None and depth_image is not None:
                # Process the image to detect the ball
                x_3d, y_3d, z_3d, ball_detected, undistorted_image = process_image_for_ball_tracking(
                    color_image, depth_image, camera_matrix, dist_coeffs
                )

                if ball_detected:
                    # Convert to world coordinates
                    x_world, y_world, z_world = convert_to_world_coordinates(
                        x_3d, y_3d, z_3d, R_camera_to_world, T_camera_to_world
                    )

                    if np.abs(x_world) < 4000:
                        # Ball is within range
                        current_time = time.time()
                        if prev_position is not None and prev_time is not None:
                            dt = current_time - prev_time
                            vx = (x_world - prev_position[0]) / dt
                            vy = (y_world - prev_position[1]) / dt
                            vz = (z_world - prev_position[2]) / dt

                            if not recording:
                                recording = True
                                recording_counter += 1
                                file, writer = create_new_csv_file(recording_counter, directory, "tracking")
                                print(f"Recording {recording_counter} started...")
                                video_filename = os.path.join(directory, f'ball_tracking_video_{recording_counter}.avi')
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                fps = 30
                                frame_size = (undistorted_image.shape[1], undistorted_image.shape[0])
                                video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

                                # Initialize EKF
                                ekf = EKFWithKnownGravity(dt=1.0 / 120.0)

                                # Start ekf_thread
                                if ekf_thread is None:
                                    stop_event.clear()
                                    prediction_file, prediction_writer = create_new_csv_file(recording_counter, directory, "predictions")
                                    ekf_thread = threading.Thread(target=ekf_prediction_loop,
                                                                  args=(ekf, stop_event, prediction_writer, 1.0 / 120.0, lock))
                                    ekf_thread.start()

                            # if ekf_thread is None:
                            #     stop_event = threading.Event()
                            #     prediction_file, prediction_writer = create_new_csv_file(recording_counter, directory, "predictions")
                            #     ekf_thread = threading.Thread(target=ekf_prediction_loop,
                            #                                   args=(ekf, stop_event, prediction_writer, 1.0 / 120.0, lock))
                            #     ekf_thread.start()

                            
                            z_state = torch.tensor([x_world, y_world, z_world, vx, vy, vz], dtype=torch.float32)
                            ekf.update(z_state)

                            state = ekf.get_state()
                            g_vector = ekf.g.cpu().numpy().flatten()
                            writer.writerow([
                                f"{x_3d:.4f}", f"{y_3d:.4f}", f"{z_3d:.4f}",
                                f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}",
                                f"{state[0]:.4f}", f"{state[1]:.4f}", f"{state[2]:.4f}",
                                f"{state[3]:.4f}", f"{state[4]:.4f}", f"{state[5]:.4f}",
                                f"{x_world:.4f}", f"{y_world:.4f}", f"{z_world:.4f}",
                                f"{g_vector[0]:.4f}", f"{g_vector[1]:.4f}", f"{g_vector[2]:.4f}",
                                f"{dt:.4f}"
                            ])

                            prev_position = (x_world, y_world, z_world)
                            prev_time = current_time

                        else:
                            # Initialize previous position and time
                            prev_position = (x_world, y_world, z_world)
                            prev_time = time.time()

                        frames_without_ball = 0  # Reset counter when ball is detected within range

                        if recording and video_writer is not None:
                            video_writer.write(undistorted_image)

                    else:
                        # Ball is beyond 3000 mm, treat as not detected
                        # ball_detected = False
                        frames_without_ball += 1

                else:
                    # Ball is not detected
                    frames_without_ball += 1

                # Check if we should stop recording
                if frames_without_ball > max_frames_without_ball:
                    # ekf = EKFWithKnownGravity(dt=1.0 / 120.0)
                    # lock = threading.Lock()
                    if recording:
                        recording = False
                        frames_without_ball = 0
                        if file is not None:
                            file.close()
                            print(f"Recording {recording_counter} stopped.")
                            file = None
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
                        if ekf_thread is not None:
                            stop_event.set()
                            ekf_thread.join()
                            ekf_thread = None
                            stop_event.clear()
                        if prediction_file is not None:
                            prediction_file.close()
                            prediction_file = None
                            print(f"Prediction recording {recording_counter} stopped.")
                            prediction_writer = None
                        prev_position = None
                        prev_time = None
                        ekf = None

                # Display status text on the image
                status_text = f"Recording: {'Yes' if recording else 'No'}"
                cv2.putText(undistorted_image, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

                # Show the image
                cv2.imshow('Color Image', undistorted_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")

    finally:
        if file is not None:
            file.close()
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
