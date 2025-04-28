import cv2
import numpy as np
import pyrealsense2 as rs

# Load camera matrix and distortion coefficients from previous calibration
camera_matrix = np.load('../camera_params/camera_matrix.npy')
dist_coeffs = np.load('../camera_params/dist_coeffs.npy')

# Define the 3D points for the chessboard corners
chessboard_size = (9,6)  # Number of inner corners per row and column
square_size = 49.5  # Square size in millimeters # 48 WAS WORKING
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)

# Center the chessboard by shifting points
objp[:, 0] -= (chessboard_size[0] - 1) / 2
objp[:, 1] -= (chessboard_size[1] - 1) / 2
objp *= square_size  # Scale to real-world size

# Define the axis points for drawing the 3D frame (origin, x-axis, y-axis, z-axis)
axis = np.float32([[0, 0, 0],
                   [150, 0, 0],
                   [0, 150, 0],
                   [0, 0, 150]])  # Z positive for right-hand

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from RealSense camera
pipeline.start(config)

try:
    while True:
        # Wait for a new frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the frame to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        gray_clahe = clahe.apply(gray)

        # Apply image preprocessing to enhance detection when the chessboard is not tilted
        # Apply Gaussian blur to reduce noise
        gray_blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

        # Apply adaptive thresholding to enhance edges
        gray_thresh = cv2.adaptiveThreshold(
            gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # kernel = np.ones((3, 3), np.uint8)
        # gray_thresh = cv2.morphologyEx(gray_thresh, cv2.MORPH_CLOSE, kernel)

        # Find the chessboard corners in the preprocessed image
        # Use flags to improve detection when the chessboard is not tilted
        ret, corners = cv2.findChessboardCornersSB(
            gray_thresh, chessboard_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

        # ret, corners = cv2.findChessboardCorners(
        #     gray_thresh, chessboard_size,
        #     flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        # )

        if ret:
            # Refine corner positions for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.0001)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria
            )

            # SolvePnP to get rvec and tvec for the current frame
            ret, rvec, tvec = cv2.solvePnP(
                objp, corners_refined, camera_matrix, dist_coeffs
            )
            print('t_vec',tvec)
            print('t_vec_norm',np.linalg.norm(tvec))
            
            # Convert rvec to rotation matrix
            Rot, _ = cv2.Rodrigues(rvec)
            
            # Define the rotation matrix for 180 degrees around the X-axis
            R_x_180 = np.array([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]])

            R_z_90 = np.array([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]])

            R_y_180 = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            # Apply the additional rotation (rotate the original rotation matrix around the X-axis)
            Rot = Rot @ R_x_180
            # Rot = Rot
            
            # Inverted rotation matrix
            Rot_inv = Rot.T 
            tvec_inv = -Rot_inv @ tvec  # Inverted translation vector

            print('Rot_inv', Rot_inv)
            print('tvec_inv',tvec_inv)
            print('tvec_iv_norm', np.linalg.norm(tvec_inv))

            # Project the 3D axis points onto the image plane
            imgpts, _ = cv2.projectPoints(
                axis, rvec, tvec, camera_matrix, dist_coeffs
            )

            # Draw the 3D frame axes on the frame using imgpts[0] as the origin
            origin = tuple(map(int, imgpts[0].ravel()))  # Center of chessboard
            color_image = cv2.line(
                color_image, origin, tuple(map(int, imgpts[1].ravel())),
                (255, 0, 0), 3
            )  # X-axis in blue
            color_image = cv2.line(
                color_image, origin, tuple(map(int, imgpts[2].ravel())),
                (0, 255, 0), 3
            )  # Y-axis in green
            color_image = cv2.line(
                color_image, origin, tuple(map(int, imgpts[3].ravel())),
                (0, 0, 255), 3
            )  # Z-axis in red

            # Save the inverted pose for debugging
            np.savez('../camera_params/camera_pose_inv.npz', R_inv=Rot_inv, tvec_inv=tvec_inv)
            cv2.imwrite('../camera_params/image.png', color_image)
        else:
            print("Chessboard not detected in this frame.")

        # Display the frame with the 3D frame overlay
        cv2.imshow('World Frame on RealSense Video', color_image)
        
        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()




