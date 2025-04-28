import cv2
import numpy as np
import pyrealsense2 as rs

# Load camera matrix and distortion coefficients from previous calibration
camera_matrix = np.load('../camera_params/camera_matrix.npy')
dist_coeffs = np.load('../camera_params/dist_coeffs.npy')

# Define the 3D points for the chessboard corners
chessboard_size = (9,6)  # Number of inner corners per row and column
square_size = 48  # Square size in millimeters
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

# Initialize VideoWriter for recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
output_video_path = '../camera_params/output.avi'  # Output file path
frame_rate = 30  # Frame rate
frame_size = (640, 480)  # Frame size
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

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

        # Apply image preprocessing to enhance detection when the chessboard is not tilted
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_thresh = cv2.adaptiveThreshold(
            gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray_thresh)

        ret, corners = cv2.findChessboardCornersSB(
            gray_enhanced, chessboard_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        100, 0.0001)
            corners_refined = cv2.cornerSubPix(
                gray, corners, (3, 3), (-1, -1), criteria
            )

            ret, rvec, tvec = cv2.solvePnP(
                objp, corners_refined, camera_matrix, dist_coeffs
            )

            Rot, _ = cv2.Rodrigues(rvec)
            R_x_180 = np.array([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]])
            Rot = Rot @ R_x_180
            Rot_inv = Rot.T 
            tvec_inv = -Rot_inv @ tvec

            imgpts, _ = cv2.projectPoints(
                axis, rvec, tvec, camera_matrix, dist_coeffs
            )

            origin = tuple(map(int, imgpts[0].ravel()))  # Center of chessboard
            color_image = cv2.line(
                color_image, origin, tuple(map(int, imgpts[1].ravel())),
                (255, 0, 0), 3
            )
            color_image = cv2.line(
                color_image, origin, tuple(map(int, imgpts[2].ravel())),
                (0, 255, 0), 3
            )
            color_image = cv2.line(
                color_image, origin, tuple(map(int, imgpts[3].ravel())),
                (0, 0, 255), 3
            )

        else:
            print("Chessboard not detected in this frame.")

        # Write the frame to the video file
        video_writer.write(color_image)

        # Display the frame
        cv2.imshow('World Frame on RealSense Video', color_image)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and release resources
    pipeline.stop()
    video_writer.release()
    cv2.destroyAllWindows()
