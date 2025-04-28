import cv2
import numpy as np
import glob
import os

# Define the chessboard size (number of inner corners per row and column)
chessboard_size = (9, 6)

# Prepare object points in 2D (x, y), with z=0
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store 3D points (objpoints) and 2D image points (imgpoints)
objpoints = []
imgpoints = []

# Define criteria for sub-pixel corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# Load all images from a folder (adjust the path/pattern to your dataset)
images = glob.glob('calibration_images/*.jpg')

# Create an output directory to store images with drawn corners
output_dir = 'calibration_images_with_corners'
os.makedirs(output_dir, exist_ok=True)

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing: CLAHE for contrast enhancement
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    # gray = clahe.apply(gray)

    # Find the chessboard corners (rough detection)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Refine corner locations to sub-pixel accuracy
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1), criteria
        )

        # Add object points and image points (refined) to the lists
        objpoints.append(objp)
        imgpoints.append(corners_subpix)

        # Draw and display the refined corners on the image
        cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)

        # Save the image with corners drawn
        base_filename = os.path.splitext(os.path.basename(fname))[0]
        output_fname = os.path.join(output_dir, f"{base_filename}_corners.jpg")
        cv2.imwrite(output_fname, img)

        print(f"Chessboard corners found and saved for: {fname}")
    else:
        # Save the image with corners drawn
        # base_filename = os.path.splitext(os.path.basename(fname))[0]
        # output_fname = os.path.join(output_dir, f"{base_filename}_grayscale.jpg")
        # cv2.imwrite(output_fname, gray)
        print(f"Could not find corners in {fname}")

# Calibrate the camera using all collected points
if len(objpoints) > 0 and len(imgpoints) > 0:
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K3  # or other flags if needed

    # Then call calibrateCamera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags
    )
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    #     objpoints, imgpoints, gray.shape[::-1], None, None
    # )

    # Save the calibration results
    np.save('../camera_params/camera_matrix.npy', camera_matrix)
    np.save('../camera_params/dist_coeffs.npy', dist_coeffs)

    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
else:
    print("No valid chessboard detections found. Calibration aborted.")
