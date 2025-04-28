import cv2
import numpy as np

# Camera intrinsic parameters (example values; replace with your camera's parameters)
CAMERA_INTRINSICS = {
    "fx": 600,  # Focal length in pixels (x-axis)
    "fy": 600,  # Focal length in pixels (y-axis)
    "cx": 320,  # Principal point x-coordinate
    "cy": 240   # Principal point y-coordinate
}

# Ball properties

BALL_COLOR_LOWER = np.array([150, 80, 150])
BALL_COLOR_UPPER = np.array([180, 255, 255])
# BALL_COLOR_LOWER = np.array([30, 100, 100])  # HSV lower bound for the ball color
# BALL_COLOR_UPPER = np.array([50, 255, 255])  # HSV upper bound for the ball color
BALL_DIAMETER = 0.1  # Real-world diameter of the ball in meters

def detect_ball_position(rgb_image, depth_image):
    """
    Detect the 3D position of a ball in space from an RGB and depth image,
    using the known ball diameter for depth refinement.

    Parameters:
    - rgb_image: The RGB image (numpy array).
    - depth_image: The depth image (numpy array, same resolution as RGB).

    Returns:
    - A tuple (x, y, z) representing the 3D coordinates of the ball in meters.
    """
    # Convert the RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Detect the ball using color segmentation
    mask = cv2.inRange(hsv_image, BALL_COLOR_LOWER, BALL_COLOR_UPPER)
    mask = cv2.medianBlur(mask, 5)  # Smooth the mask to reduce noise

    # Find contours of the detected ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No ball detected!")
        return None

    # Assume the largest contour is the ball
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the minimum enclosing circle of the ball
    ((x, y), radius_pixels) = cv2.minEnclosingCircle(largest_contour)

    # Validate the detected ball
    if radius_pixels < 5:  # Ignore small detections
        print("Ball radius too small!")
        return None

    # Compute the estimated depth from the ball's apparent size
    fx = CAMERA_INTRINSICS["fx"]
    estimated_depth = (BALL_DIAMETER * fx) / (2 * radius_pixels)

    # Get the depth value at the ball's center from the depth image
    depth_from_image = depth_image[int(y), int(x)] / 1000.0  # Convert mm to meters

    # Combine the two depth estimates (e.g., take the average, or prioritize one)
    if depth_from_image > 0:
        depth_value = (depth_from_image + estimated_depth) / 2
    else:
        depth_value = estimated_depth

    # Calculate the 3D coordinates of the ball
    cx, cy = CAMERA_INTRINSICS["cx"], CAMERA_INTRINSICS["cy"]
    X = (x - cx) * depth_value / fx
    Y = (y - cy) * depth_value / CAMERA_INTRINSICS["fy"]
    Z = depth_value

    return X, Y, Z

# Example usage
if __name__ == "__main__":
    # Load the RGB and depth images (replace with your image paths or streams)
    rgb_image = cv2.imread("rgb_image.png")
    depth_image = cv2.imread("depth_image.png", cv2.IMREAD_UNCHANGED)

    # Detect the ball's 3D position
    ball_position = detect_ball_position(rgb_image, depth_image)
    if ball_position:
        print(f"Ball position in 3D space: X={ball_position[0]:.2f}m, Y={ball_position[1]:.2f}m, Z={ball_position[2]:.2f}m")

    # Visualize the result
    if ball_position:
        output_image = rgb_image.copy()
        cv2.circle(output_image, (int(ball_position[0]), int(ball_position[1])), 10, (0, 255, 0), 2)
        cv2.imshow("Ball Detection", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
