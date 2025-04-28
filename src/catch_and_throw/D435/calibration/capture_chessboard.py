import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Initialize the pipeline
pipe = rs.pipeline()
cfg = rs.config()

# Enable the streams
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipe.start(cfg)

# Directory to save the captured images
output_dir = 'calibration_images'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# Set the interval for capturing images (in seconds)
capture_interval = 5
last_capture_time = 0
image_counter = 110

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert the image to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Show the camera stream
        cv2.imshow('Camera Stream', color_image)

        # Check if it's time to capture an image
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            # Save the image
            image_path = os.path.join(output_dir, f'calibration_image_{image_counter:03d}.jpg')
            cv2.imwrite(image_path, color_image)
            print(f"Captured image: {image_path}")
            image_counter += 1
            last_capture_time = current_time

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipe.stop()
    cv2.destroyAllWindows()
