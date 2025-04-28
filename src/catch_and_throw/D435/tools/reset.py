import pyrealsense2 as rs

def reset_camera_settings():
    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Start streaming
    pipeline_profile = pipeline.start(config)

    # Access the device and depth sensor
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    color_sensor = device.first_color_sensor()

    # Define a dictionary of options to reset with their default values
    default_options = {
        rs.option.emitter_enabled: 1,           # Typically enabled by default
        rs.option.laser_power: 150,             # Default laser power
        rs.option.enable_auto_exposure: 1,      # Auto exposure enabled
        rs.option.exposure: 156,                # Default exposure value (may vary)
        rs.option.gain: 16,                      # Default gain value (may vary)
        rs.option.visual_preset: 0,              # Default visual preset (Default)
        # Add more options as needed
    }

    # Reset each option
    for option, default_value in default_options.items():
        try:
            depth_sensor.set_option(option, default_value)
            current_value = depth_sensor.get_option(option)
            print(f"Reset {rs.option(option)} to {current_value}")
        except RuntimeError as e:
            print(f"Failed to reset {rs.option(option)}: {e}")

    # Similarly reset color sensor options if needed
    color_default_options = {
        rs.option.enable_auto_exposure: 1,
        rs.option.exposure: 156,    # Example default value
        rs.option.gain: 16,         # Example default value
        # Add more color sensor options as needed
    }

    for option, default_value in color_default_options.items():
        try:
            color_sensor.set_option(option, default_value)
            current_value = color_sensor.get_option(option)
            print(f"Reset Color {rs.option(option)} to {current_value}")
        except RuntimeError as e:
            print(f"Failed to reset Color {rs.option(option)}: {e}")

    # Stop the pipeline
    pipeline.stop()

if __name__ == "__main__":
    reset_camera_settings()
