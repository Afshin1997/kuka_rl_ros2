# import pyrealsense2 as rs

# def list_supported_streams():
#     # Create a context
#     context = rs.context()
    
#     # Get all connected devices
#     devices = context.query_devices()
#     if len(devices) == 0:
#         print("No RealSense devices connected.")
#         return
    
#     # Iterate through devices
#     for device in devices:
#         print(f"Device: {device.get_info(rs.camera_info.name)}")
#         sensors = device.query_sensors()
#         for sensor in sensors:
#             print(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
#             profiles = sensor.get_stream_profiles()
#             for profile in profiles:
#                 stream = profile.stream_type()
#                 format = profile.format()
#                 fps = profile.fps
#                 width = profile.as_video_stream_profile().width
#                 height = profile.as_video_stream_profile().height
#                 print(f"    Stream: {stream}, Format: {format}, Resolution: {width}x{height}, FPS: {fps}")

# if __name__ == "__main__":
#     list_supported_streams()


import pyrealsense2 as rs

def list_supported_options():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()

    print("Supported options for the depth sensor:")
    for option in depth_sensor.get_supported_options():
        print(f"- {rs.option(option)}")

    pipeline.stop()

if __name__ == "__main__":
    list_supported_options()
