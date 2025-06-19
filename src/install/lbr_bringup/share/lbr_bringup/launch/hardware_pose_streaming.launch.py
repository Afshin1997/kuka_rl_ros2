from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to realsense launch file
    hardware_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('lbr_bringup'),
                'launch',
                'hardware.launch.py'
            )
        ),
        launch_arguments={
            'ctrl': 'lbr_joint_position_command_controller',
            'model': 'iiwa7'
        }.items()
    )

    # Pose streaming node
    pose_streaming_node = Node(
        package='lbr_demos_advanced_cpp',
        executable='pose_streaming',
        name='pose_streaming',
        remappings=[
            ('__ns', '/lbr'),
        ]
    )


    return LaunchDescription([
        hardware_launch,
        pose_streaming_node
    ])

