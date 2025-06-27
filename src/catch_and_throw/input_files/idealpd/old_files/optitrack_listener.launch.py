from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.substitutions import FindPackageShare
import time

def generate_launch_description():

    conf_file_path = 'optitrack.yaml'

    opt_conf = PathJoinSubstitution([
        get_package_share_directory('optitrack_listener'),	
        conf_file_path
    ])

    return LaunchDescription([
        Node(
            package='optitrack_listener',
            executable='optitrack_listener',
            name='optitrack_listener',
            output='screen',
            parameters=[
                opt_conf
            ]
        )
    ])