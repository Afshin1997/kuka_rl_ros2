from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    package_dir = get_package_share_directory('catch_and_throw')
    
    # Path to your YAML config file
    config_file_1 = PathJoinSubstitution([
        package_dir,
        'config',
        'joint_state_node_config.yaml'
    ])

    # Second config file with absolute path
    config_file_2 = '/home/prisma-lab/kuka_rl_ros2/src/lbr-stack/lbr_fri_ros2_stack/lbr_description/ros2_control/lbr_system_config.yaml'
    
    # Include the robot hardware launch file
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            '/home/user/kuka_rl_ros2/src/lbr-stack/lbr_fri_ros2_stack/lbr_bringup/launch/hardware_pose_streaming.launch.py'
        )
    )
    
    # Include the OptiTrack launch file
    optitrack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            '/home/user/kuka_rl_ros2/src/ros2_optitrack_listener/launch/optitrack_listener.launch.py'
        )
    )
    
    # Add the joint state node with parameters
    joint_state_node = Node(
        package='catch_and_throw',
        executable='joint_state_node_trained_bouncing',
        name='joint_state_node',
        parameters=[config_file_1, config_file_2],
        output='screen'
    )
    
    return LaunchDescription([
        robot_launch,
        optitrack_launch,
        joint_state_node,  # Now launches automatically with parameters
    ])