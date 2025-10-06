from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Include the robot hardware launch file
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('lbr_bringup'),
                'launch',
                'hardware_pose_streaming.launch.py'
            ])
        ])
    )
    
    # Include the OptiTrack launch file
    optitrack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('optitrack_listener'),
                'launch',
                'optitrack_listener.launch.py'
            ])
        ])
    )
    
    # Your integrated node with OptiTrack support
    # Adding a delay to ensure other systems are up first
    integrated_node = TimerAction(
        period=5.0,  # 5 second delay to let other systems initialize
        actions=[
            Node(
                package='catch_and_throw',
                executable='joint_state_node',  # or 'joint_state_node_trained' if using that one
                name='joint_state_with_optitrack_node',
                output='screen',
                parameters=[
                    # Add any parameters your node needs
                ]
            )
        ]
    )
    
    return LaunchDescription([
        robot_launch,
        optitrack_launch,
        integrated_node
    ])
