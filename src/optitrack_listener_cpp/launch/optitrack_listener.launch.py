from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_file = PathJoinSubstitution([
        get_package_share_directory('optitrack_listener_cpp'),
        'config',
        'optitrack.yaml'
    ])
    
    return LaunchDescription([
        Node(
            package='optitrack_listener_cpp',
            executable='optitrack_listener',
            name='optitrack_listener',
            output='screen',
            parameters=[config_file],
            arguments=['--ros-args', '--log-level', 'info']
        )
    ])