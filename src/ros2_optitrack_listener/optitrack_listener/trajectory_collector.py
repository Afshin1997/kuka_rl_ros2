import rclpy
from rclpy.node import Node
from threading import Thread
import optirx as rx
import socket
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import numpy as np
import os
import csv
from datetime import datetime

class OptitrackMarkerRecorder(Node):
    def __init__(self):
        super().__init__('optitrack_marker_recorder')
        
        # Network configuration
        self.declare_parameter('local_interface', "172.31.1.145")
        self.ipaddr = self.get_parameter('local_interface').get_parameter_value().string_value
        
        # Frame configuration
        self.declare_parameter('fixed_frame', 'world')
        self.frame = self.get_parameter('fixed_frame').get_parameter_value().string_value
        
        # Ball marker tracking parameters
        self.declare_parameter('initial_ball_position', [0.0, 0.0, 0.0])
        self.ball_marker_old = self.get_parameter('initial_ball_position').get_parameter_value().double_array_value
        
        # EMA filtering parameters
        self.declare_parameter('ball_ema_alpha', 0.1)
        self.ema_alpha = self.get_parameter('ball_ema_alpha').get_parameter_value().double_value
        
        # Initialize EMA state
        self.ball_ema_position = np.array(self.ball_marker_old)
        self.ema_initialized = False
        
        # Trajectory recording parameters
        self.declare_parameter('trajectory_save_folder', 'marker_trajectories')
        self.trajectory_folder = self.get_parameter('trajectory_save_folder').get_parameter_value().string_value
        
        # Recording boundaries (world frame)
        self.declare_parameter('recording_x_min', 0.0)
        self.declare_parameter('recording_x_max', 3.0)
        self.declare_parameter('recording_z_min', 0.0)
        self.x_min = self.get_parameter('recording_x_min').get_parameter_value().double_value
        self.x_max = self.get_parameter('recording_x_max').get_parameter_value().double_value
        self.z_min = self.get_parameter('recording_z_min').get_parameter_value().double_value
        
        # Create trajectory folder if it doesn't exist
        if not os.path.exists(self.trajectory_folder):
            os.makedirs(self.trajectory_folder)
            self.get_logger().info(f"Created trajectory folder: {self.trajectory_folder}")
        
        # Initialize trajectory recording state
        self.is_recording = False
        self.trajectory_data = []
        self.recording_start_time = None
        self.trajectory_file_index = 0
        
        # Publisher for ball marker
        self.ball_marker_publisher = self.create_publisher(PoseStamped, "optitrack/ball_marker", 1)
        self.ball_pose_stamped = PoseStamped()
        
        self.get_logger().info(f"EMA alpha: {self.ema_alpha}")
        self.get_logger().info(f"Recording zone: {self.x_min} < x < {self.x_max}, z > {self.z_min}")
        
        # Start OptiTrack data thread
        self.thread = Thread(target=self.get_optitrack_data, daemon=True)
        self.thread.start()
    
    def transform_to_world_frame(self, pos_optitrack):
        """Transform OptiTrack coordinates to world frame coordinates"""
        X = -pos_optitrack[2] + 0.7
        Y = -pos_optitrack[0] + 0.0
        Z = pos_optitrack[1] + 0.5106 + 0.05375  # flange height + frame offset
        return np.array([X, Y, Z])
    
    def update_ema_position(self, new_position):
        """Update the exponential moving average of the ball position"""
        new_pos_array = np.array(new_position)
        
        if not self.ema_initialized:
            self.ball_ema_position = new_pos_array.copy()
            self.ema_initialized = True
        else:
            self.ball_ema_position = self.ema_alpha * new_pos_array + (1 - self.ema_alpha) * self.ball_ema_position
        
        return self.ball_ema_position
    
    def is_in_recording_zone(self, position):
        """Check if the marker position is within the recording boundaries"""
        x, y, z = position
        return self.x_min < x < self.x_max and z > self.z_min
    
    def save_trajectory(self):
        """Save the recorded trajectory to a CSV file"""
        if len(self.trajectory_data) == 0:
            self.get_logger().warn("No trajectory data to save")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{timestamp}_{self.trajectory_file_index}.csv"
        filepath = os.path.join(self.trajectory_folder, filename)
        
        # Write data to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['timestamp', 
                           'opti_x', 'opti_y', 'opti_z', 
                           'opti_x_ema', 'opti_y_ema', 'opti_z_ema',
                           'world_x', 'world_y', 'world_z',
                           'world_x_ema', 'world_y_ema', 'world_z_ema'])
            # Write data
            writer.writerows(self.trajectory_data)
        
        self.get_logger().info(f"Saved trajectory with {len(self.trajectory_data)} points to: {filepath}")
        self.trajectory_file_index += 1
    
    def record_position(self, raw_position, ema_position, world_position, world_ema_position, timestamp):
        """Record a position if in recording zone, handle recording state transitions"""
        in_zone = self.is_in_recording_zone(world_ema_position)
        
        if in_zone and not self.is_recording:
            # Start recording
            self.is_recording = True
            self.trajectory_data = []
            self.recording_start_time = timestamp
            self.get_logger().info("Started recording trajectory")
        
        elif not in_zone and self.is_recording:
            # Stop recording and save
            self.is_recording = False
            self.save_trajectory()
            self.trajectory_data = []
            self.get_logger().info("Stopped recording trajectory")
        
        if self.is_recording:
            # Add data point (both OptiTrack and world frame coordinates)
            self.trajectory_data.append([
                timestamp,
                raw_position[0], raw_position[1], raw_position[2],
                ema_position[0], ema_position[1], ema_position[2],
                world_position[0], world_position[1], world_position[2],
                world_ema_position[0], world_ema_position[1], world_ema_position[2]
            ])
    
    def find_closest_marker(self, markers, reference):
        """Find the marker closest to the reference position"""
        if not markers:
            return None
        elif len(markers) == 1:
            return markers[0]
        
        # Use only x,z coordinates for horizontal plane matching
        markers_xz = np.array(markers)[:, [0, 2]]
        reference_xz = np.array(reference)[[0, 2]]
        
        distances = np.linalg.norm(markers_xz - reference_xz, axis=1)
        closest_index = np.argmin(distances)
        
        return markers[closest_index]
    
    def get_optitrack_data(self):
        """Main data acquisition loop"""
        version = (2, 7, 0, 0)
        self.get_logger().info(f"Connecting to OptiTrack at {self.ipaddr}")
        
        try:
            optitrack_socket = rx.mkdatasock(ip_address=self.ipaddr)
        except Exception as e:
            self.get_logger().error(f"Failed to connect to OptiTrack: {e}")
            return
        
        first_packet = True
        
        while rclpy.ok():
            try:
                data = optitrack_socket.recv(rx.MAX_PACKETSIZE)
                packet = rx.unpack(data, version=version)
                
                if first_packet:
                    # self.get_logger().info(f"Connected! NatNet version: {version}")
                    first_packet = True
                
                # Update version info if received
                if isinstance(packet, rx.SenderData):
                    version = packet.natnet_version
                    # self.get_logger().info(f"NatNet version updated: {version}")
                
                # Process frame data
                if isinstance(packet, (rx.SenderData, rx.ModelDefs, rx.FrameOfData)):
                    self.process_frame_data(packet)
                    
            except socket.error as e:
                self.get_logger().warning(f"Socket error: {e}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error: {e}")
    
    def process_frame_data(self, packet):
        """Process incoming frame data and publish/record ball marker"""
        # Update message header
        current_time = self.get_clock().now()
        self.ball_pose_stamped.header.stamp = current_time.to_msg()
        self.ball_pose_stamped.header.frame_id = self.frame
        
        # Find closest marker to previous position
        if hasattr(packet, 'other_markers'):
            ball_marker = self.find_closest_marker(packet.other_markers, self.ball_marker_old)
            
            if ball_marker is not None:
                # Apply EMA filtering (in OptiTrack frame)
                ema_position = self.update_ema_position(ball_marker)
                
                # Transform to world frame
                world_position = self.transform_to_world_frame(ball_marker)
                world_ema_position = self.transform_to_world_frame(ema_position)
                
                # Record trajectory if in zone (using world frame coordinates)
                timestamp = current_time.nanoseconds / 1e9  # Convert to seconds
                self.record_position(ball_marker, ema_position, world_position, world_ema_position, timestamp)
                
                # Update reference for next iteration (use raw measurement)
                self.ball_marker_old = ball_marker
                
                # Update pose message with world frame EMA-smoothed position
                self.ball_pose_stamped.pose.position = Point(
                    x=float(world_ema_position[0]), 
                    y=float(world_ema_position[1]), 
                    z=float(world_ema_position[2])
                )
                self.ball_pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # Publish the marker pose
        self.ball_marker_publisher.publish(self.ball_pose_stamped)
        print(f"Published ball marker at {self.ball_pose_stamped.pose.position}")
    
    def destroy_node(self):
        """Clean shutdown with trajectory saving"""
        if self.is_recording:
            self.save_trajectory()
            self.get_logger().info("Saved final trajectory on shutdown")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OptitrackMarkerRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()