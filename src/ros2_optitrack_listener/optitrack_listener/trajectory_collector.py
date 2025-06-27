#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import pandas as pd
from datetime import datetime
import os
from collections import deque

class BallTrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('ball_trajectory_recorder')
        
        # Declare parameters for velocity smoothing
        self.declare_parameter('velocity_ema_alpha', 0.5)
        self.declare_parameter('velocity_window_size', 7)
        self.declare_parameter('use_savgol_filter', True)
        self.declare_parameter('savgol_window_length', 11)
        self.declare_parameter('savgol_poly_order', 3)
        
        # Get parameters
        self.vel_ema_alpha = self.get_parameter('velocity_ema_alpha').get_parameter_value().double_value
        self.window_size = self.get_parameter('velocity_window_size').get_parameter_value().integer_value
        self.use_savgol = self.get_parameter('use_savgol_filter').get_parameter_value().bool_value
        self.savgol_window = self.get_parameter('savgol_window_length').get_parameter_value().integer_value
        self.savgol_order = self.get_parameter('savgol_poly_order').get_parameter_value().integer_value
        
        # Subscribe to the ball marker topic
        self.subscription = self.create_subscription(
            PoseStamped,
            'optitrack/ball_marker',
            self.ball_callback,
            10
        )
        
        # Data storage
        self.trajectory_data = []
        self.is_recording = False
        self.last_time = None
        self.last_position = None
        
        # Velocity smoothing variables
        self.velocity_ema = None
        self.position_history = deque(maxlen=self.window_size)
        self.time_history = deque(maxlen=self.window_size)
        
        # OptiTrack data frequency
        self.data_frequency = 200.0  # Hz
        self.dt = 1.0 / self.data_frequency  # Time step in seconds
        
        # Create output directory if it doesn't exist
        self.output_dir = 'ball_trajectories'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.get_logger().info('Ball trajectory recorder started')
        self.get_logger().info(f'Velocity EMA alpha: {self.vel_ema_alpha}')
        self.get_logger().info(f'Moving average window: {self.window_size}')
        self.get_logger().info(f'Use Savitzky-Golay filter: {self.use_savgol}')
        self.get_logger().info(f'Data will be saved to: {self.output_dir}')

        self.is_recording =  True
    
    def transform_to_world_frame(self, pos_optitrack):
        """
        Transform position from OptiTrack frame to world frame
        """
        X = -pos_optitrack[2] + 0.7
        Y = -pos_optitrack[0] + 0.0
        Z = pos_optitrack[1] + 0.565
        
        return np.array([X, Y, Z])
    
    def calculate_velocity_moving_average(self):
        """
        Calculate velocity using moving average of multiple position differences
        """
        if len(self.position_history) < 2:
            return np.array([0.0, 0.0, 0.0])
        
        velocities = []
        for i in range(1, len(self.position_history)):
            dt = self.time_history[i] - self.time_history[i-1]
            if dt > 0:
                vel = (self.position_history[i] - self.position_history[i-1]) / dt
                velocities.append(vel)
        
        if velocities:
            return np.mean(velocities, axis=0)
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def calculate_velocity_central_difference(self):
        """
        Calculate velocity using central difference (more stable than forward difference)
        """
        if len(self.position_history) < 3:
            return self.calculate_velocity_moving_average()
        
        
        pos_prev = self.position_history[-2]
        pos_next = self.position_history[-1]
        time_prev = self.time_history[-2]
        time_next = self.time_history[-1]
        
        dt = time_next - time_prev
        if dt > 0:
            return (pos_next - pos_prev) / dt
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def update_velocity_ema(self, new_velocity):
        """
        Update velocity using exponential moving average
        """
        if self.velocity_ema is None:
            self.velocity_ema = new_velocity.copy()
        else:
            self.velocity_ema = (self.vel_ema_alpha * new_velocity + 
                               (1.0 - self.vel_ema_alpha) * self.velocity_ema)
        
        return self.velocity_ema
    
    def ball_callback(self, msg):
        # Extract position from message
        pos_optitrack = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Transform to world frame
        pos_world = self.transform_to_world_frame(pos_optitrack)
        
        # Get current time
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Add to history
        self.position_history.append(pos_world)
        self.time_history.append(current_time)
        
        # Calculate raw velocity using central difference
        raw_velocity = self.calculate_velocity_central_difference()
        
        # Apply EMA smoothing to velocity
        smooth_velocity = self.update_velocity_ema(raw_velocity)
        
        # Check recording conditions
        x_world = pos_world[0]
        z_world = pos_world[2]

        
        # # Start recording when x < 3.0
        # if x_world < 10.0 and not self.is_recording:
        #     self.is_recording = True
        #     self.trajectory_data = []  # Clear previous data
        #     self.get_logger().info(f'Started recording at X={x_world:.3f}m')
        
        # # Stop recording and save when x < 0.0 or z < 0.3
        # if self.is_recording and (x_world < -2.0 or z_world < 0.1):
        #     self.is_recording = False
        #     self.get_logger().info(f'Stopped recording at X={x_world:.3f}m, Z={z_world:.3f}m')
        #     self.save_trajectory()
        
        # Store data if recording
        if self.is_recording:
            data_point = {
                'timestamp': current_time,
                'x_world': pos_world[0],
                'y_world': pos_world[1],
                'z_world': pos_world[2],
                'vx_world_raw': raw_velocity[0],
                'vy_world_raw': raw_velocity[1],
                'vz_world_raw': raw_velocity[2],
                'vx_world': smooth_velocity[0],
                'vy_world': smooth_velocity[1],
                'vz_world': smooth_velocity[2],
                'speed_raw': np.linalg.norm(raw_velocity),
                'speed_smooth': np.linalg.norm(smooth_velocity)
            }
            self.trajectory_data.append(data_point)
        
        # Update last position and time
        self.last_position = pos_world
        self.last_time = current_time
    
    # def apply_savgol_filter(self, df):
    #     """
    #     Apply Savitzky-Golay filter to smooth velocities post-processing
    #     """
    #     try:
    #         from scipy.signal import savgol_filter
            
    #         if len(df) >= self.savgol_window:
    #             # Apply filter to velocity components
    #             df['vx_world_savgol'] = savgol_filter(df['vx_world_raw'], 
    #                                                 self.savgol_window, 
    #                                                 self.savgol_order)
    #             df['vy_world_savgol'] = savgol_filter(df['vy_world_raw'], 
    #                                                 self.savgol_window, 
    #                                                 self.savgol_order)
    #             df['vz_world_savgol'] = savgol_filter(df['vz_world_raw'], 
    #                                                 self.savgol_window, 
    #                                                 self.savgol_order)
                
    #             # Calculate speed from filtered velocities
    #             df['speed_savgol'] = np.sqrt(df['vx_world_savgol']**2 + 
    #                                        df['vy_world_savgol']**2 + 
    #                                        df['vz_world_savgol']**2)
                
    #             self.get_logger().info('Applied Savitzky-Golay filter to velocities')
    #         else:
    #             self.get_logger().warn(f'Not enough data points ({len(df)}) for Savitzky-Golay filter (requires {self.savgol_window})')
                
    #     except ImportError:
    #         self.get_logger().warn('scipy not available, skipping Savitzky-Golay filter')
        
    #     return df
    
    def save_trajectory(self):
        """Save the recorded trajectory to a CSV file"""
        if not self.trajectory_data:
            self.get_logger().warn('No data to save')
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'ball_trajectory_{timestamp}.csv')
        
        # Convert to DataFrame
        df = pd.DataFrame(self.trajectory_data)
        
        # # Apply post-processing filter if enabled
        # if self.use_savgol:
        #     df = self.apply_savgol_filter(df)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Calculate and log statistics
        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        max_speed_raw = df['speed_raw'].max()
        max_speed_smooth = df['speed_smooth'].max()
        max_height = df['z_world'].max()
        distance_traveled = df['x_world'].iloc[0] - df['x_world'].iloc[-1]
        
        self.get_logger().info(f'Trajectory saved to: {filename}')
        self.get_logger().info(f'Total points: {len(self.trajectory_data)}')
        self.get_logger().info(f'Duration: {duration:.3f}s')
        self.get_logger().info(f'Max speed (raw): {max_speed_raw:.3f}m/s')
        self.get_logger().info(f'Max speed (smooth): {max_speed_smooth:.3f}m/s')
        self.get_logger().info(f'Max height: {max_height:.3f}m')
        self.get_logger().info(f'Distance traveled: {distance_traveled:.3f}m')
        
        # Clear data after saving
        self.trajectory_data = []


def main(args=None):
    rclpy.init(args=args)
    node = BallTrajectoryRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Save any remaining data before shutdown
        if node.is_recording and node.trajectory_data:
            node.get_logger().info('Saving data before shutdown...')
            node.save_trajectory()
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()