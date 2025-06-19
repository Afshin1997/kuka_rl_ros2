qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Changed from BEST_EFFORT
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0, nanoseconds=1000000)  # 1ms
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/lbr/joint_states',
            self.joint_states_callback,
            qos_profile=qos_profile
        )
        self.ee_pose_sub = self.create_subscription(
            Pose,
            '/lbr/state/pose',
            self.ee_pose_callback,
            qos_profile=qos_profile
        )
        self.ee_vel_sub = self.create_subscription(
            Twist,
            '/lbr/state/twist',
            self.ee_vel_callback,
            qos_profile=qos_profile
        )
        
        # Publisher
        self.joint_ref_pub = self.create_publisher(JointState, '/lbr/command/joint_position', qos_profile=qos_profile)
        




















############################


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
import pandas as pd
import numpy as np
import os
import threading
from rclpy.executors import MultiThreadedExecutor

def save_output(outputs, output_file_path, header=None):
    np.savetxt(output_file_path, outputs, delimiter=',', header=header, comments='')
    print(f"Model outputs saved to {output_file_path}")

class JointStateNode(Node):
    def __init__(self):
        super().__init__('joint_state_node')
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/lbr/joint_states',
            self.joint_states_callback,
            10
        )
        self.ee_pose_sub = self.create_subscription(
            Pose,
            '/lbr/state/pose',
            self.ee_pose_callback,
            10
        )
        self.ee_vel_sub = self.create_subscription(
            Twist,
            '/lbr/state/twist',
            self.ee_vel_callback,
            10
        )
        
        # Publisher
        self.joint_ref_pub = self.create_publisher(JointState, '/joint_reference', 10)
        
        # Data storage
        self.received_joint_pos = []
        self.received_joint_vel = []
        self.received_ee_pos = []
        self.received_ee_orient = []
        self.received_ee_lin_vel = []
        self.recording = False

    def joint_states_callback(self, msg):
        if self.recording:
            self.received_joint_pos.append(list(msg.position))
            self.received_joint_vel.append(list(msg.velocity))

    def ee_pose_callback(self, msg):
        if self.recording:
            self.received_ee_pos.append([msg.position.x, msg.position.y, msg.position.z])
            self.received_ee_orient.append([
                msg.orientation.w, 
                msg.orientation.x, 
                msg.orientation.y, 
                msg.orientation.z
            ])

    def ee_vel_callback(self, msg):
        if self.recording:
            self.received_ee_lin_vel.append([msg.linear.x, msg.linear.y, msg.linear.z])

def main():
    rclpy.init()
    node = JointStateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Start executor in separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        # Load CSV data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(script_dir, '../../input_files/idealpd/ft_idealpd.csv')
        input_file_path = os.path.abspath(input_file_path)
        obs_data = pd.read_csv(input_file_path)
        set_target = obs_data.iloc[:800, 21:28].values  # Joint positions
        # set_target = obs_data.iloc[:800, 14:21].values  ## set target

        # Start recording
        node.recording = True

        # Publish joint references at 200Hz
        rate = node.create_rate(200)
        for t in range(set_target.shape[0]):
            if not rclpy.ok():
                break
            
            msg = JointState()
            msg.position = list(set_target[t])
            # node.joint_ref_pub.publish(msg)
            rate.sleep()

        node.get_logger().info("Finished publishing all targets")

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected")
    finally:
        # Stop recording and shutdown
        node.recording = False
        node.get_logger().info("Saving collected data...")
        
        # Ensure all callbacks have completed
        rclpy.shutdown()
        executor_thread.join()

        # Save outputs
        output_dir = os.path.join(script_dir, '../../output_files/ft/idealpd')
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        save_output(
            np.array(node.received_joint_pos), 
            os.path.join(output_dir, "ft_received_joint_pos_np.csv"),
            "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6"
        )
        save_output(
            np.array(node.received_joint_vel), 
            os.path.join(output_dir, "ft_received_joint_vel_np.csv"),
            "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6"
        )
        save_output(
            np.array(node.received_ee_pos), 
            os.path.join(output_dir, "ft_received_ee_pos_np.csv"),
            "pos_X,pos_Y,pos_Z"
        )
        save_output(
            np.array(node.received_ee_orient), 
            os.path.join(output_dir, "ft_received_ee_orient_np.csv"),
            "or_w,or_x,or_y,or_z"
        )
        save_output(
            np.array(node.received_ee_lin_vel), 
            os.path.join(output_dir, "ft_received_ee_lin_vel_np.csv"),
            "lin_vel_X,lin_vel_Y,lin_vel_Z"
        )

if __name__ == '__main__':
    main()