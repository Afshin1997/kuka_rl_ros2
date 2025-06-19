#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import pandas as pd
import os
from threading import Lock
from tf_transformations import quaternion_matrix


class DeploymentPolicy:
    def __init__(self, checkpoint_path):
        # Load exported JIT model
        self.model = torch.jit.load(checkpoint_path)
        self.model.eval()

    # In get_action():
    def get_action(self, raw_obs):
        # Ensure CPU tensor conversion
        obs_tensor = torch.as_tensor(raw_obs, dtype=torch.float32).unsqueeze(0).cpu()
        with torch.no_grad():
            action = self.model(obs_tensor)
        return action.cpu().squeeze(0).numpy()

class JointStateNode(Node):
    def __init__(self):
        super().__init__('joint_state_node')

        self.declare_parameter('model_path', 
                              '/home/user/ros_ws/src/catch_and_throw/input_files/idealpd/model_idealpd.pt')
        
        # ROS2 QoS Profile - important for real-time systems
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,  # Changed from BEST_EFFORT
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0, nanoseconds=1000000)  # 1ms
        )

        # Parameters
        self.dt = 1.0 / 200.0
        self.tennis_ball_pos_scale = 0.25
        self.lin_vel_scale = 0.15
        self.to_final_target_scale = 0.5
        self.dof_vel_scale = 0.31
        self.action_scale = 0.09
        self.to_throwing_pos_scale = 0.5
        self.to_throwing_vel_scale = 0.4
        self.moving_average = 0.8

        # Joint limits and velocity limits
        self.robot_dof_lower_limits = torch.tensor(
            [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543]
        )
        self.robot_dof_upper_limits = torch.tensor(
            [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]
        )
        self.robot_dof_vel_limit = torch.tensor(
            [1.713, 1.713, 1.745, 2.268, 2.443, 3.142, 3.142]
        )

        self.robot_dof_lower_limits_np = self.robot_dof_lower_limits.numpy()
        self.robot_dof_upper_limits_np = self.robot_dof_upper_limits.numpy()

        self.final_target_pos = np.array([-0.65, -0.4, 0.55], dtype=np.float32)
        self.throwing_pos = np.array([0.2, -0.35, 0.9], dtype=np.float32)
        self.throwing_vel = np.array([2.5, 0.0, 0.5], dtype=np.float32)

        # Load observation data
        self.load_observation_data()

        # Initialize policy
        model_path = self.get_parameter('model_path').value
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Policy not found at {model_path}")
        self.policy = DeploymentPolicy(model_path)

        self.robot_dof_targets = None
        self.lock = Lock()
        self.t = 0

        # Initialize observation buffers
        self.ee_pose = None
        self.ee_orientation = None
        self.ee_vel = None
        self.ee_angular_vel = None
        self.joint_positions_obs = None
        self.joint_velocities_obs = None

        self.action_logits = torch.zeros(7, dtype=torch.float32).unsqueeze(0)
        self.pure_action = torch.zeros(7, dtype=torch.float32).unsqueeze(0)

        # For saving outputs
        self.joint_targets = []
        self.joint_poses = []
        self.joint_velocities = []
        self.ee_poses = []
        self.ee_orientations = []
        self.ee_volocities = []

        self.action_history = np.zeros((2, 7))

        # Publishers
        self.joint_ref_pub = self.create_publisher(LBRJointPositionCommand, '/lbr/command/joint_position', 0)

        # Subscribers
        self.joint_states_sub = self.create_subscription(
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

        self.timer = self.create_timer(self.dt, self.compute_action_and_publish)

        self.get_logger().info("JointStateNode initialized")

    def load_observation_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(script_dir, '../../input_files/idealpd/ft_idealpd.csv')
        input_file_path = os.path.abspath(input_file_path)
        if not os.path.exists(input_file_path):
            self.get_logger().error(f"Observation data file not found: {input_file_path}")
            raise FileNotFoundError("Missing observation data file")

        obs_data = pd.read_csv(input_file_path)
        self.tennisball_pos = obs_data[['tennisball_pos_0', 'tennisball_pos_1', 'tennisball_pos_2']].values
        self.tennisball_lin_vel = obs_data[['tennisball_lin_vel_0', 'tennisball_lin_vel_1', 'tennisball_lin_vel_2']].values
        self.obs_task = obs_data[['obs_task_0', 'obs_task_1', 'obs_task_2', 'obs_task_3', 'obs_task_4', 'obs_task_5', 'obs_task_6']].values
        self.to_final_target = obs_data[['to_final_target_0', 'to_final_target_1', 'to_final_target_2']].values


    def joint_states_callback(self, msg):
        desired_order = ['lbr_A1', 'lbr_A2', 'lbr_A3', 'lbr_A4', 
                            'lbr_A5', 'lbr_A6', 'lbr_A7']
            
        # Get indices for proper ordering
        indices = [msg.name.index(name) for name in desired_order]
        with self.lock:
            self.joint_positions_obs = np.array(msg.position[i] for i in indices, dtype=np.float32)
            self.joint_velocities_obs = np.array(msg.velocity[i] for i in indices, dtype=np.float32)
            if self.robot_dof_targets is None:
                self.robot_dof_targets = torch.tensor(msg.position, dtype=torch.float32).unsqueeze(0)

    def ee_pose_callback(self, msg):
        with self.lock:
            self.ee_pose = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
            self.ee_orientation = np.array([msg.orientation.w, msg.orientation.x, 
                                          msg.orientation.y, msg.orientation.z], dtype=np.float32)

    def ee_vel_callback(self, msg):
        with self.lock:
            self.ee_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32)
            self.ee_angular_vel = np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32)


    def compute_action_and_publish(self):
        with self.lock:
            # Check if we have all data needed
            if self.joint_positions_obs is None or self.joint_velocities_obs is None:
                return
            if self.ee_pose is None or self.ee_orientation is None or self.ee_vel is None or self.ee_angular_vel is None:
                return
            if self.t >= len(self.tennisball_pos):
                self.get_logger().warn("Time index exceeded data length. Shutting down node.")
                self.destroy_node()
                rclpy.shutdown()
                return

            # Scale joint observations
            dof_pos_scaled_obs = 2.0 * (self.joint_positions_obs - self.robot_dof_lower_limits_np) / \
                (self.robot_dof_upper_limits_np - self.robot_dof_lower_limits_np) - 1.0
            dof_vel_scaled_obs = self.joint_velocities_obs * self.dof_vel_scale

            tennisball_pos_obs = self.tennisball_pos[self.t, :] * self.tennis_ball_pos_scale
            tennisball_lin_vel_obs = self.tennisball_lin_vel[self.t, :] * self.lin_vel_scale
            ee_lin_vel_scaled = self.ee_vel * self.lin_vel_scale
            to_final_target_obs = self.to_final_target[self.t, :] * self.to_final_target_scale
            obs_task_obs = self.obs_task[self.t, :]
            # Calculate conditions (matches training environment logic)
            condition_final_target = (obs_task_obs[3] == 1.0) or (obs_task_obs[4] == 1.0)  # Moving_to_target_pos or Stabilized_at_Target
            condition_throwing = (obs_task_obs[5] == 1.0)  # Throwing_Ball

            action = self.action_logits.cpu().numpy().flatten() if self.action_logits is not None else np.ones(7, dtype=np.float32)
            pure_action = self.action_logits.cpu().numpy().flatten() if self.pure_action is not None else np.zeros(7, dtype=np.float32)
            last_action = action


            to_final_target = np.where(
                np.expand_dims(condition_final_target, axis=-1),
                (self.final_target_pos - self.tennisball_pos[self.t, :]),
                np.zeros(3)
            )

            to_throwing_vel = np.where(
                np.expand_dims(condition_throwing, axis=-1),
                (self.throwing_vel - self.tennisball_lin_vel[self.t, :]),
                np.zeros(3)
            )

            to_throwing_pos = np.where(
                np.expand_dims(condition_throwing, axis=-1),
                self.throwing_pos - self.tennisball_pos[self.t, :],
                np.zeros(3)
            )

            # Apply scaling factors (matches training environment)
            to_final_target_obs = to_final_target * self.to_final_target_scale
            to_throwing_pos_scaled = to_throwing_pos * self.to_throwing_pos_scale
            to_throwing_vel_scaled = to_throwing_vel * self.to_throwing_vel_scale

            self.action_history = np.roll(self.action_history, shift=-1, axis=0)
            self.action_history[-1:] = pure_action


            if self.ee_pose is not None and self.ee_orientation is not None:
                new_point_pos = self.compute_offset_point(
                    position=self.ee_pose,
                    orientation=self.ee_orientation,
                    offset_local=np.array([0, 0, 0.02])  # 2 cm along z-axis
                )
                # Use the offset point position instead of the end effector's position
                ee_pos_for_obs = new_point_pos

            else:
                # Handle cases where pose or orientation is not yet available
                ee_pos_for_obs = np.array([0.33434677, -0.33252597, 0.9089085], dtype=np.float32)
                new_point_pos = np.array([0.33434677, -0.33252597, 0.9089085], dtype=np.float32)

            observations = np.concatenate((
                action.flatten(),                     # 7
                self.action_history.flatten(),        # 14 (2x7)
                dof_pos_scaled_obs.flatten(),         # 7
                dof_vel_scaled_obs.flatten(),         # 7
                tennisball_pos_obs.flatten(),         # 3
                ee_pos_for_obs.flatten(),             # 3
                tennisball_lin_vel_obs.flatten(),     # 3
                ee_lin_vel_scaled.flatten(),          # 3
                self.ee_orientation.flatten(),        # 4
                to_final_target_obs.flatten(),        # 3
                to_throwing_pos_scaled.flatten(),     # 3
                to_throwing_vel_scaled.flatten(),     # 3
                obs_task_obs.flatten()                # 7 (MUST BE LAST!)
            ))

            assert observations.size == 67, f"Observation size mismatch: expected 67, got {observations.size}"

            # Normalize and get action
            action_logits = self.policy.get_action(observations)
            self.action_logits = action_logits

            # Compute new targets
            joint_positions_obs_tensor = torch.tensor(self.joint_positions_obs, dtype=torch.float32)
            scaled_actions = action_logits.squeeze(0) * 0.1
            targets = joint_positions_obs_tensor + scaled_actions
            self.robot_dof_targets = torch.clamp(
                targets,
                0.975 * self.robot_dof_lower_limits,
                0.975 * self.robot_dof_upper_limits
            )

            # Save outputs for debugging
            self.joint_targets.append(targets.numpy().flatten())
            self.joint_poses.append(self.joint_positions_obs)
            self.joint_velocities.append(self.joint_velocities_obs)
            self.ee_poses.append(self.ee_pose)
            self.ee_orientations.append(self.ee_orientation)
            self.ee_volocities.append(self.ee_vel)

            # Publish
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            # Assuming joint names correspond to the robot
            # If you know the joint names of your robot, add them here:
            joint_state_msg.name = [f"joint_{i}" for i in range(7)]
            joint_state_msg.position = self.robot_dof_targets.squeeze(0).tolist()

            self.joint_ref_pub.publish(joint_state_msg)

            self.t += 1

    def save_output(self, outputs, output_file_path, header=None):
        outputs = np.array(outputs)
        np.savetxt(output_file_path, outputs, delimiter=',', header=header, comments='')
        self.get_logger().info(f"Model outputs saved to {output_file_path}")

    def compute_offset_point(self, position, orientation, offset_local=np.array([0, 0, 0.02])):
        """Compute the global position of a point offset from the end effector's position."""
        # Validate the quaternion
        norm = np.linalg.norm(orientation)
        if norm == 0:
            self.get_logger().warn("Received zero quaternion. Using identity rotation.")
            rot_matrix = np.eye(3)
        else:
            normalized_quat = orientation / norm
            rot_matrix = quaternion_matrix(normalized_quat)[:3, :3]

        # Compute global offset
        global_offset = rot_matrix @ offset_local  # Matrix multiplication

        # Compute new point position
        new_point_pos = position + global_offset

        return new_point_pos
    
    def destroy_node(self):
        # Save outputs on shutdown
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../../output_files/tm/idealpd')
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.save_output(self.joint_targets, os.path.join(output_dir, "tm_received_joint_target_np.csv"),
                         "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_poses, os.path.join(output_dir, "tm_received_joint_pos_np.csv"),
                         "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_velocities, os.path.join(output_dir, "tm_received_joint_vel_np.csv"),
                         "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.ee_poses, os.path.join(output_dir, "tm_received_ee_pos_np.csv"),
                         "pos_X,pos_Y,pos_Z")
        self.save_output(self.ee_orientations, os.path.join(output_dir, "tm_received_ee_orientation_np.csv"),
                         "or_w,or_x,or_y,or_z")
        self.save_output(self.ee_volocities, os.path.join(output_dir, "tm_received_ee_vel_np.csv"),
                         "lin_vel_X,lin_vel_Y,lin_vel_Z")
        
        self.get_logger().info("Model outputs saved on shutdown.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = JointStateNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in JointStateNode: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()