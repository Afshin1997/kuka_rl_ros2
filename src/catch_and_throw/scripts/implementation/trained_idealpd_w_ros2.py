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
import numpy as np
import pandas as pd
import os
from threading import Lock
from tf_transformations import quaternion_matrix  # For compute_offset_point

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {act_name}")

class ActorCriticAfshin(nn.Module):
    def __init__(
        self,
        num_actor_obs=67,       
        num_critic_obs=116,    
        num_actions=7,
        actor_hidden_dims=[256, 128, 128, 32],  # Exactly these sizes
        critic_hidden_dims=[256, 128, 128, 64], # Exactly these sizes
        num_categories=7,
        embedding_dim=5
    ):
        super().__init__()
        self.num_categories = num_categories
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Actor Network (EXACT original structure)
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs - num_categories + embedding_dim, 256), # layer 0
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 7),
            nn.Softsign()
        )
        
        # Critic Network (EXACT original structure)
        self.critic = nn.Sequential(
            nn.Linear(114, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
        self.std = nn.Parameter(torch.ones(num_actions))

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        

    def _process_observations(self, observations: torch.Tensor):
        """
        Process observations by replacing the last num_categories features (one-hot)
        with an embedding vector.
        """
        batch_size = observations.shape[0]
        categorical_obs = torch.argmax(observations[:, -self.num_categories:], dim=1)  # [batch]
        continuous_obs = observations[:, :-self.num_categories]  # [batch, original_dim - num_categories]

        embedded = self.embedding(categorical_obs)  # [batch, embedding_dim]

        combined = torch.cat([continuous_obs, embedded], dim=-1)  # [batch, adjusted_dim]
        print("combined", combined)
        return combined



    def act_inference(self, observations):
        combined = self._process_observations(observations)
        actions_mean = self.actor(combined)
        return actions_mean


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""
    def __init__(self, shape=67, eps=1e-8, task_dim=7):
        super().__init__()
        self.eps = eps
        self.task_dim = task_dim
        self.main_dim = shape - task_dim
        
        # Register buffers properly
        self.register_buffer("_mean", torch.zeros(self.main_dim).unsqueeze(0))
        self.register_buffer("_var", torch.ones(self.main_dim).unsqueeze(0))
        self.register_buffer("_std", torch.ones(self.main_dim).unsqueeze(0))

    def forward(self, x):
        x_main = x[:, :-self.task_dim]
        x_task = x[:, -self.task_dim:]
        
        # Use registered buffers
        x_main_norm = (x_main - self._mean) / (self._std + self.eps)
        return torch.cat([x_main_norm, x_task], dim=1)

class JointStateNode(Node):
    def __init__(self):
        super().__init__('joint_state_node')

        self.declare_parameter('model_path', 
                              '/home/user/ros_ws/src/catch_and_throw/input_files/idealpd/model_idealpd.pt')
        
        # ROS2 QoS Profile - important for real-time systems
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Changed from BEST_EFFORT
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

        # Load actor and normalizer
        self.actor, self.obs_normalizer = self.load_actor_and_normalization()

        if self.actor is None or self.obs_normalizer is None:
            self.get_logger().error("Actor or Normalizer failed to load. Shutting down node.")
            raise RuntimeError("Actor or Normalizer failed to load")

        self.actor.eval()
        self.obs_normalizer.eval()

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
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_reference',
            qos_profile=qos_profile
        )
        # Subscribers
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            qos_profile=qos_profile
        )

        self.ee_pose_sub = self.create_subscription(
            Pose,
            '/ee_pose',
            self.ee_pose_callback,
            qos_profile=qos_profile
        )

        self.ee_vel_sub = self.create_subscription(
            Twist,
            '/ee_vel',
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
        self.obs_task = obs_data[['obs_task_0', 'obs_task_1', 'obs_task_2', 'obs_task_3', 'obs_task_4']].values
        self.to_final_target = obs_data[['to_final_target_0', 'to_final_target_1', 'to_final_target_2']].values

    def load_actor_and_normalization(self):
        num_actor_obs = 67
        num_actions = 7
        num_categories = 7
        embedding_dim = 5

        actor = ActorCriticAfshin(
                num_actor_obs=67,
                num_critic_obs=116,
                num_actions=7,
                num_categories=num_categories,
                embedding_dim=embedding_dim
            )

        model_path = self.get_parameter('model_path').value
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            return None, None

        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('actor.')}
            actor.load_state_dict(actor_state_dict, strict=True)
            actor.eval()
            self.get_logger().info("Actor model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Error loading actor model: {e}")
            return None, None

        obs_normalizer = EmpiricalNormalization(shape=67, task_dim=7)
        obs_normalizer_state_dict = checkpoint.get('obs_norm_state_dict', None)
        if obs_normalizer_state_dict is not None:
            try:
                norm_state = checkpoint['obs_norm_state_dict']
                renamed_norm_state = {
                    '_mean': norm_state['_mean_cont'],
                    '_var': norm_state['_var_cont'],
                    '_std': norm_state['_std_cont']
                }
                obs_normalizer.load_state_dict(renamed_norm_state)
                obs_normalizer.eval()
                self.get_logger().info("Normalization state loaded successfully.")
            except Exception as e:
                self.get_logger().warning(f"Failed to load normalization state: {e}")
        else:
            self.get_logger().warning("No normalization state found in checkpoint.")

        return actor, obs_normalizer

    def joint_states_callback(self, msg):
        with self.lock:
            self.joint_positions_obs = np.array(msg.position, dtype=np.float32)
            self.joint_velocities_obs = np.array(msg.velocity, dtype=np.float32)
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
                action.flatten(), #7
                self.action_history.flatten(),
                dof_pos_scaled_obs.flatten(), #7
                dof_vel_scaled_obs.flatten(), #7
                tennisball_pos_obs.flatten(), #3
                ee_pos_for_obs.flatten(), #3
                tennisball_lin_vel_obs.flatten(), #3
                ee_lin_vel_scaled.flatten(), #3
                self.ee_orientation.flatten(), #4
                to_final_target_obs.flatten(), #3
                to_throwing_pos_scaled.flatten(), #3
                to_throwing_vel_scaled.flatten(), #3
                obs_task_obs.flatten() #5
            ))

            assert observations.size == 67, f"Observation size mismatch: expected 67, got {observations.size}"

            # Normalize and get action
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).float().unsqueeze(0)
                
                normalized_obs = self.obs_normalizer(observations_tensor)
                
                action_logits = self.actor.act_inference(normalized_obs)
                self.action_logits = action_logits

                if self.t < 3:
                    print('without normalization', observations_tensor)
                    print('with normalization', normalized_obs)


                # Compute new targets
                joint_positions_obs_tensor = torch.tensor(self.joint_positions_obs, dtype=torch.float32)
                scaled_actions = action_logits.squeeze(0) * 0.12
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

            self.joint_state_pub.publish(joint_state_msg)

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