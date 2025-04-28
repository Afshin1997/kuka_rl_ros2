#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from threading import Lock
import os
import tf.transformations as tf_trans
import time

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of only the continuous parts of the observation."""

    def __init__(self, num_continuous: int, total_dim: int, eps: float = 1e-2, until: int = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.num_continuous = num_continuous
        self.total_dim = total_dim

        # Initialize mean and variance for continuous features
        self.register_buffer("_mean_cont", torch.zeros(num_continuous).unsqueeze(0))
        self.register_buffer("_var_cont", torch.ones(num_continuous).unsqueeze(0))
        self.register_buffer("_std_cont", torch.ones(num_continuous).unsqueeze(0))
        self.count = 0

    def load_state_dict(self, state_dict, strict=True):
        # Handle old checkpoint format where keys are "_mean", "_var", "_std"
        if "_mean" in state_dict:
            state_dict["_mean_cont"] = state_dict.pop("_mean")
        if "_var" in state_dict:
            state_dict["_var_cont"] = state_dict.pop("_var")
        if "_std" in state_dict:
            state_dict["_std_cont"] = state_dict.pop("_std")

        # Call the parent class's load_state_dict method
        super().load_state_dict(state_dict, strict=strict)

    @property
    def mean_cont(self):
        return self._mean_cont.squeeze(0).clone()

    @property
    def std_cont(self):
        return self._std_cont.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x[:, :self.num_continuous])
        normalized_continuous = (x[:, :self.num_continuous] - self._mean_cont) / (self._std_cont + self.eps)
        return torch.cat([normalized_continuous, x[:, self.num_continuous:]], dim=1)

    def update(self, x: torch.Tensor):
        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        self.count += batch_size

        if self.count == 0:
            rate = 0.0  # Ensure rate is a float
        else:
            rate = batch_size / self.count  # This is already a float

        # Compute batch statistics
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, unbiased=False, keepdim=True)

        # Update mean
        delta_mean = batch_mean - self._mean_cont
        self._mean_cont += rate * delta_mean

        # Update variance
        self._var_cont += rate * (batch_var - self._var_cont + delta_mean * (batch_mean - self._mean_cont))
        self._std_cont = torch.sqrt(self._var_cont)


class ActorNetwork(nn.Module):
    def __init__(self, num_obs: int, num_actions: int):
        super().__init__()
        self.embedding = nn.Embedding(7, 5)  # Matches checkpoint's embedding.weight: [7, 5]

        # Adjusted to EXACTLY match checkpoint dimensions
        self.actor = nn.Sequential(
            nn.Linear(65, 256),       # actor.0.weight: [256, 65]
            nn.ELU(),
            nn.Linear(256, 256),      # actor.2.weight: [256, 256]
            nn.ELU(),
            nn.Linear(256, 128),      # actor.4.weight: [128, 256]
            nn.ELU(),
            nn.Linear(128, 64),       # actor.6.weight: [64, 128]
            nn.ELU(),
            nn.Linear(64, 7),         # actor.8.weight: [7, 64]
            nn.Softsign()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_categories = 7
        categorical_obs = torch.argmax(x[:, -num_categories:], dim=1).long()
        continuous_obs = x[:, :-num_categories]
        
        embedded = self.embedding(categorical_obs)
        combined = torch.cat([continuous_obs, embedded], dim=-1)
        return self.actor(combined)


class JointStateNode:
    def __init__(self):
        rospy.init_node('joint_state_node', anonymous=True)

        # Initialize variables
        self.dt = 1.0 / 200.0  # Time step of 200 Hz
        self.tennis_ball_pos_scale = 0.25
        self.lin_vel_scale = 0.15
        self.to_final_target_scale = 0.5
        self.dof_vel_scale = 0.31
        self.action_scale = 0.09  # Define scaling factor for actions
        self.to_throwing_pos_scale = 0.5
        self.to_throwing_vel_scale = 0.4
        self.moving_average = 0.8


        ####
        self.dummy_mode = True
        self.dummy_count = 0  # Added dummy counter
        ####

        self.robot_dof_lower_limits = torch.tensor(
            [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543]
        )
        self.robot_dof_upper_limits = torch.tensor(
            [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]
        )

        self.robot_dof_vel_limit = torch.tensor(
            [1.7130, 1.7130, 1.7450, 2.2680, 2.4430, 3.1420, 3.1420]
        )

        self.robot_dof_lower_limits_np = self.robot_dof_lower_limits.numpy()
        self.robot_dof_upper_limits_np = self.robot_dof_upper_limits.numpy()

        self.final_target_pos = np.array([-0.7, -0.16, 0.56], dtype=np.float32)
        self.throwing_pos = np.array([0.2, -0.35, 0.9], dtype=np.float32)
        self.throwing_vel = np.array([2.5, 0.0, 0.5], dtype=np.float32)
        
        print("22", time.time())
        self.load_observation_data()
        print("22", time.time())

        # Load the model and normalization
        print(time.time())
        self.model, self.obs_normalizer = self.load_model_and_normalization()
        print(time.time())
        # Initialize variables
        self.robot_dof_targets = None
        self.lock = Lock()
        self.t = 0

        # For saving outputs
        self.joint_targets = []
        self.joint_poses = []
        self.joint_velocities = []
        self.ee_poses = []
        self.ee_orientations = []
        self.ee_volocities = []


        ##########
        self.ee_pose = np.array([0.34217024, -0.3304696, 0.9004011], dtype=np.float32)  # Default home position
        self.ee_orientation = np.array([0.764173, -0.057726, 0.62555796, -0.14623435], dtype=np.float32)
        self.ee_vel = np.zeros(3, dtype=np.float32)
        self.robot_dof_targets = None
        self.data_ready = False

        self.action_history = np.zeros((2, 7))

        # Add initialization synchronization
        self.initialize_robot_state()
        self.wait_for_initial_data()
        ###########


        # self.ee_pose = torch.array([], dtype=torch.float32)
        # self.ee_vel = None
        # self.ee_orientation = None

        # self.action_logits = None  # Initialize action_logits here
        # self.pure_action = None

        # self.pure_action = torch.zeros(7, dtype=torch.float32)
        # self.action_logits = np.zeros(7, dtype=np.float32)
        # self.last_action = np.zeros(7, dtype=np.float32)

        self.joint_state_pub = rospy.Publisher("/joint_reference", JointState, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, tcp_nodelay=True)
        rospy.Subscriber("/ee_pose", Pose, self.ee_pose_callback)
        rospy.Subscriber("/ee_vel", Twist, self.ee_vel_callback)

        rospy.sleep(1)

        self.run()

    
    def initialize_robot_state(self):
        """Set initial joint positions from simulation configuration"""
        self.initial_joint_pos = np.array([
            0.7775,   # iiwa_joint_1
            0.01425,    # iiwa_joint_2
            1.3362,   # iiwa_joint_3
            1.2374,   # iiwa_joint_4
            1.5912,   # iiwa_joint_5
            -0.8966,  # iiwa_joint_6
            1.5541    # iiwa_joint_7
        ], dtype=np.float32)
        
        self.robot_dof_targets = torch.tensor(self.initial_joint_pos, dtype=torch.float32)
        self.pure_action = torch.zeros(7, dtype=torch.float32)
        self.action_logits = np.zeros(7, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)

    def wait_for_initial_data(self):
        """Wait for first valid sensor data with timeout"""
        timeout = rospy.Duration(10.0)  # 2-second timeout
        start_time = rospy.Time.now()
        
        rospy.loginfo("Waiting for initial sensor data...")
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start_time) > timeout:
                rospy.logerr("Failed to receive initial data within 2 seconds!")
                rospy.signal_shutdown("Initialization timeout")
                return

            if self.check_data_ready():
                self.data_ready = True
                rospy.loginfo("All initial data received!")
                return

            rospy.sleep(0.1)

    def check_data_ready(self):
        """Check if all required data has been received"""
        return all([
            self.ee_pose is not None,
            self.ee_orientation is not None,
            self.ee_vel is not None,
            self.robot_dof_targets is not None
        ])


    def load_observation_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(script_dir, '../../input_files/idealpd/ft_idealpd_9.csv')
        input_file_path = os.path.abspath(input_file_path)  # Convert to absolute path
        obs_data = pd.read_csv(input_file_path)

        self.tennisball_pos = obs_data.loc[:, ['tennisball_pos_0', 'tennisball_pos_1', 'tennisball_pos_2']].values
        self.tennisball_lin_vel = obs_data.loc[:, ['tennisball_lin_vel_0', 'tennisball_lin_vel_1', 'tennisball_lin_vel_2']].values
        self.obs_task = obs_data.loc[:, ['obs_task_0', 'obs_task_1', 'obs_task_2', 'obs_task_3', 'obs_task_4', 'obs_task_5', 'obs_task_6']].values
        self.to_final_target = obs_data.loc[:, ['to_final_target_0', 'to_final_target_1', 'to_final_target_2']].values

    def load_model_and_normalization(self):
        obs_space = 67  # Ensure this matches the training setup
        num_categories = 7
        num_actions = 7
        model = ActorNetwork(
            num_obs=obs_space,
            num_actions=num_actions
        )
        # Load model state
        checkpoint = torch.load('/home/user/ros_ws/src/catch_and_throw/input_files/idealpd/model_idealpd_9.pt', map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model_state_dict']
        actor_state_dict = {k: v for k, v in model_state_dict.items() if 'critic' not in k}
        model.load_state_dict(actor_state_dict, strict=False)
        model.eval()

        #########

        print("Network Structure:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")

        checkpoint_weight = actor_state_dict['actor.0.weight'][0,:5]
        current_weight = model.actor[0].weight[0,:5].detach()
        print(f"Checkpoint first weights: {checkpoint_weight}")
        print(f"Current first weights: {current_weight}")
        print(f"Match: {torch.allclose(checkpoint_weight, current_weight, atol=1e-5)}")


        checkpoint_shapes = {k: v.shape for k, v in actor_state_dict.items()}
        current_shapes = {k: v.shape for k, v in model.state_dict().items()}

        for key in checkpoint_shapes:
            if key not in current_shapes:
                print(f"Missing key: {key}")
            elif checkpoint_shapes[key] != current_shapes[key]:
                print(f"Shape mismatch: {key} | Checkpoint: {checkpoint_shapes[key]} vs Current: {current_shapes[key]}")
        ##############
                
        # Initialize normalizer with correct parameters
        obs_normalizer = EmpiricalNormalization(
            num_continuous=60,  # Was 67-7=60, but your output shows 56
            total_dim=67,
            until=1.0e8
        )
    
        # Load normalization state
        obs_normalizer_state_dict = checkpoint.get('obs_norm_state_dict', None)
        if obs_normalizer_state_dict is not None:
            obs_normalizer.load_state_dict(obs_normalizer_state_dict)
        print("Normalizer State Keys:", obs_normalizer.state_dict().keys())
        print("Checkpoint Normalizer Keys:", obs_normalizer_state_dict.keys())
        print("Normalizer mean:", obs_normalizer.mean_cont)
        print("Normalizer std:", obs_normalizer.std_cont)
        print("Checkpoint mean:", obs_normalizer_state_dict['_mean_cont'])
        print("Checkpoint std:", obs_normalizer_state_dict['_std_cont'])
        obs_normalizer.eval()
    
        return model, obs_normalizer

    def ee_pose_callback(self, data):
        with self.lock:
            self.ee_pose = np.array([data.position.x, data.position.y, data.position.z], dtype=np.float32)
            self.ee_orientation = np.array([data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z], dtype=np.float32)

    def ee_vel_callback(self, data):
        with self.lock:
            self.ee_vel = np.array([data.linear.x, data.linear.y, data.linear.z], dtype=np.float32)
            self.ee_angular_vel = np.array([data.angular.x, data.angular.y, data.angular.z], dtype=np.float32)

    def joint_states_callback(self, data):
        with self.lock:
            #####
            if self.dummy_mode:
                joint_positions_obs = np.array(data.position, dtype=np.float32).flatten()
                
                # Use initial positions as dummy targets
                self.robot_dof_targets = torch.tensor(self.initial_joint_pos, dtype=torch.float32)
                
                # Publish dummy target
                joint_state_msg = JointState()
                joint_state_msg.header = Header()
                joint_state_msg.header.stamp = rospy.Time.now()
                joint_state_msg.name = data.name
                joint_state_msg.position = self.robot_dof_targets.tolist()
                self.joint_state_pub.publish(joint_state_msg)
                
                print("self.dummy_count", self.dummy_count)
                self.dummy_count += 1
                if self.dummy_count >= 200:
                    self.dummy_mode = False
                    rospy.loginfo("Dummy commands completed. Switching to real control.")
                return  # Exit early during dummy phase
            ######

            if self.robot_dof_targets is None:
                self.robot_dof_targets = torch.tensor(data.position, dtype=torch.float32).unsqueeze(0)

            joint_positions_obs = np.array(data.position, dtype=np.float32).flatten()
            joint_velocities_obs = np.array(data.velocity, dtype=np.float32).flatten()

            # Scale joint position and velocity observations
            dof_pos_scaled_obs = (2.0 * (joint_positions_obs - self.robot_dof_lower_limits_np) /
                                (self.robot_dof_upper_limits_np - self.robot_dof_lower_limits_np) - 1.0).flatten()
            dof_vel_scaled_obs = (joint_velocities_obs * self.dof_vel_scale).flatten()

            ee_pos_obs = self.ee_pose.flatten() if self.ee_pose is not None else np.zeros(3, dtype=np.float32)
            ee_orientation_obs = self.ee_orientation.flatten() if self.ee_orientation is not None else np.zeros(4, dtype=np.float32)
            ee_vel_obs = self.ee_vel.flatten() if self.ee_vel is not None else np.zeros(3, dtype=np.float32)
            action = self.action_logits.flatten() if self.action_logits is not None else np.zeros(7, dtype=np.float32)
            pure_action = self.pure_action.cpu().numpy().flatten() if self.pure_action is not None else np.zeros(7, dtype=np.float32)
            last_action = action

            # Get current task from observation data
            current_task = self.obs_task[self.t, :]
            
            # Calculate conditions (matches training environment logic)
            condition_final_target = (current_task[3] == 1.0) or (current_task[4] == 1.0)  # Moving_to_target_pos or Stabilized_at_Target
            condition_throwing = (current_task[5] == 1.0)  # Throwing_Ball

            # Convert to tensors for consistent calculations
            final_target_pos_tensor = torch.from_numpy(self.final_target_pos).float()
            throwing_pos_tensor = torch.from_numpy(self.throwing_pos).float()
            throwing_vel_tensor = torch.from_numpy(self.throwing_vel).float()
            tennisball_pos_tensor = torch.from_numpy(self.tennisball_pos[self.t, :]).float()
            tennisball_vel_tensor = torch.from_numpy(self.tennisball_lin_vel[self.t, :]).float()

            # Calculate conditional targets
            to_final_target = torch.where(
                torch.tensor(condition_final_target).unsqueeze(-1),
                (final_target_pos_tensor - tennisball_pos_tensor),
                torch.zeros(3)
            ).numpy()

            to_throwing_vel = torch.where(
                torch.tensor(condition_throwing).unsqueeze(-1),
                (throwing_vel_tensor - tennisball_vel_tensor),
                torch.zeros(3)
            ).numpy()

            to_throwing_pos = torch.where(
                torch.tensor(condition_throwing).unsqueeze(-1),
                (throwing_pos_tensor - tennisball_pos_tensor),
                torch.zeros(3)
            ).numpy()

            if self.t >= len(self.tennisball_pos):
                rospy.signal_shutdown("Time index exceeded data length.")
                return

            tennisball_pos_obs = (self.tennisball_pos[self.t, :] * self.tennis_ball_pos_scale).flatten()
            tennisball_lin_vel_obs = (self.tennisball_lin_vel[self.t, :] * self.lin_vel_scale).flatten()
            ee_lin_vel_scaled = (ee_vel_obs * self.lin_vel_scale).flatten()
            
            # Apply scaling factors (matches training environment)
            to_final_target_obs = (to_final_target * self.to_final_target_scale).flatten()
            to_throwing_pos_scaled = (to_throwing_pos * self.to_throwing_pos_scale).flatten()
            to_throwing_vel_scaled = (to_throwing_vel * self.to_throwing_vel_scale).flatten()
            
            obs_task_obs = self.obs_task[self.t, :].flatten()

            self.action_history = np.roll(self.action_history, shift=-1, axis=0)
            self.action_history[-1:] = self.pure_action.cpu().numpy()
            
            # Compute the new offset point position 2cm along the z-axis of the end effector
            if self.ee_pose is not None and self.ee_orientation is not None:
                new_point_pos = self.compute_offset_point(
                    position=ee_pos_obs,
                    orientation=ee_orientation_obs,
                    offset_local=np.array([0, 0, 0.02])  # 2 cm along z-axis
                )
                # Use the offset point position instead of the end effector's position
                ee_pos_for_obs = new_point_pos

            else:
                # Handle cases where pose or orientation is not yet available
                ee_pos_for_obs = np.array(3, dtype=np.float32)
                new_point_pos = np.zeros(3, dtype=np.float32)

            # Concatenate all observations, replacing ee_pos_obs with ee_pos_for_obs
            observations = np.concatenate((
                pure_action,
                self.action_history.flatten(),
                dof_pos_scaled_obs,
                dof_vel_scaled_obs,
                tennisball_pos_obs,
                new_point_pos,  # Replaced with offset point
                tennisball_lin_vel_obs,
                ee_lin_vel_scaled,
                ee_orientation_obs,
                to_final_target_obs,
                to_throwing_pos_scaled,
                to_throwing_vel_scaled,
                obs_task_obs
            ))
            print("observations.shape", observations.shape)
            # print("pure_action:", pure_action)
            # print("self.action_history.flatten():", self.action_history.flatten())
            # print("dof_pos_scaled_obs:", dof_pos_scaled_obs)
            # print("dof_vel_scaled_obs:", dof_vel_scaled_obs)
            # print("tennisball_pos_obs:", tennisball_pos_obs)
            # print("new_point_pos:", new_point_pos)
            # print("tennisball_lin_vel_obs:", tennisball_lin_vel_obs)
            # print("ee_lin_vel_scaled:", ee_lin_vel_scaled)
            # print("ee_orientation_obs:", ee_orientation_obs)
            # print("to_final_target_obs:", to_final_target_obs)
            # print("to_throwing_pos_scaled:", to_throwing_pos_scaled)
            # print("to_throwing_vel_scaled:", to_throwing_vel_scaled)
            # print("obs_task_obs:", obs_task_obs)
            # print("##################################################")

            # Split observations into continuous and categorical parts
            num_categories = 7  # Number of categories in obs_task
            continuous_obs = observations[:-num_categories]
            obs_task_obs = observations[-num_categories:]

        # Apply normalization to the continuous observations
        with torch.no_grad():
            continuous_obs_tensor = torch.from_numpy(continuous_obs).float().unsqueeze(0)
            normalized_continuous_obs = self.obs_normalizer(continuous_obs_tensor)
            obs_task_tensor = torch.from_numpy(obs_task_obs).float().unsqueeze(0)
            # Reconstruct the observations tensor
            observations_tensor = torch.cat([normalized_continuous_obs, obs_task_tensor], dim=1)

            # Get action from the model
            action_mean = self.model(observations_tensor)
            self.pure_action = action_mean
            action_mean_np = action_mean.cpu().numpy().flatten()
            if last_action.shape != action_mean_np.shape:
                last_action = last_action.reshape(action_mean_np.shape)
            self.action_logits = self.moving_average * last_action + (1.0 - self.moving_average) * action_mean_np
            # print("self.action_logits", self.action_logits)
            # Apply joint velocity limits and calculate new targets
            joint_positions_obs_tensor = torch.tensor(joint_positions_obs, dtype=torch.float32)
            scaled_actions = np.squeeze(self.action_logits) * self.action_scale
            targets = joint_positions_obs_tensor + scaled_actions
            self.robot_dof_targets = torch.clamp(
                targets,
                0.96 * self.robot_dof_lower_limits,
                0.96 * self.robot_dof_upper_limits
            )
            robot_dof_targets_list = self.robot_dof_targets.tolist()

            
        # Append the output to the list for later saving
        self.joint_targets.append(self.robot_dof_targets.numpy())
        self.joint_poses.append(joint_positions_obs)
        self.joint_velocities.append(joint_velocities_obs)
        self.ee_poses.append(ee_pos_obs)  # Store the offset point position
        self.ee_orientations.append(ee_orientation_obs)
        self.ee_volocities.append(ee_vel_obs)

        # Publish joint state message
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = data.name
        joint_state_msg.position = robot_dof_targets_list

        self.joint_state_pub.publish(joint_state_msg)

        self.t += 1
        print(self.t)

    def save_output(self, outputs, output_file_path, header=None):
        outputs = np.array(outputs)
        np.savetxt(output_file_path, outputs, delimiter=',', header=header, comments='')
        print(f"Model outputs saved to {output_file_path}")

    def run(self):
        rate = rospy.Rate(200)  # 200 Hz
        while not rospy.is_shutdown():
            rate.sleep()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../../output_files/tm/idealpd_9')
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.save_output(self.joint_targets, os.path.join(output_dir, "tm_received_joint_target_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_poses, os.path.join(output_dir, "tm_received_joint_pos_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_velocities, os.path.join(output_dir, "tm_received_joint_vel_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.ee_poses, os.path.join(output_dir, "tm_received_ee_pos_np.csv"), "pos_X,pos_Y,pos_Z")
        self.save_output(self.ee_orientations, os.path.join(output_dir, "tm_received_ee_orientation_np.csv"), "or_w,or_x,or_y,or_z")
        self.save_output(self.ee_volocities, os.path.join(output_dir, "tm_received_ee_vel_np.csv"), "lin_vel_X,lin_vel_Y,lin_vel_Z")

    def compute_offset_point(self, position, orientation, offset_local=np.array([0, 0, 0.02])):
        """
        Compute the global position of a point offset from the end effector's position.

        Args:
            position (np.ndarray): The current position of the end effector (x, y, z).
            orientation (np.ndarray): The current orientation of the end effector as a quaternion (w, x, y, z).
            offset_local (np.ndarray): The local offset in the end effector's frame.

        Returns:
            np.ndarray: The global position of the offset point.
        """
        # Validate the quaternion
        norm = np.linalg.norm(orientation)
        if norm == 0:
            rospy.logwarn("Received zero quaternion. Using identity rotation.")
            rot_matrix = np.eye(3)
        else:
            normalized_quat = orientation / norm
            rot_matrix = tf_trans.quaternion_matrix(normalized_quat)[:3, :3]

        # Compute global offset
        global_offset = rot_matrix @ offset_local  # Matrix multiplication

        # Compute new point position
        new_point_pos = position + global_offset

        return new_point_pos

if __name__ == '__main__':
    try:
        JointStateNode()
    except rospy.ROSInterruptException:
        pass