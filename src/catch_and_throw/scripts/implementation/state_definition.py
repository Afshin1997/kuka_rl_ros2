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
import os  # Import os module for path handling
import tf.transformations as tf_trans


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
        # Handle old checkpoint format where keys were different
        if "_mean_cont" in state_dict:
            state_dict["_mean"] = state_dict.pop("_mean_cont")
            state_dict["_var"] = state_dict.pop("_var_cont")
            state_dict["_std"] = state_dict.pop("_std_cont")

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
    """Actor network with embedding for categorical inputs."""

    def __init__(
        self,
        num_obs,
        num_actions,
        actor_hidden_dims=[256, 128, 64, 64],
        activation="elu",
    ):
        
        super().__init__()

        num_categories = 6        # Number of categories in obs_task
        embedding_dim = 4         # Dimension of the embedding for obs_task

        # Activation functions
        activation_fn = nn.ELU()
        activation_final = nn.Softsign()

        # Embedding for categorical input (obs_task)
        self.embedding = nn.Embedding(num_categories, embedding_dim)

        # Calculate new input dimensions by replacing one-hot with embedding
        adjusted_num_obs = num_obs - num_categories + embedding_dim

        # Policy Network (Actor)
        layers = []
        layers.append(nn.Linear(adjusted_num_obs, actor_hidden_dims[0]))
        layers.append(activation_fn)
        for i in range(1, len(actor_hidden_dims)):
            layers.append(nn.Linear(actor_hidden_dims[i-1], actor_hidden_dims[i]))
            layers.append(activation_fn)
        # Output layer
        layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        layers.append(activation_final)
        self.actor = nn.Sequential(*layers)

        print(f"Actor MLP: {self.actor}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean actions for inference (deterministic policy).

        Args:
            observations (torch.Tensor): Batch of observations.

        Returns:
            torch.Tensor: Mean actions.
        """
        num_categories = self.embedding.num_embeddings
        categorical_obs = torch.argmax(observations[:, -num_categories:], dim=1).long()
        continuous_obs = observations[:, :-num_categories]

        # Pass categorical observations through embedding
        embedded = self.embedding(categorical_obs)

        # Concatenate embedded vector with continuous observations
        combined = torch.cat([continuous_obs, embedded], dim=-1)

        actions_mean = self.actor(combined)
        return actions_mean


class JointStateNode:
    def __init__(self):
        rospy.init_node('joint_state_node', anonymous=True)

        ############

        # Initialize task-related variables
        self.task = torch.zeros(6)  # One-hot encoded task vector
        self.catched_ball = False
        self.stabilized_ball = False
        self.moving_to_target_pos = False
        self.stabilized_at_target = False
        self.throwing_ball = False
        self.stabilized_ball_counter = 0
        self.stabilized_ball_counter_at_target = 0
        self.revert_penalty = 0.0
        # Task indices
        self.catching_ball_index = 0
        self.stabilizing_ball_index = 1
        self.stabilized_ball_index = 2
        self.moving_to_target_pos_index = 3
        self.stabilized_at_target_index = 4
        self.throwing_ball_index = 5

        # One-hot encoding for tasks
        self.one_hot_encoding = torch.eye(6)

        ############

        # Initialize variables
        self.dt = 1.0 / 250.0  # Time step of 250 Hz
        self.tennis_ball_pos_scale = 0.25
        self.lin_vel_scale = 0.15
        self.to_final_target_scale = 1.0
        self.dof_vel_scale = 0.31
        self.action_scale = 0.07  # Define scaling factor for actions
        self.to_throwing_pos_scale = 0.5
        self.to_throwing_vel_scale = 0.15

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
        self.throwing_vel = np.array([7.7731, -0.1820, 0.0], dtype=np.float32)
        

        self.load_observation_data()

        # Load the model and normalization
        self.model, self.obs_normalizer = self.load_model_and_normalization()

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

        self.ee_pose = None
        self.ee_vel = None

        self.action_logits = None  # Initialize action_logits here

        self.joint_state_pub = rospy.Publisher("/joint_reference", JointState, queue_size=10)
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        rospy.Subscriber("/ee_pose", Pose, self.ee_pose_callback)
        rospy.Subscriber("/ee_vel", Twist, self.ee_vel_callback)

        rospy.sleep(1)

        self.run()

    def load_observation_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(script_dir, '../../input_files/idealpd/ft_idealpd_5.csv')
        input_file_path = os.path.abspath(input_file_path)  # Convert to absolute path
        obs_data = pd.read_csv(input_file_path)

        self.tennisball_pos = obs_data.loc[:, ['tennisball_pos_0', 'tennisball_pos_1', 'tennisball_pos_2']].values
        self.tennisball_lin_vel = obs_data.loc[:, ['tennisball_lin_vel_0', 'tennisball_lin_vel_1', 'tennisball_lin_vel_2']].values
        self.obs_task = obs_data.loc[:, ['obs_task_0', 'obs_task_1', 'obs_task_2', 'obs_task_3', 'obs_task_4', 'obs_task_5']].values
        self.to_final_target = obs_data.loc[:, ['to_final_target_0', 'to_final_target_1', 'to_final_target_2']].values

    def update_task_state(self):
        # Calculate distance between end effector and tennis ball
        distance = np.linalg.norm(self.ee_pose - self.tennisball_pos[self.t, :])

        # Calculate relative position termination condition
        normalized_end_effector_z = tf_trans.quaternion_multiply(
            self.ee_orientation, [0, 0, 1, 0]
        )  # Assuming z_unit_tensor is [0, 0, 1]
        relative_position_end_effector_ball = self.tennisball_pos[self.t, :] - self.ee_pose
        rel_position_termination = np.dot(normalized_end_effector_z[:3], relative_position_end_effector_ball)

        # Check if the ball is caught
        self.catched_ball = (distance < 0.12) and (rel_position_termination < 0.03 * 1.5)  # Assuming tennisball_radius is 0.03

        # Transition logic
        if self.catched_ball and np.argmax(self.task) == self.catching_ball_index:
            self.task = self.one_hot_encoding[self.stabilizing_ball_index]
        elif not self.catched_ball and np.argmax(self.task) == self.stabilizing_ball_index:
            self.task = self.one_hot_encoding[self.catching_ball_index]
            self.revert_penalty = -1.0

        # Conditions for 'Stabilized_Ball'
        distance_condition = (distance < 0.12) and (rel_position_termination < 0.03 * 1.2)
        velocity_condition = np.linalg.norm(self.tennisball_lin_vel[self.t, :]) < 0.1
        rel_velocity_condition = np.linalg.norm(self.tennisball_lin_vel[self.t, :] - self.ee_vel) < 0.15

        self.stabilized_ball = (
            distance_condition and velocity_condition and rel_velocity_condition and 
            (np.argmax(self.task) == self.stabilizing_ball_index or np.argmax(self.task) == self.stabilized_ball_index)
        )

        if self.stabilized_ball:
            self.task = self.one_hot_encoding[self.stabilized_ball_index]
            self.stabilized_ball_counter += 1
        elif not (distance_condition and velocity_condition and rel_velocity_condition) and np.argmax(self.task) == self.stabilized_ball_index:
            self.task = self.one_hot_encoding[self.stabilizing_ball_index]
            self.stabilized_ball_counter -= 1
            self.revert_penalty = -1.0

        # Transition to 'Moving_to_target_pos'
        rel_velocity_condition_to_target = np.linalg.norm(self.tennisball_lin_vel[self.t, :] - self.ee_vel) < 0.1
        self.moving_to_target_pos = (self.stabilized_ball_counter > 50) and distance_condition and rel_velocity_condition_to_target

        if self.moving_to_target_pos:
            self.task = self.one_hot_encoding[self.moving_to_target_pos_index]
        elif not self.moving_to_target_pos and np.argmax(self.task) == self.moving_to_target_pos_index:
            self.task = self.one_hot_encoding[self.stabilized_ball_index]
            self.revert_penalty = -1.0

        # Transition to 'Stabilized_at_Target'
        dist_to_target_pos = np.linalg.norm(self.final_target_pos - self.tennisball_pos[self.t, :])
        final_rel_dis_cond = (distance < 0.12) and (rel_position_termination < 0.03 * 1.2)
        final_vel_cond = np.linalg.norm(self.tennisball_lin_vel[self.t, :]) < 0.1
        final_rel_velocity_condition = np.linalg.norm(self.tennisball_lin_vel[self.t, :] - self.ee_vel) < 0.1
        final_target_dist_reward = dist_to_target_pos < 0.1

        self.stabilized_at_target = (
            final_rel_dis_cond and final_vel_cond and final_rel_velocity_condition and final_target_dist_reward and 
            (np.argmax(self.task) == self.moving_to_target_pos_index or np.argmax(self.task) == self.stabilized_at_target_index)
        )

        if self.stabilized_at_target:
            self.task = self.one_hot_encoding[self.stabilized_at_target_index]
            self.stabilized_ball_counter_at_target += 1
        elif not final_rel_dis_cond and np.argmax(self.task) == self.stabilized_at_target_index:
            self.task = self.one_hot_encoding[self.moving_to_target_pos_index]
            self.stabilized_ball_counter_at_target -= 1
            self.revert_penalty = -1.0

        # Transition to 'Throwing_Ball'
        self.throwing_ball = (self.stabilized_ball_counter_at_target > 50)

        if self.throwing_ball:
            self.task = self.one_hot_encoding[self.throwing_ball_index]

    def load_model_and_normalization(self):
        obs_space = 52  # Ensure this matches the training setup
        num_categories = 6
        num_actions = 7
        model = ActorNetwork(
            num_obs=obs_space,
            num_actions=num_actions,
            actor_hidden_dims=[256, 128, 64, 64],
            activation="elu",
        )
        # Load model state
        checkpoint = torch.load('/home/user/ros_ws/src/catch_and_throw/input_files/idealpd/model_ideal_pd_6s.pt', map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model_state_dict']
        actor_state_dict = {k: v for k, v in model_state_dict.items() if 'critic' not in k}
        model.load_state_dict(actor_state_dict, strict=False)
        model.eval()
    
        # Initialize normalizer with correct parameters
        obs_normalizer = EmpiricalNormalization(
            num_continuous=obs_space - num_categories,
            total_dim=obs_space,
            until=1.0e8
        )
    
        # Load normalization state
        obs_normalizer_state_dict = checkpoint.get('obs_norm_state_dict', None)
        if obs_normalizer_state_dict is not None:
            obs_normalizer.load_state_dict(obs_normalizer_state_dict)
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
            action = self.action_logits.cpu().numpy().flatten() if self.action_logits is not None else np.zeros(7, dtype=np.float32)

            to_final_target = (self.final_target_pos - self.tennisball_pos[self.t, :]).flatten()
            to_throwing_pos = (self.throwing_pos - self.tennisball_pos[self.t, :]).flatten()
            to_throwing_vel = (self.throwing_vel - self.tennisball_lin_vel[self.t, :]).flatten()

            if self.t >= len(self.tennisball_pos):
                rospy.signal_shutdown("Time index exceeded data length.")
                return

            tennisball_pos_obs = (self.tennisball_pos[self.t, :] * self.tennis_ball_pos_scale).flatten()
            tennisball_lin_vel_obs = (self.tennisball_lin_vel[self.t, :] * self.lin_vel_scale).flatten()
            ee_lin_vel_scaled = (ee_vel_obs * self.lin_vel_scale).flatten()
            to_final_target_obs = (to_final_target * self.to_final_target_scale).flatten()
            to_throwing_pos_scaled = (to_throwing_pos * self.to_throwing_pos_scale).flatten()
            to_throwing_vel_scaled = (to_throwing_vel * self.to_throwing_vel_scale).flatten()
            obs_task_obs = self.obs_task[self.t, :].flatten()

            # Compute the new offset point position 2cm along the z-axis of the end effector
            if self.ee_pose is not None and self.ee_orientation is not None:
                new_point_pos = self.compute_offset_point(
                    position=ee_pos_obs,
                    orientation=ee_orientation_obs,
                    offset_local=np.array([0, 0, 0.02])  # 2 cm along z-axis
                )
                # Use the offset point position instead of the end effector's position
                ee_pos_for_obs = new_point_pos
                # Optionally, publish the offset point (see Optional Step 4)
                # point_msg = Point()
                # point_msg.x, point_msg.y, point_msg.z = new_point_pos
                # self.offset_point_pub.publish(point_msg)
            else:
                # Handle cases where pose or orientation is not yet available
                ee_pos_for_obs = np.zeros(3, dtype=np.float32)
                new_point_pos = np.zeros(3, dtype=np.float32)

            # Append the new offset point position to the list
            # self.offset_points.append(new_point_pos)


            self.update_task_state()
            obs_task_obs = self.task.numpy()  # Use the current task state

            # Concatenate all observations, replacing ee_pos_obs with ee_pos_for_obs
            observations = np.concatenate((
                action,
                dof_pos_scaled_obs,
                dof_vel_scaled_obs,
                tennisball_pos_obs,
                ee_pos_for_obs,  # Replaced with offset point
                tennisball_lin_vel_obs,
                ee_lin_vel_scaled,
                ee_orientation_obs,
                to_final_target_obs,
                to_throwing_pos_scaled,
                to_throwing_vel_scaled,
                obs_task_obs
            ))
            print("action", action.shape)
            print("dof_pos_scaled_obs", dof_pos_scaled_obs.shape)
            print("dof_vel_scaled_obs", dof_vel_scaled_obs.shape)
            print("tennisball_pos_obs", tennisball_pos_obs.shape)
            print("ee_pos_for_obs", ee_pos_for_obs.shape)
            print("tennisball_lin_vel_obs", tennisball_lin_vel_obs.shape)
            print("ee_lin_vel_scaled", ee_lin_vel_scaled.shape)
            print("ee_orientation_obs", ee_orientation_obs.shape)
            print("to_final_target_obs", to_final_target_obs.shape)
            print("to_throwing_pos_scaled", to_throwing_pos_scaled.shape)
            print("to_throwing_vel_scaled", to_throwing_vel_scaled.shape)
            print("obs_task_obs", obs_task_obs.shape)

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
                self.action_logits = action_mean

                # Apply joint velocity limits and calculate new targets
                joint_positions_obs_tensor = torch.tensor(joint_positions_obs, dtype=torch.float32)
                scaled_actions = action_mean.squeeze(0) * self.action_scale
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
            self.ee_poses.append(ee_pos_for_obs)  # Store the offset point position
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


    def save_output(self, outputs, output_file_path, header=None):
        outputs = np.array(outputs)
        np.savetxt(output_file_path, outputs, delimiter=',', header=header, comments='')
        print(f"Model outputs saved to {output_file_path}")

    def run(self):
        rate = rospy.Rate(250)  # 250 Hz
        while not rospy.is_shutdown():
            rate.sleep()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../../output_files/tm/idealpd_6s')
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.save_output(self.joint_targets, os.path.join(output_dir, "tm_received_joint_target_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_poses, os.path.join(output_dir, "tm_received_joint_pos_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_velocities, os.path.join(output_dir, "tm_received_joint_vel_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.ee_poses, os.path.join(output_dir, "tm_received_ee_pos_np.csv"), "pos_X,pos_Y,pos_Z")
        self.save_output(self.ee_orientations, os.path.join(output_dir, "tm_received_ee_orientation_np.csv"), "or_w,or_x,or_y,or_z")
        self.save_output(self.ee_volocities, os.path.join(output_dir, "tm_received_ee_vel_np.csv"), "lin_vel_X,lin_vel_Y,lin_vel_Z")
        self.save_output(self.task.numpy(), os.path.join(output_dir, "tm_received_task_state_np.csv"), "task_0,task_1,task_2,task_3,task_4,task_5,task_5")

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