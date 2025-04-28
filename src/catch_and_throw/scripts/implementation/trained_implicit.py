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



class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class ActorNetwork(nn.Module):
    """Actor network with embedding for categorical inputs."""

    def __init__(
        self,
        num_obs,
        num_actions,
        actor_hidden_dims=[256, 128, 64],
        activation="elu",
    ):
        
        super().__init__()

        num_categories = 5        # Number of categories in obs_task
        embedding_dim = 6         # Dimension of the embedding for obs_task

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

        # Initialize variables
        self.dt = 1.0 / 250.0  # Time step of 250 Hz
        self.tennis_ball_pos_scale = 0.5
        self.lin_vel_scale = 0.2
        self.to_final_target_scale = 1.0
        self.dof_vel_scale = 0.31
        self.action_scale = 3.6  # Define scaling factor for actions

        self.robot_dof_lower_limits = torch.tensor(
            [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543]
        )
        self.robot_dof_upper_limits = torch.tensor(
            [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]
        )

        self.robot_dof_vel_limit = torch.tensor(
            [1.7104, 1.7104, 1.7453, 2.2689, 2.4435, 3.1416, 3.1416]
        )

        self.robot_dof_lower_limits_np = self.robot_dof_lower_limits.numpy()
        self.robot_dof_upper_limits_np = self.robot_dof_upper_limits.numpy()

        self.final_target_pos = np.array([0.4, -0.4, 0.46], dtype=np.float32)

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
        input_file_path = os.path.join(script_dir, '../../input_files/implicit/ft_implicit.csv')
        input_file_path = os.path.abspath(input_file_path)  # Convert to absolute path
        obs_data = pd.read_csv(input_file_path)

        self.tennisball_pos = obs_data.loc[:, ['tennisball_pos_0', 'tennisball_pos_1', 'tennisball_pos_2']].values
        self.tennisball_lin_vel = obs_data.loc[:, ['tennisball_lin_vel_0', 'tennisball_lin_vel_1', 'tennisball_lin_vel_2']].values
        self.obs_task = obs_data.loc[:, ['obs_task_0', 'obs_task_1', 'obs_task_2', 'obs_task_3', 'obs_task_4']].values
        self.to_final_target = obs_data.loc[:, ['to_final_target_0', 'to_final_target_1', 'to_final_target_2']].values

    def load_model_and_normalization(self):
        obs_space = 45  # Updated observation space size
        num_categories = 5
        num_actions = 7
        model = ActorNetwork(
            num_obs=obs_space,
            num_actions=num_actions,
            actor_hidden_dims=[256, 128, 64],
            activation="elu",
        )
        # Load model state
        checkpoint = torch.load('/home/user/ros_ws/src/python_communication/input_files/implicit/model_implicit.pt', map_location=torch.device('cpu'))
        model_state_dict = checkpoint['model_state_dict']
        # Filter out critic parameters
        actor_state_dict = {k: v for k, v in model_state_dict.items() if 'critic' not in k}
        model.load_state_dict(actor_state_dict, strict=False)
        model.eval()
        # Load normalization state
        obs_normalizer = EmpiricalNormalization(shape=[obs_space - num_categories])
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

            if self.t >= len(self.tennisball_pos):
                rospy.signal_shutdown("Time index exceeded data length.")
                return

            tennisball_pos_obs = (self.tennisball_pos[self.t, :] * self.tennis_ball_pos_scale).flatten()
            tennisball_lin_vel_obs = (self.tennisball_lin_vel[self.t, :] * self.lin_vel_scale).flatten()
            ee_lin_vel_scaled = (ee_vel_obs * self.lin_vel_scale).flatten()
            to_final_target_obs = (self.to_final_target[self.t, :] * self.to_final_target_scale).flatten()
            obs_task_obs = self.obs_task[self.t, :].flatten()

            # Concatenate all observations
            observations = np.concatenate((
                action,
                dof_pos_scaled_obs,
                dof_vel_scaled_obs,
                tennisball_pos_obs,
                ee_pos_obs,
                tennisball_lin_vel_obs,
                ee_lin_vel_scaled,
                ee_orientation_obs,
                to_final_target_obs,
                obs_task_obs
            ))

            # Split observations into continuous and categorical parts
            num_categories = 5  # Number of categories in obs_task
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
                scaled_actions = action_mean.squeeze(0) * self.robot_dof_vel_limit * self.action_scale
                targets = joint_positions_obs_tensor + self.dt * scaled_actions
                self.robot_dof_targets = torch.clamp(
                    targets,
                    self.robot_dof_lower_limits,
                    self.robot_dof_upper_limits
                )
                robot_dof_targets_list = self.robot_dof_targets.tolist()

            # Append the output to the list for later saving
            self.joint_targets.append(self.robot_dof_targets.numpy())
            self.joint_poses.append(joint_positions_obs)
            self.joint_velocities.append(joint_velocities_obs)
            self.ee_poses.append(ee_pos_obs)
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
        output_dir = os.path.join(script_dir, '../../output_files/tm/implicit')
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.save_output(self.joint_targets, os.path.join(output_dir, "tm_received_joint_target_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_poses, os.path.join(output_dir, "tm_received_joint_pos_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.joint_velocities, os.path.join(output_dir, "tm_received_joint_vel_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
        self.save_output(self.ee_poses, os.path.join(output_dir, "tm_received_ee_pos_np.csv"), "pos_X,pos_Y,pos_Z")
        self.save_output(self.ee_orientations, os.path.join(output_dir, "tm_received_ee_orientation_np.csv"), "or_w,or_x,or_y,or_z")
        self.save_output(self.ee_volocities, os.path.join(output_dir, "tm_received_ee_vel_np.csv"), "lin_vel_X,lin_vel_Y,lin_vel_Z")


if __name__ == '__main__':
    try:
        JointStateNode()
    except rospy.ROSInterruptException:
        pass
