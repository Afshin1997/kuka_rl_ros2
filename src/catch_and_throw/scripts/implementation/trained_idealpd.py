#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from threading import Lock

def get_activation(act_name):
    """Utility function to get the activation function by name."""
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if act_name in activations:
        return activations[act_name]
    else:
        raise ValueError(f"Invalid activation function: {act_name}")

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None, task_dim=5):
        super().__init__()
        self.eps = eps
        self.until = until
        self.task_dim = task_dim

        if isinstance(shape, int):
            main_dim = shape - task_dim
            assert main_dim > 0, "Shape must be larger than the task_dim."
            self.main_dim = main_dim
        else:
            main_dim = shape[-1] - task_dim
            assert main_dim > 0, "The last dimension must be larger than task_dim."
            self.main_dim = main_dim

        self.register_buffer("_mean", torch.zeros(self.main_dim).unsqueeze(0))
        self.register_buffer("_var", torch.ones(self.main_dim).unsqueeze(0))
        self.register_buffer("_std", torch.ones(self.main_dim).unsqueeze(0))

        self.count = 0

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        x_main = x[:, :-self.task_dim]
        x_task = x[:, -self.task_dim:]

        if self.training:
            self.update(x_main)

        x_main_norm = (x_main - self._mean) / (self._std + self.eps)

        # Convert task vector to proper one-hot
        task_indices = torch.argmax(x_task, dim=1)
        x_task_refined = torch.zeros_like(x_task)
        x_task_refined[torch.arange(x_task_refined.size(0)), task_indices] = 1.0

        return torch.cat([x_main_norm, x_task_refined], dim=1)

    def update(self, x_main):
        if self.until is not None and self.count >= self.until:
            return

        batch_count = x_main.shape[0]
        total_count = self.count + batch_count

        batch_mean = torch.mean(x_main, dim=0, keepdim=True)
        batch_var = torch.var(x_main, dim=0, unbiased=False, keepdim=True)

        if self.count == 0:
            new_mean = batch_mean
            new_var = batch_var
        else:
            delta = batch_mean - self._mean
            new_mean = self._mean + (batch_count / total_count) * delta
            new_var = (self._var * self.count + batch_var * batch_count +
                       (delta ** 2) * (self.count * batch_count / total_count)) / total_count

        self._mean = new_mean
        self._var = new_var
        self._std = torch.sqrt(self._var + self.eps)
        self.count = total_count

    def inverse(self, y):
        y_main = y[:, :-self.task_dim]
        y_task = y[:, -self.task_dim:]
        x_main = y_main * (self._std + self.eps) + self._mean
        return torch.cat([x_main, y_task], dim=1)

class Actor(nn.Module):
    """Actor network for generating actions."""

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        hidden_dims=[256, 128, 64, 64],
        activation="elu",
    ):
        super(Actor, self).__init__()
        activation_fn = get_activation(activation)

        layers = []
        layers.append(nn.Linear(num_actor_obs, hidden_dims[0]))
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dims[1], hidden_dims[2]))
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dims[2], num_actions))
        layers.append(nn.Softsign())

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class JointStateNode:
    def __init__(self):
        rospy.init_node('joint_state_node', anonymous=True)

        # Parameters
        self.dt = 1.0 / 250.0
        self.tennis_ball_pos_scale = 0.25
        self.lin_vel_scale = 0.15
        self.to_final_target_scale = 1.0
        self.dof_vel_scale = 0.31
        self.action_scale = 1.2

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

        self.final_target_pos = np.array([0.4, -0.4, 0.46], dtype=np.float32)

        # Load observation data
        self.load_observation_data()

        # Load actor and normalizer
        self.actor, self.obs_normalizer = self.load_actor_and_normalization()

        if self.actor is None or self.obs_normalizer is None:
            rospy.logerr("Actor or Normalizer failed to load. Shutting down node.")
            rospy.signal_shutdown("Actor or Normalizer failed to load.")
            return

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

        # For saving outputs
        self.joint_targets = []
        self.joint_poses = []
        self.joint_velocities = []
        self.ee_poses = []
        self.ee_orientations = []
        self.ee_volocities = []

        # Publishers
        self.joint_state_pub = rospy.Publisher("/joint_reference", JointState, queue_size=10)

        # Subscribers
        rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, queue_size=10)
        rospy.Subscriber("/ee_pose", Pose, self.ee_pose_callback, queue_size=10)
        rospy.Subscriber("/ee_vel", Twist, self.ee_vel_callback, queue_size=10)

        # On shutdown
        rospy.on_shutdown(self.shutdown_hook)

        rospy.sleep(1)

        self.run()

    def load_observation_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(script_dir, '../../input_files/idealpd/ft_idealpd_3.csv')
        input_file_path = os.path.abspath(input_file_path)
        if not os.path.exists(input_file_path):
            rospy.logerr(f"Observation data file not found: {input_file_path}")
            rospy.signal_shutdown("Missing observation data file.")
            return

        obs_data = pd.read_csv(input_file_path)
        self.tennisball_pos = obs_data[['tennisball_pos_0', 'tennisball_pos_1', 'tennisball_pos_2']].values
        self.tennisball_lin_vel = obs_data[['tennisball_lin_vel_0', 'tennisball_lin_vel_1', 'tennisball_lin_vel_2']].values
        self.obs_task = obs_data[['obs_task_0', 'obs_task_1', 'obs_task_2', 'obs_task_3', 'obs_task_4']].values
        self.to_final_target = obs_data[['to_final_target_0', 'to_final_target_1', 'to_final_target_2']].values

    def load_actor_and_normalization(self):
        num_actor_obs = 45
        num_actions = 7

        actor = Actor(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            activation="elu"
        )

        model_path = rospy.get_param('~model_path', default='/home/user/ros_ws/src/catch_and_throw/input_files/idealpd/model_idealpd_3.pt')
        if not os.path.exists(model_path):
            rospy.logerr(f"Model file not found: {model_path}")
            rospy.signal_shutdown("Missing model file.")
            return None, None

        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('actor.')}
            actor.load_state_dict(actor_state_dict, strict=False)
            actor.eval()
            rospy.loginfo("Actor model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Error loading actor model: {e}")
            rospy.signal_shutdown("Failed to load actor model.")
            return None, None

        obs_normalizer = EmpiricalNormalization(shape=45, task_dim=5)
        obs_normalizer_state_dict = checkpoint.get('obs_norm_state_dict', None)
        if obs_normalizer_state_dict is not None:
            try:
                obs_normalizer.load_state_dict(obs_normalizer_state_dict)
                obs_normalizer.eval()
                rospy.loginfo("Normalization state loaded successfully.")
            except Exception as e:
                rospy.logwarn(f"Failed to load normalization state: {e}")
        else:
            rospy.logwarn("No normalization state found in checkpoint.")

        return actor, obs_normalizer

    def joint_states_callback(self, data):
        with self.lock:
            self.joint_positions_obs = np.array(data.position, dtype=np.float32)
            self.joint_velocities_obs = np.array(data.velocity, dtype=np.float32)
            if self.robot_dof_targets is None:
                self.robot_dof_targets = torch.tensor(data.position, dtype=torch.float32).unsqueeze(0)

    def ee_pose_callback(self, data):
        with self.lock:
            self.ee_pose = np.array([data.position.x, data.position.y, data.position.z], dtype=np.float32)
            self.ee_orientation = np.array([data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z], dtype=np.float32)

    def ee_vel_callback(self, data):
        with self.lock:
            self.ee_vel = np.array([data.linear.x, data.linear.y, data.linear.z], dtype=np.float32)
            self.ee_angular_vel = np.array([data.angular.x, data.angular.y, data.angular.z], dtype=np.float32)

    def compute_action_and_publish(self):
        with self.lock:
            # Check if we have all data needed
            if self.joint_positions_obs is None or self.joint_velocities_obs is None:
                return
            if self.ee_pose is None or self.ee_orientation is None or self.ee_vel is None or self.ee_angular_vel is None:
                return
            if self.t >= len(self.tennisball_pos):
                rospy.logwarn("Time index exceeded data length. Shutting down node.")
                rospy.signal_shutdown("Time index exceeded data length.")
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

            action = self.action_logits.cpu().numpy().flatten() if self.action_logits is not None else np.ones(7, dtype=np.float32)

            observations = np.concatenate((
                action.flatten(),
                dof_pos_scaled_obs.flatten(),
                dof_vel_scaled_obs.flatten(),
                tennisball_pos_obs.flatten(),
                self.ee_pose.flatten(),
                tennisball_lin_vel_obs.flatten(),
                ee_lin_vel_scaled.flatten(),
                self.ee_orientation.flatten(),
                to_final_target_obs.flatten(),
                obs_task_obs.flatten()
            ))

            assert observations.size == 45, f"Observation size mismatch: expected 45, got {observations.size}"

            # Normalize and get action
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).float().unsqueeze(0)
                
                normalized_obs = self.obs_normalizer(observations_tensor)
                
                action_logits = self.actor(normalized_obs)
                self.action_logits = action_logits

                if self.t < 3:
                    print('without normalization', observations_tensor)
                    print('with normalization', normalized_obs)


                # Compute new targets
                joint_positions_obs_tensor = torch.tensor(self.joint_positions_obs, dtype=torch.float32)
                scaled_actions = action_logits.squeeze(0) * 18.0
                targets = joint_positions_obs_tensor + self.dt * scaled_actions
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
            joint_state_msg.header.stamp = rospy.Time.now()
            # Assuming joint names correspond to the robot
            # If you know the joint names of your robot, add them here:
            joint_state_msg.name = [f"joint_{i}" for i in range(7)]
            joint_state_msg.position = self.robot_dof_targets.squeeze(0).tolist()

            self.joint_state_pub.publish(joint_state_msg)

            self.t += 1

    def save_output(self, outputs, output_file_path, header=None):
        outputs = np.array(outputs)
        np.savetxt(output_file_path, outputs, delimiter=',', header=header, comments='')
        rospy.loginfo(f"Model outputs saved to {output_file_path}")

    def shutdown_hook(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '../../output_files/tm/idealpd_3')
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

        rospy.loginfo("Model outputs saved on shutdown.")

    def run(self):
        rate = rospy.Rate(250)  # 250 Hz
        while not rospy.is_shutdown():
            self.compute_action_and_publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        JointStateNode()
    except rospy.ROSInterruptException:
        pass