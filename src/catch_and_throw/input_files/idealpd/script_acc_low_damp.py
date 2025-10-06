# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
import numpy as np
import torch.nn.functional as F
from functools import reduce
import csv
import torch.nn as nn

# from scipy.signal import butter

from omni.isaac.lab.utils.math import quat_apply
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.sim as sim_utils
from .actuators import IdealPDActuatorCfg, DelayedPDActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, quat_unique
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from .noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfgObs, NoiseModelWithAdditiveBiasCfgAction


num_record_envs = 5
num_observations = 59
data_buffers = [[] for _ in range(num_record_envs)]
done_flags = [False] * num_record_envs
headers = []
headers += [f'torque{i}' for i in range(7)]
headers += [f'action{i}' for i in range(7)]
headers += [f'set_target_{i}' for i in range(7)]
headers += [f'joint_pos_{i}' for i in range(7)]
headers += [f'joint_vel_{i}' for i in range(7)]
# headers += [f'joint_vel_disc_{i}' for i in range(7)]
# headers += [f'to_target_pos{i}' for i in range(3)]
# headers += [f'to_target_vel{i}' for i in range(3)]
# headers += [f'ee_orien{i}' for i in range(4)]
headers += [f'tennisball_pos_{i}' for i in range(3)]
headers += [f'tennisball_lin_vel_{i}' for i in range(3)]
# headers += [f'to_target_{i}' for i in range(3)]
headers += [f'end_effector_pos_{i}' for i in range(3)]
headers += [f'end_effector_rot_{i}' for i in range(4)]
headers += [f'end_effector_lin_vel_{i}' for i in range(3)]
headers += [f'obs_task_{i}' for i in range(7)]
headers += [f'to_final_target_{i}' for i in range(3)]
headers += [f'joint_acc_{i}' for i in range(7)]
# headers += [f'joint_acc_disc_{i}' for i in range(7)]
# headers += [f'joint_jerk_{i}' for i in range(7)]


@configclass
class EventCfg:
    """Configuration for randomization."""

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tennisball"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -0.1], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "stiffness_distribution_params": (0.95, 1.05),
          "damping_distribution_params": (0.95, 1.05),
          "operation": "scale",
          "distribution": "log_uniform",
      },
    )

    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


    end_effector_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="iiwa_link_7"),
            "static_friction_range": (0.5, 0.7),      # End effector friction range
            "dynamic_friction_range": (0.45, 0.65),     # End effector friction range  
            "restitution_range": (0.55, 0.75),          # End effector restitution (lower for grip)
            "num_buckets": 250,
        },
    )

    # Randomize tennis ball material properties  
    tennis_ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("tennisball", body_names=".*"),
            "static_friction_range": (0.4, 0.7),      # Tennis ball friction range
            "dynamic_friction_range": (0.35, 0.65),     # Tennis ball friction range
            "restitution_range": (0.55, 0.75),          # Tennis ball restitution (higher for bounce)
            "num_buckets": 250,
        },
    )

@configclass
class Lbr_iiwaSixStatesEnvCfg(DirectRLEnvCfg):

    # env
    episode_length_s = 40  # 500 timesteps
    decimation = 2
    action_space = 7
    observation_space = 44 ##41+12
    state_space = 0

    ###
    viewer = ViewerCfg(eye=(10.0, 10.0, 10.0), lookat=(0.0, 0.0, 0.0))

    action_scale = 7.5
    dof_velocity_scale = 0.31
    dof_acc_scale = 0.01
    dof_jerk_scale = 1e-6
    dof_torque_scale = 8e-3      
    tennis_ball_pos_scale = 0.25
    lin_vel_scale = 0.15
    lin_acc_scale = 0.05
    to_final_target_scale = 0.5
    to_throwing_pos_scale = 0.5
    to_throwing_vel_scale = 0.4
    

    act_moving_average = 0.7

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.7,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=7.0, replicate_physics=True)

    # domain randomization config
    events: EventCfg = EventCfg()

    tennisball_radius = 0.031
    stiffness_f = 700
    damping_f = 18

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/prisma-lab-ws/Scrivania/Omniverse/IsaacLab/source/IsaacSimPrismaLab/Lbr_iiwa/assets/lbr_iiwa7_3/lbr_description/urdf/iiwa7/lbr_iiwa7_322.usd",

            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                enable_gyroscopic_forces=True,
                max_depenetration_velocity=100.0,
                stabilization_threshold=0.001,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=8, sleep_threshold=0.005, stabilization_threshold=0.001,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(

            # joint_pos={
            #     "iiwa_joint_1": 45 * math.pi / 180,
            #     "iiwa_joint_2": 0 * math.pi / 180,
            #     "iiwa_joint_3": 76 * math.pi / 180,
            #     "iiwa_joint_4": 70 * math.pi / 180,
            #     "iiwa_joint_5": 92 * math.pi / 180,
            #     "iiwa_joint_6": -52 * math.pi / 180,
            #     "iiwa_joint_7": 90 * math.pi / 180,
            # },

            joint_pos={
                "iiwa_joint_1": 102 * math.pi / 180,
                "iiwa_joint_2": -61 * math.pi / 180,
                "iiwa_joint_3": -104 * math.pi / 180,
                "iiwa_joint_4": -64 * math.pi / 180,
                "iiwa_joint_5": 111 * math.pi / 180,
                "iiwa_joint_6": 64 * math.pi / 180,
                "iiwa_joint_7": -52.5 * math.pi / 180,
            },

            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_vel={".*": 0.0},
        ), 
        actuators={
            "A1": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_1"],
                velocity_limit=0.98 * 98.0 * math.pi / 180,  # ≈ 1.713 rad/s
                effort_limit=176.0,
                stiffness=625,
                damping=35,
                # min_delay=0,
                # max_delay=2,
            ),
            "A2": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_2"],
                velocity_limit=0.98 * 98.0 * math.pi / 180,  # ≈ 1.713 rad/s
                effort_limit=176.0,
                stiffness=625,
                damping=35,
                # min_delay=0,
                # max_delay=2,
            ),
            "A3": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_3"],
                velocity_limit=0.98 * 100.0 * math.pi / 180,  # ≈ 1.745 rad/s
                effort_limit=110.0,
                stiffness=625,
                damping=35,
                # min_delay=0,
                # max_delay=2,
            ),
            "A4": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_4"],
                velocity_limit=0.98 * 130.0 * math.pi / 180,  # ≈ 2.268 rad/s
                effort_limit=110.0,
                stiffness=625,
                damping=35,
                # min_delay=0,
                # max_delay=2,
            ),
            "A5": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_5"],
                velocity_limit=0.98 * 140.0 * math.pi / 180,  # ≈ 2.443 rad/s
                effort_limit=110.0,
                stiffness=400,
                damping=28,
                # min_delay=0,
                # max_delay=2,
            ),
            "A6": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_6"],
                velocity_limit=0.98 * 180.0 * math.pi / 180,  # ≈ 3.142 rad/s
                effort_limit=40.0,
                stiffness=100,
                damping=14,
                # min_delay=0,
                # max_delay=2,
            ),
            "A7": IdealPDActuatorCfg(
                joint_names_expr=["iiwa_joint_7"],
                velocity_limit=0.98 * 180.0 * math.pi / 180,  # ≈ 3.142 rad/s
                effort_limit=40.0,
                stiffness=100,
                damping=14,
                # min_delay=0,
                # max_delay=2,
            ),
        },
        
    )

    tennisball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/tennisball",
        spawn=sim_utils.SphereCfg(
            radius=tennisball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,  # Increased for better collision accuracy
                solver_velocity_iteration_count=16,  # Increased for better collision accuracy
                sleep_threshold=0.0008,  # Reduced - keeps ball active longer
                stabilization_threshold=0.0008,  # Reduced - less energy dissipation
                max_depenetration_velocity=8000.0,  # Increased for faster separation
                linear_damping=0.015,  # 0.083
                angular_damping=0.01, #0.033
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.052),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,  # Slightly increased for better contact detection
                rest_offset=0.0005,
                torsional_patch_radius=0.015,
                min_torsional_patch_radius=0.006
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.7,  # Reduced - less energy loss from friction
                dynamic_friction=0.6,  # Reduced - less energy loss from friction
                restitution=0.7,  # Increased to near-perfect bounce
                improve_patch_friction=True,  # Disabled - can reduce bouncing
                friction_combine_mode="multiply",  # Changed to reduce overall friction
                restitution_combine_mode="average",  # Changed to use highest restitution value
                compliant_contact_stiffness=1e6,  # Keep at 0 for rigid contacts
                compliant_contact_damping=1e3,  # Keep at 0 to avoid energy dissipation
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, -0.45, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(-3.5, 0.0, 5.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.5,
        ),
    )
    tasks = ['Catching_Ball', 'Stabilizing_Ball', 'Stabilized_Ball', 'Moving_to_target_pos', 'Stabilized_at_Target', 'Throwing_Ball', 'Stop']

    # reward scales
    dist_reward_scale = 1.2
    orient_reward_scale = 0.55
    stability_reward_scale = 2.5
    dist_to_target_pos_reward_scale = 2.0
    dof_at_limit_cost_scale = 0.2
    dof_vel_lim_penalty_scale = 1.0
    joint_accel_scale = 5.0e-7
    joint_jerk_scale = 8.0e-11
    joint_torque_scale = 8.0e-7
    to_throwing_pos_reward_scale = 2.5
    relative_velocity_reward_scale = 2.0
    action_sign_change_penalty_scale = 0.02
    joint_vel_penalty_scale = 5.0
    action_penalty_scale = 0.001

    action_noise_model: NoiseModelWithAdditiveBiasCfgAction = NoiseModelWithAdditiveBiasCfgAction(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
    )

    observation_noise_model: NoiseModelWithAdditiveBiasCfgObs = NoiseModelWithAdditiveBiasCfgObs(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0015, operation="abs"),
    )

class Lbr_iiwaSixStatesEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Lbr_iiwaSixStatesEnvCfg

    def __init__(self, cfg: Lbr_iiwaSixStatesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_vel_limit = torch.tensor([1.713, 1.713, 1.745, 2.268, 2.443, 3.142, 3.142], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.effort_limit = torch.tensor([176, 176, 110, 110, 110, 40, 40], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.robot_dof_targets = torch.tensor([102 * math.pi / 180, -61 * math.pi / 180, -104 * math.pi / 180,
                                        -64 * math.pi / 180, 111 * math.pi / 180, 64 * math.pi / 180,
                                        -52.5 * math.pi / 180], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        self.stiffness = torch.tensor([625, 625, 625, 625, 400, 100, 100], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.damping = torch.tensor([35, 35, 35, 35, 28, 14, 14], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        self.robot_dof_vel_targets = torch.zeros((self.num_envs, 7), device=self.device)
        # self.torque_target = torch.zeros((self.num_envs, 7), device=self.device)

        self.prev_joint_acc = torch.zeros((self.num_envs, 7), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 7), device=self.device)
        
        self.prev_joint_pos = torch.tensor([102 * math.pi / 180, -61 * math.pi / 180, -104 * math.pi / 180,
                                        -64 * math.pi / 180, 111 * math.pi / 180, 64 * math.pi / 180,
                                        -52.5 * math.pi / 180], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.prev_torque = torch.zeros((self.num_envs, 7), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, 7), device=self.device)
        
        self.end_effector_idx = self._robot.find_bodies("ee_surface")[0][0]

        self.tennisball_idx = self._tennisball.find_bodies("tennisball")[0][0]

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # defining tasks
        num_tasks = len(self.cfg.tasks)
        self.one_hot_encoding = torch.tensor(torch.eye(num_tasks), dtype=torch.float, device=self.device)
        task_catching_ball_index = self.cfg.tasks.index('Catching_Ball')
        self.task = torch.tensor(self.one_hot_encoding[task_catching_ball_index], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # define a counter for stabilization
        self.stabilized_ball_counter = torch.zeros((self.num_envs), device=self.device)
        self.stabilized_ball_counter_at_target = torch.zeros((self.num_envs), device=self.device)

        # define the acceleration matrix
        self.last_ee_vel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.saving_cond_counter = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.terminated_penalty = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.truncated_test = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)

        # define the final targets of revolute joints after stabilization
        self.final_target_pos = torch.tensor([-0.65, -0.4, 0.55], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.throwing_vel = torch.ones((self.num_envs, 3), device=self.device)
        self.throwing_vel_norm = torch.ones((self.num_envs, 3), device=self.device)
        self.throwing_pos = torch.tensor([0.2, -0.35, 0.9], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.finished_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bouncing_happens = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.pure_actions = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.target_term = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        self.termination_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.throwing_termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_history_actor = torch.zeros((self.num_envs, 2, 7),device=self.device)
        self.action_history_critic = torch.zeros((self.num_envs, 5, 7),device=self.device)

        self.curriculum_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.max_curriculum_level = 7000  # After this, threshold is fixed
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._tennisball = RigidObject(self.cfg.tennisball)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["tennisball"] = self._tennisball

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):

        self.actions = actions.clone()
        # print("self.actions", self.actions[0])
        self.pure_actions = actions.clone()
        self.action_history_critic = torch.roll(self.action_history_critic, shifts=-1, dims=1)
        self.action_history_critic[:, -1] = actions.clone()

        self.action_history_actor = torch.roll(self.action_history_actor, shifts=-1, dims=1)
        self.action_history_actor[:, -1] = actions.clone()
        targets = self.robot_dof_pos + self.actions * 0.1
        self.target_term = targets
        self.robot_dof_targets = torch.clamp(targets, 0.96 * self.robot_dof_lower_limits, 0.96 * self.robot_dof_upper_limits)
        pos_error = self.robot_dof_targets - self.robot_dof_pos
        vel_error = -self.robot_dof_vel  # desired velocity is zero
        des_acc = self.stiffness * pos_error + self.damping * vel_error

        mass_matrix = self._robot.root_physx_view.get_mass_matrices()
        gravity_forces = self._robot.root_physx_view.get_generalized_gravity_forces()
        coriolis_centrifugal = self._robot.root_physx_view.get_coriolis_and_centrifugal_forces()
        torques = torch.bmm(mass_matrix, des_acc.unsqueeze(-1)).squeeze(-1) + gravity_forces + coriolis_centrifugal
        self.torque_target = torch.clip(torques, min=-self.effort_limit, max=self.effort_limit)

    def _apply_action(self):
        self._robot.set_joint_effort_target(self.torque_target)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        conditions = [
            torch.any((self.dof_pos_scaled.abs() > 0.98), dim=-1),
            torch.any((self.robot_dof_vel.abs() > 0.98 * self.robot_dof_vel_limit), dim=-1),
            torch.any((self.pure_actions.abs() > 0.94), dim=-1),
            self.end_effector_pos[:, 2] < 0.25,
            # self.tennisball_pos[:, 2] < 0.25,
            torch.logical_and(self.rel_position_termination < 0, self.tennisball_pos[:, 0] < 2.0),
            torch.logical_and(self.tennisball_pos[:, 2] - self.end_effector_pos[:, 2] < -0.1, torch.logical_and(self.tennisball_pos[:, 0] < 1.0, self.task[:, -1]==0))
            # self.bouncing_happens
        ]
        terminated = reduce(torch.logical_or, conditions)
        self.terminated_penalty = terminated
        truncated = (self.episode_length_buf >= self.max_episode_length - 1) | (self.finished_episode) | torch.logical_and(self.tennisball_pos[:, 0] > 2.0, self.task[:, -1]==1)
        self.truncated_test = truncated

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:

        return self._compute_rewards(
            self.actions,
            self.tennisball_pos,
            self.tennisball_lin_vel,
            self.end_effector_pos,
            self.end_effector_rot,
            self.end_effector_lin_vel,
            self.cfg.dist_reward_scale,
            self.cfg.orient_reward_scale,
            self.cfg.stability_reward_scale,
            self.cfg.dist_to_target_pos_reward_scale,
            self.cfg.dof_at_limit_cost_scale,
            self.cfg.dof_vel_lim_penalty_scale,
            self.cfg.joint_accel_scale,
            self.cfg.joint_jerk_scale,
            self.cfg.joint_torque_scale,
            self.cfg.to_throwing_pos_reward_scale,
            self.cfg.relative_velocity_reward_scale,
            self.cfg.action_sign_change_penalty_scale,
            self.cfg.joint_vel_penalty_scale,
            self.cfg.action_penalty_scale,
            self.task,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
      

        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.06,
            0.06,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )

        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot_dof_targets[env_ids] = joint_pos
        self.prev_joint_pos[env_ids] = joint_pos
        
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_vel_targets[env_ids] = joint_vel
        
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # tennisball state
        self.tennisball_default_state = self._tennisball.data.default_root_state.clone()[env_ids]

        self.init_ball_state = self._tennisball.data.default_root_state.clone()[env_ids, :3]

        # estimated_catching_pos = torch.tensor([0.51, -0.45, 0.65], dtype=torch.float, device=self.device).repeat(len(env_ids), 1)

        random_ball_states = self.random_throw(self.init_ball_state)

        self.tennisball_default_state[:, 0:3] = random_ball_states[:, :3] + self.scene.env_origins[env_ids]
        self.tennisball_default_state[:, 7:10] = random_ball_states[:, 3:]
        self.tennisball_default_state[:, 10:] = (torch.rand_like(self.init_ball_state) * 2 - 1) * 0.2

        self._tennisball.write_root_state_to_sim(self.tennisball_default_state, env_ids)

        throwing_pos_init = torch.tensor([0.2, -0.35, 0.9], dtype=torch.float, device=self.device).repeat(len(env_ids), 1)
        throwing_pos_final = self.tennisball_default_state[:, 0:3] - self.scene.env_origins[env_ids]
        throwing_pos_final[:, 2] = -1.0       
        self.throwing_vel[env_ids] = self.random_throw_back(init_state=throwing_pos_init, final_state=throwing_pos_final)
        self.throwing_vel_norm[env_ids] = F.normalize(self.throwing_vel[env_ids], p=2, dim=-1)
        # Reset task state
        task_catching_ball_index = self.cfg.tasks.index('Catching_Ball')
        self.task[env_ids] = self.one_hot_encoding[task_catching_ball_index]

        # reset the stabilized counter
        self.stabilized_ball_counter[env_ids] = torch.zeros_like(env_ids, dtype=torch.float, device=self.device)
        self.stabilized_ball_counter_at_target[env_ids] = torch.zeros_like(env_ids, dtype=torch.float, device=self.device)

        # reset the accelerations
        self.last_ee_vel[env_ids] = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)

        self.last_actions[env_ids] = torch.zeros((len(env_ids), 7), dtype=torch.float, device=self.device)
        self.pure_actions[env_ids] = torch.zeros((len(env_ids), 7), dtype=torch.float, device=self.device)
        
        self.prev_joint_acc[env_ids] = torch.zeros_like(joint_pos)
        self.prev_joint_vel[env_ids] = torch.zeros_like(joint_pos)
        self.prev_torque[env_ids] = torch.zeros_like(joint_pos)

        self.termination_state[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        self.throwing_termination[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
       
        self.action_history_actor[env_ids] = torch.zeros((len(env_ids), 2, 7),device=self.device)
        self.action_history_critic[env_ids] = torch.zeros((len(env_ids), 5, 7),device=self.device)

        self.target_term[env_ids] = torch.zeros((len(env_ids), 7), dtype=torch.float, device=self.device)

        self.finished_episode[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        self.bouncing_happens[env_ids] = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        # Increment curriculum level for reset environments
        self.curriculum_level[env_ids] += 1
        self.curriculum_level[env_ids] = torch.clamp(self.curriculum_level[env_ids], max=self.max_curriculum_level)


        self._compute_intermediate_values(env_ids)    

    def _get_observations(self) -> dict:

        self.to_final_target = self.final_target_pos - self.tennisball_pos
        self.to_throwing_vel = self.throwing_vel - self.tennisball_lin_vel
        self.to_throwing_pos = self.throwing_pos - self.tennisball_pos

        obs_task = self.task
        self.joint_vel = (self.robot_dof_pos - self.prev_joint_pos)/self.dt
        self.joint_acc = (self.joint_vel - self.prev_joint_vel)/self.dt
        self.prev_joint_vel = self.joint_vel
        self.prev_joint_pos = self.robot_dof_pos

        # # ############################################################################################
        # row_counter = {env: 0 for env in range(num_record_envs)}
        # max_rows_to_save = 2000  # Set the limit to 50 rows

        # data_set = torch.cat(
        #     (   self._robot.data.applied_torque,
        #         self.actions,  # 7
        #         self.robot_dof_targets,  # 7
        #         self._robot.data.joint_pos,  # 7
        #         self._robot.data.joint_vel,  # 7
        #         # self.joint_vel,
        #         self.tennisball_pos,  # 3
        #         self.tennisball_lin_vel,  # 3
        #         self.end_effector_pos,  # 3
        #         self.end_effector_rot,  # 3
        #         self.end_effector_lin_vel,  # 4
        #         obs_task,  # 5
        #         self.to_final_target,  # 3
        #         self._robot.data.joint_acc,
        #         # self.joint_acc,
        #     ),
        #     dim=-1,
        # )

        # for env in range(num_record_envs):
        #     # Check if the row count for this environment is less than the maximum allowed rows
        #     if row_counter[env] < max_rows_to_save:

        #         data_per_env = data_set[env].cpu().numpy()
        #         data_buffers[env].append(data_per_env)
        #         row_counter[env] += 1  # Increment the row counter for this environment

        # # Save the data to CSV files
        # for env in range(num_record_envs):
        #     filename = f'env_{env}_data.csv'
        #     data_array = np.array(data_buffers[env])  # Convert list of observations to NumPy array
        #     with open(filename, 'w', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow(headers)
        #         writer.writerows(data_array)

        # #######################################################################################################
        
        obs_actor = torch.cat(
            (   self.action_history_actor.flatten(start_dim=1), #14
                self.dof_pos_scaled, # 7
                (self.joint_vel + (torch.rand_like(self.joint_vel) * 2 - 1) * 0.01) * self.cfg.dof_velocity_scale,
                # self.joint_vel * self.cfg.dof_velocity_scale, # 7
                (self.tennisball_pos + (torch.rand_like(self.tennisball_pos) * 2 - 1) * 0.005) * self.cfg.tennis_ball_pos_scale, # 3
                # (self.tennisball_pos_history + torch.rand_like(self.tennisball_pos_history) * 0.002).flatten(start_dim=1) * self.cfg.tennis_ball_pos_scale, ##9
                (self.tennisball_lin_vel + (torch.rand_like(self.tennisball_lin_vel) * 2 - 1) * 0.01) * self.cfg.lin_vel_scale,
                # (self.tennisball_vel_history + torch.rand_like(self.tennisball_vel_history) * 0.01).flatten(start_dim=1) * self.cfg.lin_vel_scale, ##9
                # self.tennisball_lin_vel * self.cfg.lin_vel_scale, #3
                (self.end_effector_pos + (torch.rand_like(self.end_effector_pos) * 2 - 1) * 0.005),
                # self.end_effector_lin_vel * self.cfg.lin_vel_scale,
                (self.end_effector_lin_vel + (torch.rand_like(self.end_effector_lin_vel) * 2 - 1) * 0.005) * self.cfg.lin_vel_scale,
                self.end_effector_rot,
            ),
            dim=-1,
        )
        obs_critic = torch.cat(
            (   self.pure_actions,
                self._robot.data.applied_torque * self.cfg.dof_torque_scale,
                self.action_history_critic.flatten(start_dim=1),
                self.dof_pos_scaled, # 7
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self._robot.data.joint_acc * self.cfg.dof_acc_scale,
                self.tennisball_pos * self.cfg.tennis_ball_pos_scale, # 3
                self.end_effector_pos, #3
                F.tanh(self.rel_position_termination.unsqueeze(-1)),#1
                self.tennisball_lin_vel * self.cfg.lin_vel_scale, #3
                self.end_effector_lin_vel * self.cfg.lin_vel_scale, #3
                self.end_effector_lin_acc * self.cfg.lin_acc_scale,
                self.end_effector_rot, #4
                self.to_final_target * self.cfg.to_final_target_scale, #3
                self.to_throwing_pos * self.cfg.to_throwing_pos_scale, #3
                self.to_throwing_vel * self.cfg.to_throwing_vel_scale, #3
                obs_task , #7
            ),
            dim=-1,
        )

        return {"policy":obs_actor , "critic":obs_critic}
        

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
            
        self.revert_penalty = torch.zeros((self.num_envs), device=self.device)
        # Compute positions and velocities
        self.end_effector_pos = self._robot.data.body_pos_w[:, self.end_effector_idx].clone() - self.scene.env_origins
        self.end_effector_rot = self._robot.data.body_quat_w[:, self.end_effector_idx].clone()
        self.end_effector_rot = quat_unique(self.end_effector_rot)
        self.end_effector_lin_vel = self._robot.data.body_lin_vel_w[:, self.end_effector_idx].clone()
        self.end_effector_lin_acc = self._robot.data.body_lin_acc_w[:, self.end_effector_idx].clone()
        self.robot_dof_pos = self._robot.data.joint_pos.clone()
        self.robot_dof_vel = self.joint_vel
        self.dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        self.tennisball_pos = self._tennisball.data.root_pos_w - self.scene.env_origins
        self.tennisball_lin_vel = self._tennisball.data.root_lin_vel_w
        self.tennisball_lin_acc = self._tennisball.data.body_lin_acc_w[:, 0, :].clone()
        
        self.r_v = self.end_effector_lin_vel - self.tennisball_lin_vel
        # Calculate distance between end effector and tennis ball
        distance = torch.norm(self.end_effector_pos - self.tennisball_pos, p=2, dim=-1)
        
        # Calculate jerk
        self.joint_jerk = (self._robot.data.joint_acc - self.prev_joint_acc) / self.dt
        self.prev_joint_acc = self._robot.data.joint_acc.clone()
        self.joint_torque_rate = self._robot.data.applied_torque - self.prev_torque
        self.prev_torque = self._robot.data.applied_torque.clone()


        self.normalized_end_effector_z = quat_apply(self.end_effector_rot, self.z_unit_tensor)
        relative_position_end_effector_ball = self.tennisball_pos - self.end_effector_pos
        self.rel_position_termination = (self.normalized_end_effector_z * relative_position_end_effector_ball).sum(dim=-1)

        # Define task indices
        catching_ball_index = self.cfg.tasks.index('Catching_Ball')
        stabilizing_ball_index = self.cfg.tasks.index('Stabilizing_Ball')
        stabilized_ball_index = self.cfg.tasks.index('Stabilized_Ball')
        moving_to_target_pos_index = self.cfg.tasks.index('Moving_to_target_pos')
        stabilized_at_target_index = self.cfg.tasks.index('Stabilized_at_Target')
        throwing_ball_index = self.cfg.tasks.index('Throwing_Ball')
        stop_manipulator_index = self.cfg.tasks.index('Stop')

        # Get current task indices
        current_task_indices = self.task.argmax(dim=-1)

        # Define masks for current tasks
        in_catching_ball = current_task_indices == catching_ball_index
        in_stabilizing_ball = current_task_indices == stabilizing_ball_index
        in_stabilized_ball = current_task_indices == stabilized_ball_index
        in_moving_to_target = current_task_indices == moving_to_target_pos_index
        in_stabilized_at_target = current_task_indices == stabilized_at_target_index
        in_throwing_ball = current_task_indices == throwing_ball_index
        in_stop_manipulator = current_task_indices == stop_manipulator_index

        # Check if the ball is caught
        self.catched_ball = (distance < 0.12) & (self.rel_position_termination < (self.cfg.tennisball_radius * 1.5 + torch.rand_like(self.rel_position_termination) * 0.01))

        # print(self.rel_position_termination[0])
        # Transition logic
        curriculum_progress = torch.clamp(self.curriculum_level / self.max_curriculum_level, min=0.0, max=1.0)
        self.bouncing_velocity_threshold = 2.5 - 2.0 * curriculum_progress  # 5.0 -> 2.0

        relative_vel = self.end_effector_lin_vel - self.tennisball_lin_vel
        relative_vel_magnitude = torch.norm(relative_vel, p=2, dim=-1)
        velocity_factor = 1.0 - (1.0 / (1.0 + 1.2 * relative_vel_magnitude**2))
        
        can_transition_to_stabilizing_ball = self.catched_ball & in_catching_ball
        can_revert_to_catching_ball = ~self.catched_ball & in_stabilizing_ball
        # high_bouncing = can_revert_to_catching_ball & (relative_vel_magnitude > 2.0)
        high_bouncing = can_revert_to_catching_ball & (relative_vel_magnitude > self.bouncing_velocity_threshold)

        # Update tasks for 'Catching_Ball' ↔ 'Stabilizing_Ball'
        if can_transition_to_stabilizing_ball.any():
            self.task[can_transition_to_stabilizing_ball] = self.one_hot_encoding[stabilizing_ball_index]
        if can_revert_to_catching_ball.any():
            self.task[can_revert_to_catching_ball] = self.one_hot_encoding[catching_ball_index]
            self.revert_penalty[can_revert_to_catching_ball] = -1 - 10 * velocity_factor[can_revert_to_catching_ball]
            # self.revert_penalty[can_revert_to_catching_ball] = -20
            if high_bouncing.any():
                # self.bouncing_happens[high_bouncing] = True
                penalty_scale = 1.0 + curriculum_progress[high_bouncing] * 2.0  # Penalty increases with curriculum
                self.revert_penalty[high_bouncing] = -1 - 10 * velocity_factor[high_bouncing] - penalty_scale * (2 * velocity_factor[high_bouncing])

        distance_condition = (distance < 0.12) & (self.rel_position_termination < (self.cfg.tennisball_radius * 1.2 + torch.rand_like(self.rel_position_termination) * 0.01))
        velocity_condition = torch.norm(self.tennisball_lin_vel, p=2, dim=-1) < 0.1
        rel_velocity_condition = torch.norm(self.tennisball_lin_vel - self.end_effector_lin_vel, p=2, dim=-1) < 0.1

        self.stabilized_ball = (
            distance_condition & velocity_condition & rel_velocity_condition & (in_stabilizing_ball | in_stabilized_ball)
        )
        can_revert_to_stabilizing_ball = (~distance_condition | ~velocity_condition | ~rel_velocity_condition) & in_stabilized_ball

        # Update tasks for 'Stabilizing_Ball' ↔ 'Stabilized_Ball'
        if self.stabilized_ball.any():
            self.task[self.stabilized_ball] = self.one_hot_encoding[stabilized_ball_index]
            self.stabilized_ball_counter[self.stabilized_ball] += 1
        if can_revert_to_stabilizing_ball.any():
            self.task[can_revert_to_stabilizing_ball] = self.one_hot_encoding[stabilizing_ball_index]
            self.stabilized_ball_counter[can_revert_to_stabilizing_ball] += -1
            self.revert_penalty[can_revert_to_stabilizing_ball] = -0.4
        
        rel_velocity_condition_to_taret = torch.norm(self.tennisball_lin_vel - self.end_effector_lin_vel, p=2, dim=-1) < 0.1

        self.moving_to_target_pos = (self.stabilized_ball_counter > 20) & distance_condition & rel_velocity_condition_to_taret
        can_revert_to_stabilized_ball = ~self.moving_to_target_pos & in_moving_to_target

        if self.moving_to_target_pos.any():
            self.task[self.moving_to_target_pos] = self.one_hot_encoding[moving_to_target_pos_index]
        if can_revert_to_stabilized_ball.any():
            self.task[can_revert_to_stabilized_ball] = self.one_hot_encoding[stabilized_ball_index]
            self.revert_penalty[can_revert_to_stabilized_ball] = -0.4

        # Transition to 'Stabilized_at_Target' from 'Moving_to_target_pos'
        dist_to_target_pos = torch.norm((self.final_target_pos - self.tennisball_pos), p=2, dim=-1)
        final_rel_dis_cond = (distance < 0.12) & (self.rel_position_termination < (self.cfg.tennisball_radius * 1.2 + torch.rand_like(self.rel_position_termination) * 0.01))
        final_vel_cond = torch.norm(self.tennisball_lin_vel, p=2, dim=-1) < 0.1
        final_rel_velocity_condition = torch.norm(self.tennisball_lin_vel - self.end_effector_lin_vel, p=2, dim=-1) < 0.1
        final_target_dist_reward = dist_to_target_pos < 0.1

        self.stabilized_at_target = (
            final_rel_dis_cond
            & final_vel_cond
            & final_rel_velocity_condition
            & final_target_dist_reward
            & (in_moving_to_target | in_stabilized_at_target)
        )

        can_revert_to_moving_to_target = ~final_rel_dis_cond & in_stabilized_at_target

        if self.stabilized_at_target.any():
            self.task[self.stabilized_at_target] = self.one_hot_encoding[stabilized_at_target_index]
            self.stabilized_ball_counter_at_target[self.stabilized_at_target] += 1
        if can_revert_to_moving_to_target.any():
            self.task[can_revert_to_moving_to_target] = self.one_hot_encoding[moving_to_target_pos_index]
            self.stabilized_ball_counter_at_target[can_revert_to_moving_to_target] += -1
            self.revert_penalty[can_revert_to_moving_to_target] = -0.4

        self.throwing_ball = (self.stabilized_ball_counter_at_target > 20)

        if self.throwing_ball.any():
            self.task[self.throwing_ball] = self.one_hot_encoding[throwing_ball_index]
            self.throwing_termination[self.throwing_ball] = True

        self.stop_manipulator = (in_throwing_ball & (self.tennisball_pos[:, 0] > 0.2)) | in_stop_manipulator

        if self.stop_manipulator.any():
            self.task[self.stop_manipulator] = self.one_hot_encoding[stop_manipulator_index]

        self.finished_episode = self.stop_manipulator & torch.all(self.robot_dof_vel.abs()< 0.05, dim=-1)

    def _compute_rewards(
        self,
        actions,
        tennisball_pos,
        tennisball_lin_vel,
        end_effector_pos,
        end_effector_rot,
        end_effector_lin_vel,
        dist_reward_scale,
        orient_reward_scale,
        stability_reward_scale,
        dist_to_target_pos_reward_scale,
        dof_at_limit_cost_scale,
        dof_vel_lim_penalty_scale,
        joint_accel_scale,
        joint_jerk_scale,
        joint_torque_scale,
        to_throwing_pos_reward_scale,
        relative_velocity_reward_scale,
        action_sign_change_penalty_scale,
        joint_vel_penalty_scale,
        action_penalty_scale,
        task,
    ):
        # distance from end effector to the tennisball
        d = torch.norm(end_effector_pos - tennisball_pos, p=2, dim=-1)
        tennisball_lin_vel_normalized = F.normalize(tennisball_lin_vel, p=2, dim=-1)
        tennisball_lin_vel_normalized[:, 2] = -torch.abs(tennisball_lin_vel_normalized[:, 2])
        # dist_reward = torch.zeros((self.num_envs), device=self.device)
        dist_reward = 0.05 / (1.0 + d**2)

        # orientation of the end effector be along the opposite side of the tennisball velocity
        normalized_end_effector_z = quat_apply(end_effector_rot, self.z_unit_tensor)
        # print(tennisball_lin_vel_normalized.shape)
        orient_reward = -0.2 * (normalized_end_effector_z * tennisball_lin_vel_normalized).sum(dim=-1)

        #Relative speed between end effecor and tennisball
        # r_v = torch.norm(end_effector_lin_vel - tennisball_lin_vel, p=2, dim=-1)
        # r_v = torch.exp(-0.15 * r_v**2)
        # r_v_reward = torch.exp(-2.0 * d**2) * r_v
        r_v = torch.norm(end_effector_lin_vel - tennisball_lin_vel, p=2, dim=-1)
        rel_acc  = torch.norm(self.end_effector_lin_acc - (0.8 * self.tennisball_lin_acc), p=2, dim=-1)
        distance_weight = torch.exp(-1.0 * d**2)
        r_v_weight = torch.exp(-0.2 * r_v**2)
        r_acc_weight = torch.exp(-0.05*rel_acc**2)

        r_v_reward = distance_weight * r_v_weight * r_acc_weight + 0.1 * r_v_weight 
        # r_v_reward = torch.exp(-0.5 * r_v**2)

        
        # rel_acc_norm =  torch.norm(rel_acc, p=2, dim=-1)
        # acc_pen  = 0.01 * rel_acc_norm * torch.exp(-4*d)

        # Penalty while reaching the terminations
        termination_penalty = torch.zeros((self.num_envs), device=self.device)
        termination_penalty[self.terminated_penalty] = 5.0

        stability_reward = torch.zeros((self.num_envs), device=self.device)
        dist_to_target_pos_reward = torch.zeros((self.num_envs), device=self.device)

        tennisball_linear_vel_mag = torch.norm(tennisball_lin_vel, p=2, dim=-1)
        
        dist_to_target_pos =  torch.norm((self.final_target_pos - self.tennisball_pos), p=2, dim=-1)

        finished_reward = torch.zeros((self.num_envs), device=self.device)
        finished_reward[self.finished_episode] = 400

        to_throwing_vel_reward = torch.zeros((self.num_envs), device=self.device)
        to_throwing_pos_reward = torch.zeros((self.num_envs), device=self.device)

        joint_vel_penalty = torch.zeros((self.num_envs), device=self.device)

        # Initialize a mask for each condition
        stop_manipulator = self.stop_manipulator
        
        throwing_ball_mask = self.throwing_ball & ~self.stop_manipulator

        final_target_reached_mask = self.stabilized_at_target & ~throwing_ball_mask & ~self.stop_manipulator
        # If not final target reached, check moving to target
        moving_to_target_mask = self.moving_to_target_pos & ~final_target_reached_mask & ~throwing_ball_mask & ~self.stop_manipulator
        # If not final target reached or moving to target, check stabilized ball
        stabilized_ball_mask = self.stabilized_ball & ~final_target_reached_mask & ~moving_to_target_mask & ~throwing_ball_mask & ~self.stop_manipulator
        # If not in any above states, check catched ball
        catched_ball_mask = self.catched_ball & ~final_target_reached_mask & ~moving_to_target_mask & ~stabilized_ball_mask & ~throwing_ball_mask & ~self.stop_manipulator

        alive_reward = torch.ones((self.num_envs), device=self.device) * 0.3
        dof_at_limit_cost = torch.sum((self.dof_pos_scaled).abs() > 0.97, dim=-1)
        dof_vel_lim_penalty = torch.sum((self.robot_dof_vel).abs() > 0.97 * self.robot_dof_vel_limit, dim=-1)

        self.joint_jerk_penalty = torch.sum(self.joint_jerk**2, dim=-1)
        self.joint_torque_rate_penalty = torch.sum(self.joint_torque_rate**2, dim=-1)

        self.joint_acc_penalty = torch.sum(self._robot.data.joint_acc**2, dim=-1)

        action_sign_change = torch.sign(actions) - torch.sign(self.last_actions)
        action_sign_change_penalty = torch.sum(action_sign_change**2, dim=-1)
        self.last_actions = self.pure_actions.clone()

        action_penalty = torch.sum(self.pure_actions**2, dim=-1)
        
        if stop_manipulator.any():
            dist_reward[stop_manipulator] = 1.6 * 2
            stability_reward[stop_manipulator] = 1.6
            dist_to_target_pos_reward[stop_manipulator] = 1.2
            to_throwing_pos_reward[stop_manipulator] = 3.0
            # alive_reward[stop_manipulator] += 1.8
            r_v_reward[stop_manipulator] = 1.6
            joint_vel_penalty[stop_manipulator] =  1.0 / (1.0 + 0.05 * torch.norm(self.joint_vel[stop_manipulator], p=2, dim=-1)**2)
            orient_reward[stop_manipulator] = 1.0
            # acc_pen[stop_manipulator] = 0.0

        if throwing_ball_mask.any():
            dist_reward[throwing_ball_mask] = 1.6 * 2
            stability_reward[throwing_ball_mask] = 1.6
            dist_to_target_pos_reward[throwing_ball_mask] = 1.2
            alive_reward[throwing_ball_mask] += 1.6
            r_v_reward[throwing_ball_mask] = 1.6
            to_throwing_vel_reward[throwing_ball_mask] = (1.5 / (1.0 + 0.5 * torch.norm(self.throwing_vel[throwing_ball_mask] - self.tennisball_lin_vel[throwing_ball_mask], p=2, dim=-1)**2))
            to_throwing_pos_reward[throwing_ball_mask] = 1.5 / (1.0 + 2.0 * torch.norm(self.throwing_pos[throwing_ball_mask] - self.tennisball_pos[throwing_ball_mask], p=2, dim=-1)**2)
            to_throwing_pos_reward[throwing_ball_mask] = to_throwing_pos_reward[throwing_ball_mask] * to_throwing_vel_reward[throwing_ball_mask] * torch.norm(self.tennisball_lin_vel[throwing_ball_mask], p=2, dim=-1)
            # orient_reward[throwing_ball_mask] = 2.0 * (self.throwing_vel_norm[throwing_ball_mask] * normalized_end_effector_z[throwing_ball_mask]).sum(dim=-1)
            orient_reward[throwing_ball_mask] = 1.0
            # acc_pen[throwing_ball_mask] = 0.0
            # print("hhhhhhhhhhhhere")
            # print(orient_reward[throwing_ball_mask])

        if final_target_reached_mask.any():
            dist_reward[final_target_reached_mask] = 1.6 * (1.0 / (1.0 + d[final_target_reached_mask]**2))
            stability_reward[final_target_reached_mask] = 1.6 / (1.0 + tennisball_linear_vel_mag[final_target_reached_mask])
            dist_to_target_pos_reward[final_target_reached_mask] = (1.2/ (1.0 + 5.0 * (dist_to_target_pos[final_target_reached_mask])**2))
            # alive_reward[final_target_reached_mask] += 1.4
            r_v_reward[final_target_reached_mask] = 1.6 * torch.exp(-0.5 * r_v[final_target_reached_mask]**2)
            orient_reward[final_target_reached_mask] = 1.0
            # acc_pen[final_target_reached_mask] = 0.0
            # orient_reward[final_target_reached_mask] = 2.0 * (self.throwing_vel_norm[final_target_reached_mask] * normalized_end_effector_z[final_target_reached_mask]).sum(dim=-1) + 1.0
            # print("hhhhhhhhhhhhere")
            # print(orient_reward[final_target_reached_mask])

        if moving_to_target_mask.any():
            dist_reward[moving_to_target_mask] = 1.5 * (1.0 / (1.0 + d[moving_to_target_mask]**2))
            stability_reward[moving_to_target_mask] = 1.3
            dist_to_target_pos_reward[moving_to_target_mask] = (1.0 / (1.0 + 5.0 * (dist_to_target_pos[moving_to_target_mask])**2))
            # alive_reward[moving_to_target_mask] += 1.0
            r_v_reward[moving_to_target_mask] = 1.5 * torch.exp(-0.5 * r_v[moving_to_target_mask]**2)
            orient_reward[moving_to_target_mask] = 1.0
            # acc_pen[moving_to_target_mask] = 0.0

        if stabilized_ball_mask.any():
            dist_reward[stabilized_ball_mask] = 1.3 * (1.0 / (1.0 + d[stabilized_ball_mask]**2))
            stability_reward[stabilized_ball_mask] = 1.3 / (1.0 + tennisball_linear_vel_mag[stabilized_ball_mask])
            # alive_reward[stabilized_ball_mask] += 0.6
            r_v_reward[stabilized_ball_mask] = 1.3 * torch.exp(-0.5 * r_v[stabilized_ball_mask]**2)
            orient_reward[stabilized_ball_mask] = 1.0
            # acc_pen[stabilized_ball_mask] = 0.0

        if catched_ball_mask.any():
            dist_reward[catched_ball_mask] = 1.1 * (1.0 / (1.0 + d[catched_ball_mask]**2))
            stability_reward[catched_ball_mask] = 1.1 / (1.0 + tennisball_linear_vel_mag[catched_ball_mask])
            # alive_reward[catched_ball_mask] += 0.2
            r_v_reward[catched_ball_mask] = 1.1 * torch.exp(-0.5 * r_v[catched_ball_mask]**2)
            orient_reward[catched_ball_mask] = 1.0
            # acc_pen[catched_ball_mask] = 0.0


        rewards = (
            dist_reward_scale * dist_reward     #wt
            + orient_reward_scale * orient_reward
            + r_v_reward * relative_velocity_reward_scale       #wt
            + stability_reward * stability_reward_scale     #wt                                                                                         
            + dist_to_target_pos_reward * dist_to_target_pos_reward_scale       #wt
            + to_throwing_pos_reward * to_throwing_pos_reward_scale #w
            - termination_penalty       #wt
            - dof_at_limit_cost * dof_at_limit_cost_scale       #w
            - dof_vel_lim_penalty * dof_vel_lim_penalty_scale     #w  
            - self.joint_acc_penalty * joint_accel_scale        #wt
            # - self.joint_jerk_penalty * joint_jerk_scale        #wt
            - self.joint_torque_rate_penalty * joint_torque_scale       #w
            - action_sign_change_penalty * action_sign_change_penalty_scale     #w
            # + alive_reward
            + self.revert_penalty       #w
            + finished_reward #w
            + joint_vel_penalty * joint_vel_penalty_scale #w
            # - action_penalty * action_penalty_scale
            # - acc_pen
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "orient_reward": (orient_reward_scale * orient_reward).mean(),
            "r_v_reward": (relative_velocity_reward_scale * r_v_reward).mean(),
            "stability_reward": (stability_reward * stability_reward_scale).mean(),
            "dist_to_final_target_pos_reward": (dist_to_target_pos_reward * dist_to_target_pos_reward_scale).mean(),
            "to_throwing_pos": (to_throwing_pos_reward * to_throwing_pos_reward_scale).mean(),
            "termination_penalty": (-termination_penalty).mean(),
            "dof_at_limit_cost": (-dof_at_limit_cost * dof_at_limit_cost_scale).mean(),
            "dof_vel_lim_penalty": (-dof_vel_lim_penalty * dof_vel_lim_penalty_scale).mean(),
            "joint_accel": (-self.joint_acc_penalty * joint_accel_scale).mean(),
            "joint_jerk": (-self.joint_jerk_penalty * joint_jerk_scale).mean(),
            "joint_torque_rate_penalty": (-self.joint_torque_rate_penalty * joint_torque_scale).mean(),
            "action_sign_change_penalty": (- action_sign_change_penalty * action_sign_change_penalty_scale).mean(),
            "alive_reward": (alive_reward).mean(),
            "revert_penalty": (self.revert_penalty).mean(),
            "finished_reward": (finished_reward).mean(),
            "joint_vel_penalty": (joint_vel_penalty * joint_vel_penalty_scale).mean(),
            # "action_penalty": (action_penalty * action_penalty_scale).mean(),
            "Curriculum_value": (self.bouncing_velocity_threshold).mean(),
            # "acc_pen": (-acc_pen).mean()
        }

        return rewards


    def random_ranges(self, env_ids):
        # Define ranges for each axis
        x_range = (-0.1, 0.1)
        y_range_1 = (-0.05, 0.05)
        y_range_2 = (-0.05, 0.05)
        z_range = (-0.05, 0.05)

        num_envs = len(env_ids)

        # Sample random numbers for each axis
        x_noise = torch.FloatTensor(num_envs, 1).uniform_(*x_range).to(self.device)
        
        # Determine y values based on whether env_id is odd or even
        y_noise = torch.zeros(num_envs, 1).to(self.device)
        for i, env_id in enumerate(env_ids):
            if env_id % 2 == 0:  # Even env_id: select from y_range_2 (left side)
                y_noise[i] = torch.FloatTensor(1).uniform_(*y_range_2).to(self.device)
            else:  # Odd env_id: select from y_range_1 (right side)
                y_noise[i] = torch.FloatTensor(1).uniform_(*y_range_1).to(self.device)
        
        z_noise = torch.FloatTensor(num_envs, 1).uniform_(*z_range).to(self.device)

        # Concatenate the noise for each axis
        pos_noise = torch.cat((x_noise, y_noise, z_noise), dim=1)

        return pos_noise
    
    # def random_throw(
    #     self,
    #     init_state,  # (N, 3) = [ [x0,y0,z0], [x0,y0,z0], ... ]
    #     final_state, # (N, 3) = [ [xT,yT,zT], [xT,yT,zT], ... ]
    #     g=9.81,
    #     x_range=0.3,  # 10 cm
    #     y_range=0.25,  # 5 cm
    #     z_range=0.2,  # 10 cm
    #     t_min=0.8,
    #     t_max=1.2,
    #     device=None
    # ):
        
    #     if device is None:
    #         device = init_state.device

    #     N = init_state.shape[0]  # number of parallel envs
        
    #     # Generate random offsets in [-x_range, x_range], shape (N,)
    #     rand_x_init = (torch.rand(N, device=device) * 2 - 1) * x_range
    #     rand_y_init = (torch.rand(N, device=device) * 2 - 1) * y_range
    #     rand_z_init = (torch.rand(N, device=device) * 2 - 1) * z_range

    #     # rand_x_final = (torch.rand(N, device=device) * 2 - 1) * x_range
    #     # rand_x_final = - torch.rand(N, device=device) * x_range * 2.0 - 0.2
    #     # # rand_y_final = (torch.rand(N, device=device) * 2 - 1) * y_range
    #     # rand_y_final = -torch.rand(N, device=device) * 0.4 + 0.1  # +0.1 to -0.2 (i.e., +10cm to -20cm)
    #     # # rand_z_final = (torch.rand(N, device=device) * 2 - 1) * z_range
    #     # rand_z_final = torch.rand(N, device=device) *  z_range * 2.0

    #     rand_x_final = -torch.rand(N, device=device) * 0.7 - 0.31
    #     rand_y_final = torch.rand(N, device=device) * 0.55 - 0.3
    #     rand_z_final = torch.rand(N, device=device) * 0.6 - 0.25

    #     # Perturbed initial positions
    #     x_init = init_state[:, 0] + rand_x_init
    #     y_init = init_state[:, 1] + rand_y_init
    #     z_init = init_state[:, 2] + rand_z_init

    #     # Perturbed final positions
    #     x_final = final_state[:, 0] + rand_x_final
    #     y_final = final_state[:, 1] + rand_y_final
    #     z_final = final_state[:, 2] + rand_z_final + 0.1

    #     # Random flight times in [t_min, t_max]
    #     t = t_min + (t_max - t_min) * torch.rand(N, device=device)

    #     # Compute velocities
    #     vx = (x_final - x_init) / t
    #     vy = (y_final - y_init) / t
    #     vz = (z_final - z_init + 0.5 * g * t * t) / t
        
    #     # Stack the results in shape (N, 7)
    #     out = torch.stack([x_init, y_init, z_init, vx, vy, vz], dim=1)
    #     return out

    def random_throw(
        self,
        init_state,  # (N, 3)  world-frame default ball position
        g=9.81,
        t_min=0.8,
        t_max=1.2,
        device=None,
    ):
        """
        Sample only the initial position (from init_state ± ranges) and
        return the 6-DOF state [x, y, z, vx, vy, vz] so the ball lands
        anywhere inside the world-frame target box
            x : ‑0.7 … -0.2 m
            y : ‑0.75 … ‑0.1 m
            z : 0.2 … 0.85 m
        """
        if device is None:
            device = init_state.device
        N = init_state.shape[0]

        # ---------- initial position ----------
        rand_x_init = (torch.rand(N, device=device) * 2 - 1) * 0.30  # ±0.30
        rand_y_init = (torch.rand(N, device=device) * 2 - 1) * 0.25  # ±0.25
        rand_z_init = (torch.rand(N, device=device) * 2 - 1) * 0.20  # ±0.20

        x_init = init_state[:, 0] + rand_x_init
        y_init = init_state[:, 1] + rand_y_init
        z_init = init_state[:, 2] + rand_z_init

        # ---------- target position ----------
        x_final = torch.rand(N, device=device) * 0.5 - 0.7          # [-0.75, 0.0]
        y_final = torch.rand(N, device=device) * 0.65 - 0.75          # [-0.75, -0.1]
        z_final = torch.rand(N, device=device) * 0.65 + 0.20          # [0.40, 1.05]

        # ---------- flight time ----------
        t = t_min + (t_max - t_min) * torch.rand(N, device=device)

        # ---------- required velocities ----------
        vx = (x_final - x_init) / t
        vy = (y_final - y_init) / t
        vz = (z_final - z_init + 0.5 * g * t * t) / t

        return torch.stack([x_init, y_init, z_init, vx, vy, vz], dim=1)
    

    def random_throw_back(
        self,
        init_state,  # (N, 3) = [ [x0,y0,z0], [x0,y0,z0], ... ]
        final_state, # (N, 3) = [ [xT,yT,zT], [xT,yT,zT], ... ]
        g=9.81,
        device=None
    ):
        
        if device is None:
            device = init_state.device

        N = init_state.shape[0]

        # Calculate flight time from vertical displacement:
        t = torch.sqrt(2 * (init_state[:, 2] / g))

        # Compute velocities required to cover horizontal displacement in time t
        vx = torch.ones((N), device=self.device) * 3
        vy = torch.zeros((N), device=self.device)
        vz = torch.ones((N), device=self.device) * 0.2

        # Stack the velocity components in shape (N, 3)
        out = torch.stack([vx, vy, vz], dim=1)
        return out
    