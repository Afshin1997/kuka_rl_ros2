# Implementation on real Lbr iiwa 7



# Fake Trajectory

### Implicit Controller

```sh
python3 scripts/fake_trajectory/test_implicit.py
```

this script reads and publishes the joint position of `input_files/implicit/ft_implicit.csv` in `200` Hz frequency.

the following data will be recorded in `output_files/ft/implicit`.

- `received_joint_pos`
- `received_joint_vel`
- `eceived_ee_pos`
- `received_ee_orient` in quaternion
- `received_ee_lin_vel`

then, to compare the recorded joint positions w.r.t the simulation environment, the following command should be run:

- **Joint Position**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:set_target_:200::input_files/implicit/ft_implicit.csv:joint_pos_:200::output_files/ft/implicit/ft_received_joint_pos_np.csv:joint_:200]
```

- **Joint Velocity**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:joint_vel_:200::output_files/ft/implicit/ft_received_joint_vel_np.csv:joint_:200]
```

- ****End effector Position****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:end_effector_pos_:200::output_files/ft/implicit/ft_received_ee_pos_np.csv:pos_:200]
```

- ****End effector Orientation****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:ee_orien:200::output_files/ft/implicit/ft_received_ee_orient_np.csv:or_:200]
```

- ****End effector Linear Velocity****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:end_effector_lin_vel_:200::output_files/ft/implicit/ft_received_ee_lin_vel_np.csv:lin_vel_:200]
```

### IdealPD Controller

```sh
python3 scripts/fake_trajectory/test_idealpd.py
```
this script reads and publishes the joint position of `input_files/idealpd/ft_idealpd.csv` in `200` Hz frequency.

the following data will be recorded in `output_files/ft/idealpd`.

- `received_joint_pos`
- `received_joint_vel`
- `eceived_ee_pos`
- `received_ee_orient` in quaternion
- `received_ee_lin_vel`

then, to compare the recorded joint positions w.r.t the simulation environment, the following command should be run:

- **Joint Position**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd_1.csv:set_target_:100::input_files/idealpd/ft_idealpd_1.csv:joint_pos_:100::output_files/ft/idealpd/ft_received_joint_pos_np.csv:joint_:400]

  
```


- **Joint Velocity**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd_1.csv:joint_vel_:100::output_files/ft/idealpd/ft_received_joint_vel_np.csv:joint_:400]
```

- **Joint Efforts**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:torque:200::output_files/ft/idealpd_3/ft_received_joint_effort_np.csv:joint_:200]
```

- ****End effector Position****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:end_effector_pos_:200::output_files/ft/idealpd_3/ft_received_ee_pos_np.csv:pos_:200]
```

- ****End effector Orientation****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:end_effector_rot_:200::output_files/ft/idealpd_7/ft_received_ee_orient_np.csv:or_:200]
```

- ****End effector Linear Velocity****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:end_effector_lin_vel_:200::output_files/ft/idealpd_7/ft_received_ee_lin_vel_np.csv:lin_vel_:200]
```

# Trained Models

### Implicit Controller

```sh
python3 scripts/implementation/trained_implicit.py
```

this script reads and publishes the joint position of `input_files/implicit/ft_implicit.csv` in `200` Hz frequency. Then, the collected data through real Lbr iiwa manipulator and the csv file fed into the network and the target joint positions are published to the manipulator.

the following data will be recorded in `output_files/tm/implicit`.

- `joint_target`
- `received_joint_pos`
- `received_joint_vel`
- `eceived_ee_pos`
- `received_ee_orient` in quaternion
- `received_ee_lin_vel`

then, to compare the recorded joint positions w.r.t the simulation environment, the following command should be run:

- **Joint Position**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:set_target_:200::input_files/implicit/ft_implicit.csv:joint_pos_:200::output_files/tm/implicit/tm_received_joint_target_np.csv:joint_:200::output_files/tm/implicit/tm_received_joint_pos_np.csv:joint_:200]
```


- **Joint Velocity**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:joint_vel_:200::output_files/tm/implicit/tm_received_joint_vel_np.csv:joint_:200]
```

- ****End effector Position****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:end_effector_pos_:200::output_files/tm/implicit/tm_received_ee_pos_np.csv:pos_:200]
```

- ****End effector Orientation****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:ee_orien:200::output_files/tm/implicit/tm_received_ee_orientation_np.csv:or_:200]
```

- ****End effector Linear Velocity****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/implicit/ft_implicit.csv:end_effector_lin_vel_:200::output_files/tm/implicit/tm_received_ee_vel_np.csv:lin_vel_:200]
```

### IdealPD Controller

```sh
python3 scripts/implementation/trained_idealpd.py
```
this script reads and publishes the joint position of `input_files/idealpd/ft_idealpd.csv` in `200` Hz frequency.

the following data will be recorded in `output_files/tm/idealpd`.
- `joint_target`
- `received_joint_pos`
- `received_joint_vel`
- `eceived_ee_pos`
- `received_ee_orient` in quaternion
- `received_ee_lin_vel`

then, to compare the recorded joint positions w.r.t the simulation environment, the following command should be run:

- **Joint Position**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd_1.csv:joint_pos_:100::output_files/tm/idealpd/tm_received_joint_pos_np.csv:joint_:100]
```

python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:tennisball_:100]

python3 scripts/tools/plot.py \
--window [output_files/tm/idealpd/tm_ball_positions_world_np.csv:ball_:100]

python3 scripts/tools/plot.py \
--window [output_files/tm/idealpd/tm_ball_velocities_ema_world_np.csv:ball_:100]


- **Joint Velocity**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd_1.csv:joint_vel_:100::output_files/tm/idealpd/tm_received_joint_vel_np.csv:joint_:100]
```

- **Joint Efforts**

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:torque:100::output_files/tm/idealpd/tm_received_joint_effort_np.csv:joint_:200]
```

- ****End effector Position****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:end_effector_pos_:100::output_files/tm/idealpd/tm_received_ee_pos_np.csv:pos_:100]
```

- ****End effector Orientation****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:end_effector_rot_:100::output_files/tm/idealpd/tm_received_ee_orientation_np.csv:or_:100]
```

- ****End effector Linear Velocity****

```sh
python3 scripts/tools/plot.py \
  --window [input_files/idealpd/ft_idealpd.csv:end_effector_lin_vel_:100::output_files/tm/idealpd/tm_received_ee_vel_np.csv:lin_vel_:200]
```


python3 scripts/tools/plot.py \
  --window [output_files/tm/idealpd/tm_filtered_actions_np.csv:action_:200::output_files/tm/idealpd/tm_raw_actions_np.csv:action_:200]
