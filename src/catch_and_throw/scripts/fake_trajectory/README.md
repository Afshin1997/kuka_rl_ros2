# Fake Trajectory

## Overview

This script is a ROS (Robot Operating System) node designed to control a robotic manipulator by publishing target joint positions and recording various state information during operation. It performs the following functions:

- **Reads Target Joint Positions**
- **Publishes Joint Positions**: Sends the target joint positions to the robot via the `/joint_reference` topic
- **Subscribes to Robot State Topics**: Listens to `/joint_states`, `/ee_pose`, and `/ee_vel` to receive the robot's current joint states, end-effector pose, and end-effector velocity.
- **Records Data**
- **Saves Data to CSV Files**

## Troubleshooting

- **File Not Found Errors**: Ensure that the input CSV file is placed correctly in the `input_files` directory.
- **Topic Subscription Errors**: Verify that the robot is publishing the necessary topics using `rostopic list`.
- **Permission Denied**: Ensure the script has executable permissions: `chmod +x joint_pos_test.py`.