# OptiTrack Marker Listener

This repository contains a simplified ROS2 package that implements the NatNet protocol to receive OptiTrack marker data and stream it over the ROS2 network.

# What This Package Does
This package provides two complementary scripts for tracking individual markers (not rigid bodies) from an OptiTrack system:
   - **Basic Marker Tracking**: Tracks and publishes ball marker positions with EMA filtering
   - **Marker Recording**: Same tracking functionality plus automatic trajectory recording to CSV files

# Key Features
- Tracks individual markers instead of rigid bodies
- Applies exponential moving average (EMA) filtering for smooth ball tracking
- Transforms coordinates from OptiTrack frame to world frame
- Finds the closest marker to a reference position for consistent tracking
- Automatic trajectory recording based on configurable spatial boundaries
- Publishes marker position as ROS2 PoseStamped messages

# Requirements
- ROS2 installation (tested with Humble)
- Python3 pip: sudo apt-get install python3-pip
- optirx library: pip3 install optirx

# Installation
Navigate to your ROS2 workspace and clone the repository:

```sh
cd ros2_ws/src
git clone https://github.com/Afshin1997/kuka_rl_ros2.git
```
Build the workspace:
```sh
colcon build --symlink-install
```

# Scripts
1- listener.py

**Purpose**: Basic marker tracking and publishing

**Output**: Publishes filtered marker positions on /optitrack/ball_marker

**Use case**: Real-time marker tracking for control or monitoring applications

2- trajectory_collector.py

**Purpose**: Marker tracking with automatic trajectory recording

**Output**: Publishes marker positions + saves trajectory data to CSV files

**Use case**: Data collection, trajectory analysis, motion studies

# Execution
For basic marker tracking:

```sh
ros2 launch optitrack_listener optitrack_listener.launch.py
```

For marker tracking with recording:

```sh
python3 trajectory_collector.py
```

# Motive Configuration

In Motive's Streaming panel:
- Check "Broadcast Frame Data"
- Set "Stream Markers" to true
- Select "Multicast" communication type
- Ports: Command port 1510, Data port 1511
- Local interface: Select "Local Loopback"
- Multicast Interface: Set to receiving device IP (e.g., 172.31.1.145)
# Marker Setup
No rigid body creation needed - the system tracks individual markers directly.
# Configuration
```sh
optitrack_listener:
  ros__parameters:
    fixed_frame: "world"
    local_interface: '172.31.1.145'
    initial_ball_position: [0.0, 0.0, 0.0]
    ball_ema_alpha: 0.1
    trajectory_save_folder: 'marker_trajectories'
    recording_x_min: 0.0
    recording_x_max: 3.0
    recording_z_min: 0.0
```
## Parameter Descriptions:
### Common Parameters (used by both scripts):
- fixed_frame: Reference frame for published poses (typically "world")
- local_interface: IP address for receiving OptiTrack data
- initial_ball_position: Starting reference position for marker tracking [x, y, z]
- update_rate_hz: Expected data update rate for filtering
- ball_ema_alpha: EMA smoothing factor (0-1, higher = less smoothing)

### Recording-Only Parameters (used only by recorder script):

- trajectory_save_folder: Directory name for saving CSV trajectory files
- recording_x_min: Minimum X coordinate for recording zone (world frame)
- recording_x_max: Maximum X coordinate for recording zone (world frame)
- recording_z_min: Minimum Z coordinate for recording zone (world frame)

# Output
## Basic Listener Output
The node publishes:
- /optitrack/ball_marker: PoseStamped message with filtered ball marker position

## Recorder Output
- /optitrack/ball_marker: PoseStamped message with filtered ball marker position
- CSV files: Automatic trajectory recording when marker enters defined spatial zone

# Coordinate Transformation
The system transforms OptiTrack coordinates to world frame using:

- X = -opti_z + 0.7

- Y = -opti_x + 0.0

- Z = opti_y + 0.5106 + 0.05375

# Recording Behavior
The recorder automatically:

1- Starts recording when the marker enters the defined zone (x_min < x < x_max AND z > z_min)

2- Continues recording while marker remains in zone

3- Stops and saves trajectory when marker exits the zone

4- Creates new file for each recording session with timestamp

# Tips and tricks
- The initial_ball_position of the marker should be considered in the cameras frame, not the manipulator's frame.
- After clibration of the cameras, in the "Reconstruction" tab chnage the following parameters to have consistent tracking.

   - " maximum Calculation Time" : 10
   - "Circulatory": 0.5 - 0.7
   - "Maximum Marker Spacing(mm)": 40 - 50

- For Markers Group, change the following parameters as well:
   - FPS: 100
   - EXP: 200
   - THR: 200
