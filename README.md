This repo contains the code for Multi Task Reinforcement Learning for Non-Prehensile Manipulation (ball catching and throwing)

## Prerequisites

### Docker
Before setting up the project, you have to install docker. If you already installed docker, go to the next section
```sh
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install the Docker packages
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add docker user
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
Then log out and log in.

### Git LFS
You need git LFS to pull large files (torch)
```sh
sudo apt install git-lfs
```
## Experimental setup
To prepare the experimental setup, you need to calibrate optitrack and to mount the end-effector

# OptiTrack System Calibration

## Setup
1. **Connect Hardware:**
   - Connect cameras to switches, then link switches together to form a camera network.
   - Connect one switch to the PC running Motive Software.
   - Connect this PC to the NETGEAR switch.

## Calibration Steps
1. **Open Motive Software:**
   - Navigate to `Layout` → `Calibrate`.
   - Ensure all cameras appear in the `Camera Preview` window.

2. **Camera Configuration:**
   - Set the **frequency** according to your data acquisition needs.
   - Adjust **exposure** and **threshold** to minimize environmental noise.
   - Verify camera coverage of the workspace. Reposition cameras if needed.

3. **Wanding Process:**
   - Remove all markers from the workspace.
   - Click `Mask Visible`, then `Start Wanding`.
   - Move the calibration wand through the workspace until enough samples are collected (status turns green).
   - Click `Calculate` → `Apply Results`.

4. **Data Streaming Setup:**
   - Go to `View` → `Data Streaming`.
   - Enable `Broadcast Frame Data`.
   - Under `Network Interface` → `Local Interface`:
     - Confirm two camera IPs (`172.31.1.150` and `172.31.1.200`) and `Local Loopback` are listed.
     - Ensure the subnet mask matches the manipulator (`255.255.255.0`). Manually adjust IPs if needed.
   - In `Advanced Network Options`, set `Multicast Interface` to `172.31.1.145`.

5. **Save Calibration:**
   - Save the calibrated model before exiting.

### Robot end-effector
1. Mount the flange adapter and the tray
@AFSHIN



## Compilation
1. Clone the repo.
```sh
git clone --recurse-submodules -j8 https://github.com/Afshin1997/kuka_rl_ros2.git
```

2. Pull LFS files
```sh
git lfs pull
```

3. Open a terminal in the cloned repo and type 
```sh
source docker_build.sh
```

4. Run the container
```sh
source docker_run.sh
```
Now you can jump to the next session.

Note: When finished working, stop the container,
```sh
docker stop <CONTAINER_ID>
```
To start working again, start the container,
```sh
docker start <CONTAINER_ID>
```

## Execution

### 1. Connect to the robot
1. Turn on the robot power switch (turn it off and again turn it on until the robot works)

2. Connect your computer to the network NETGEAR switch

3. Set your static IP address 
Settings -> Network -> Wired -> settings icon -> tab IPv4 -> method = manual -> in Address write 172.31.1.145 netmask 255.255.255.0.

4. Verify your connection state with Kuka whose ip is 172.31.1.147.
```sh
ping 172.31.1.147
```


### 2. Remote side: run the application

1. From teach pendant, switch on aut mode
teach pendant -> turn right the key -> select AUT -> turn left the key.

2. On the robot side, for Calibration:
Click on Applications -> select PositionAndGSMReferncing -> click play (on teach pendant left side, green button)

3. On the robot side, from teach pendant, select the application for command the robot in position through FRI in joint impedance mode.
Click on Applications -> select AFS_FRI_joint_impedance -> click play (on teach pendant left side, green button)

### 3. Client side: run the application
After running the container:

1. Attach one terminal to the container (`docker exec -it <CONTAINER_ID> bash`) and type
```sh
cd kuka_rl_ros2
colcon build
```
to build the packages
and then:
```sh
ros2 launch catch_and_throw manipulator_optitrack.launch.py
```