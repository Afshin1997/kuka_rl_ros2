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

### Optitrack calibration
1. Position and connect the cameras such that the field of view covers the entire scene.

2. Position the robot in the default configuration (run the application @AFSHIN) and mount the ground frame to the end-effector
@AFSHIN

3. Calibrate the optitrack from the optitrack pc 
@AFSHIN

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
Click on Applications -> select AFS_FRI_joint_impedance @AFSHIN -> click play (on teach pendant left side, green button)

### 3. Client side: run the application
After running the container:

1. Attach one terminal to the container (`docker exec -it <CONTAINER_ID> bash`) and type
```sh
@AFSHIN
```
2. Attach another terminal to the container (`docker exec -it <CONTAINER_ID> bash`) and type
```sh
@AFSHIN
```
