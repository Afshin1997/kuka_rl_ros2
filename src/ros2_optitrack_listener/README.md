# Optitrack listener

This repository contains a ROS2 package implementing the NatNet _v2.5_ protocol to received Optitrack data and stream over the ROS2 network.

### Motivation
NatNet _v2.5_ protocol is old and it's contained in the Motive software 1.5 (2013). Other packages available online failed to receive data. o t

### Requirements
- A ROS2 installation (this package has been tested with humble)
 - Pip3 (you can install it using apt)

	   $ apt-get install python3-pip
- optirx (you can install it using pip3)
	
	  $ pip3 install optirx

### Installation
Move in your ROS2 workspace

	$ cd ros2_ws/src
	$ git clone git@github.com:jocacace/ros2_optitrack_listener.git

Compile the workspace

	$ colcon build --symlink-install 

### Execution
To start the optitrack listener node, you can use a convenient launch file included in the launch folder of the package. 

	$ ros2 launch optitrack_listener optitrack_listener.launch.py 

## Motive configuration

### Network
Before to start the optitrack node, you should properly configure the Motive software. Here is how to set it.

- Open the _Streaming panel_:
	- Pin the _Broadcast Frame Data_ checkbox
	- Set the _Stream Rigid body_ to true
	- Select the  _Multicast_ in the communication type
	- Ports: 
		- Command port: 1510
		- Data port: 1511
	- Local interface: this interface must be set to use the one communicating with the remote computer. For example if you have two network devices and only one of them can ping the remote computer, the local interface must be set to that Ip address
- Multicast Interface: This Ip address must be the one of the device receiving data

### Trackable
Before to start the data streaming, a set of trackable should be created. You can directly select the markers composing a given trackable. However, after created the object, in the trackable proprieties panel you can set its name and its id. The id is important to properly configure the ROS2 data streaming

### Node configuration

To configure the node a proper configuration file is already contained in the config folder of the package. It is a yaml file 

	optitrack_listener:
		ros__parameters:
			fixed_frame: "map"
			local_interface: '192.168.1.4'
			rigid_object_list: 'body_2,body_3'
			trackables:
				body_2:
					id: 1
					name: ground_plan_tool
				body_3:
					id: 4
					name: calibration_wand
			publish_tf: True

Here we can specify:
- Fixed frame: the frame toward publish the pose data
- local_interface: the Ip address where optitrack data are received
- rigid_object list: this is a list of names. The names are used to initialize the other parameters of the file. For example, here is expected to find the body_2 and body_3 parameter definition. The names included here must be separated by the "," (comma) separator. No matter the space before or after the comma
- trackables: the list of all the rigid object that can be found into the optitrack data. Each of them has an id, that is the id received via the optitrack network protocol and a name. The name is also used to create a topic publishing data in the ROS2 network
- publish_tf: set this variable to True to publish data also in tf form, false otherwise

### Output
For each trackable  (${trackable}) name in the yaml configuration file the following output is generated:
- /optitrack/${trackable}: a PoseStamped message containing timestamp and pose data of the object
- A tf data if the publish_tf param is set to true 


