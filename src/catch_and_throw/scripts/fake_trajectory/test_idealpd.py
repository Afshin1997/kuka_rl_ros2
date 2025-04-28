#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import pandas as pd
import numpy as np
from geometry_msgs.msg import Pose, Twist
import matplotlib
import matplotlib.pyplot as plt
import os  # Import os module for path handling

received_joint_pos = []
received_joint_vel = []
received_ee_pos = []
received_ee_orient = []
received_ee_lin_vel = []
recording = False

def joint_states_callback(msg):
    if recording:
        received_joint_pos.append(msg.position)
        received_joint_vel.append(msg.velocity)

def ee_pose_callback(msg):
    if recording:
        pos = [msg.position.x, msg.position.y, msg.position.z]
        orient = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        
        received_ee_pos.append(pos)
        received_ee_orient.append(orient)

def ee_vel_callback(msg):
    if recording:
        lin_vel = [msg.linear.x, msg.linear.y, msg.linear.z]
        received_ee_lin_vel.append(lin_vel)

def save_output(outputs, output_file_path, header=None):
    np.savetxt(output_file_path, outputs, delimiter=',', header=header, comments='')
    print(f"Model outputs saved to {output_file_path}")

def joint_pos_test():
    global recording

    rospy.init_node('joint_state_node', anonymous=True)
    rate = rospy.Rate(200)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the input file path
    input_file_path = os.path.join(script_dir, '../../input_files/idealpd/ft_idealpd_9.csv')
    input_file_path = os.path.abspath(input_file_path)  # Convert to absolute path

    # Read the observation data
    obs_data = pd.read_csv(input_file_path)
    set_target = obs_data.iloc[:1200, 21:28].values
    # set_target = obs_data.iloc[:1200, 14:21].values

    # Set up ROS publishers and subscribers
    pub = rospy.Publisher("/joint_reference", JointState, queue_size=10)
    rospy.Subscriber("/joint_states", JointState, joint_states_callback)
    rospy.Subscriber("/ee_pose", Pose, ee_pose_callback)
    rospy.Subscriber("/ee_vel", Twist, ee_vel_callback)

    rospy.sleep(1)

    # Clear previous data and start recording
    received_joint_pos.clear()
    received_joint_vel.clear()
    received_ee_pos.clear()
    received_ee_orient.clear()
    received_ee_lin_vel.clear()
    recording = True

    joint_state_msg = JointState()
    t = 0
    while not rospy.is_shutdown():
        if t < set_target.shape[0]:
            joint_state_msg.position = set_target[t]
            pub.publish(joint_state_msg)
            t += 1
        else:
            rospy.loginfo("Reached the end of the data.")
            break
        rate.sleep()

    recording = False

    # Construct the output directory path
    output_dir = os.path.join(script_dir, '../../output_files/ft/idealpd_9_2')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Save the outputs with headers
    save_output(np.array(received_joint_pos), os.path.join(output_dir, "ft_received_joint_pos_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
    save_output(np.array(received_joint_vel), os.path.join(output_dir, "ft_received_joint_vel_np.csv"), "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6")
    save_output(np.array(received_ee_pos), os.path.join(output_dir, "ft_received_ee_pos_np.csv"), "pos_X,pos_Y,pos_Z")
    save_output(np.array(received_ee_orient), os.path.join(output_dir, "ft_received_ee_orient_np.csv"), "or_w,or_x,or_y,or_z")
    save_output(np.array(received_ee_lin_vel), os.path.join(output_dir, "ft_received_ee_lin_vel_np.csv"), "lin_vel_X,lin_vel_Y,lin_vel_Z")

if __name__ == "__main__":
    joint_pos_test()
