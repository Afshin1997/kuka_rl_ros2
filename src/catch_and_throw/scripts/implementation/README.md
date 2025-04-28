# Implementation

This repository contains a ROS node script written in Python that utilizes an actor-critic neural network model to control the joint states of a robot arm. The node processes sensor data, makes predictions using a pre-trained model, and publishes joint reference commands to control the robot in real-time.

## Introduction

The scripts are designed for robotic applications that require precise joint control based on sensor inputs. It leverages a pre-trained actor-critic neural network model to compute the necessary joint positions to achieve a desired end-effector state. The node subscribes to sensor topics, processes observations, and publishes joint position commands at a frequency of 250 Hz.

## Features
- **Real-time Joint Control**
- **Neural Network Integration**
- **Observation Normalization**
- **Safety Checks**
- **Data Logging**