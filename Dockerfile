FROM osrf/ros:humble-desktop-full

# Change default shell to bash
SHELL ["/bin/bash", "-c"]

# Upgrade packages and install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    git \
    python3-pip \
    python3-scipy \
    nlohmann-json3-dev

# Install Python packages with pip
RUN pip3 install torch imageio pandas scipy optirx joblib scikit-learn seaborn "numpy<2"  

# Add non-root user
ENV HOME /home/user
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && \
    echo "user:user" | chpasswd && \
    echo "user ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

# Create workspace
USER user
RUN mkdir -p ${HOME}/kuka_rl_ros2/src
WORKDIR ${HOME}/kuka_rl_ros2/

# Copy your source code (including the submodule)
COPY --chown=user ./src ${HOME}/kuka_rl_ros2/src

# Initialize and build
USER root
RUN rosdep update && \
    rosdep install -i --from-paths src --rosdistro ${ROS_DISTRO} -y
USER user

# Build with symlink install for development
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install

# Set up environment
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${HOME}/kuka_rl_ros2/install/local_setup.bash" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=20" >> ~/.bashrc

# Clean up
USER root
RUN rm -rf /var/lib/apt/lists/*
USER user

WORKDIR ${HOME}/kuka_rl_ros2
CMD ["/bin/bash"]
