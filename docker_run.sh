#!/bin/bash
xhost +local:root
docker run -it --privileged \
  --name=cathc_throw_container \
  --net=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume ./src:/home/user/kuka_rl_ros2/src \
  catch_throw
