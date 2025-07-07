# cmake_minimum_required(VERSION 3.5)
# project(catch_and_throw)

# # Default to C++14
# if(NOT CMAKE_CXX_STANDARD)
#     set(CMAKE_CXX_STANDARD 14)
# endif()

# # Set Torch directory
# set(Torch_DIR "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/share/cmake/Torch")

# # Find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(lbr_fri_idl REQUIRED)
# find_package(Torch REQUIRED)
# find_package(Eigen3 REQUIRED)

# # Set CMAKE flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# # Set RPATH so executables can find PyTorch libraries at runtime
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_INSTALL_RPATH "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/lib")

# # Add and configure joint_state_node
# add_executable(joint_state_node src/joint_state_node.cpp)

# ament_target_dependencies(joint_state_node
#     rclcpp
#     sensor_msgs
#     std_msgs
#     geometry_msgs
#     lbr_fri_idl
# )

# # Link PyTorch libraries to joint_state_node
# target_link_libraries(joint_state_node
#     ${TORCH_LIBRARIES}
# )

# # Set RPATH for joint_state_node
# set_target_properties(joint_state_node PROPERTIES
#     INSTALL_RPATH "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/lib"
#     BUILD_WITH_INSTALL_RPATH TRUE
# )

# # Add and configure joint_state_node_trained_bouncing
# add_executable(joint_state_node_trained_bouncing src/joint_state_node_trained_bouncing.cpp)

# ament_target_dependencies(joint_state_node_trained_bouncing
#     rclcpp
#     sensor_msgs
#     std_msgs
#     geometry_msgs
#     lbr_fri_idl
# )

# # Link PyTorch libraries to joint_state_node_trained_bouncing
# target_link_libraries(joint_state_node_trained_bouncing
#     ${TORCH_LIBRARIES}
#     Eigen3::Eigen
# )

# # Set RPATH for joint_state_node_trained_bouncing
# set_target_properties(joint_state_node_trained_bouncing PROPERTIES
#     INSTALL_RPATH "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/lib"
#     BUILD_WITH_INSTALL_RPATH TRUE
# )

# # Install targets
# install(TARGETS joint_state_node DESTINATION lib/${PROJECT_NAME})
# install(TARGETS joint_state_node_trained_bouncing DESTINATION lib/${PROJECT_NAME})

# # Install launch files
# install(DIRECTORY
#     launch
#     DESTINATION share/${PROJECT_NAME}/
# )

# # Tests
# if(BUILD_TESTING)
#     find_package(ament_lint_auto REQUIRED)
#     ament_lint_auto_find_test_dependencies()
# endif()

# # Final ament package call
# ament_package()


cmake_minimum_required(VERSION 3.5)
project(catch_and_throw)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

# Set Torch directory
set(Torch_DIR "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/share/cmake/Torch")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lbr_fri_idl REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)

# Set CMAKE flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set RPATH so executables can find PyTorch libraries at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/lib")

# Add and configure joint_state_node_trained_bouncing
add_executable(joint_state_node_trained_bouncing src/joint_state_node_trained_bouncing.cpp src/manip_filter.cpp)
ament_target_dependencies(joint_state_node_trained_bouncing
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    lbr_fri_idl
    kdl_parser
    orocos_kdl
)

# Link PyTorch libraries to joint_state_node_trained_bouncing
target_link_libraries(joint_state_node_trained_bouncing
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)
target_link_libraries(joint_state_node_trained_bouncing nlohmann_json::nlohmann_json)

# Set RPATH for joint_state_node_trained_bouncing
set_target_properties(joint_state_node_trained_bouncing PROPERTIES
    INSTALL_RPATH "/home/user/kuka_rl_ros2/src/catch_and_throw/libs/libtorch/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install targets
install(TARGETS joint_state_node_trained_bouncing DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
)

# Install config files (NEW: This installs your YAML files)
install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}/
)

# Tests
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

# Final ament package call
ament_package()