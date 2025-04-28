cmake_minimum_required(VERSION 3.5)
project(catch_and_throw)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python modules
# ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
# install(PROGRAMS
#   scripts/test.py
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install launch files if you have any
# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()