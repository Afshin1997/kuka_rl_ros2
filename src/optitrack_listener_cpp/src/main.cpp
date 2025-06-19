#include <rclcpp/rclcpp.hpp>
#include "optitrack_listener_cpp/optitrack_listener.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<optitrack_listener_cpp::OptiTrackListener>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}