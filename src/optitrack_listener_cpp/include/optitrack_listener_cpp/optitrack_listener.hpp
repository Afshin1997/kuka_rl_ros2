#ifndef OPTITRACK_LISTENER_CPP_OPTITRACK_LISTENER_HPP
#define OPTITRACK_LISTENER_CPP_OPTITRACK_LISTENER_HPP

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "optitrack_listener_cpp/optitrack_receiver.hpp"

#include <thread>
#include <atomic>
#include <map>

namespace optitrack_listener_cpp {

class OptiTrackListener : public rclcpp::Node {
private:
    // Configuration parameters
    std::string fixed_frame_;
    std::string local_interface_;
    bool publish_tf_;
    std::string rigid_object_list_;
    
    // Trackable mapping
    std::map<int, std::string> id_to_name_;        // ID -> trackable name
    std::map<int, int> id_to_publisher_index_;     // ID -> publisher index
    
    // ROS publishers and TF
    std::vector<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> publishers_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // OptiTrack receiver
    std::unique_ptr<OptiTrackReceiver> receiver_;
    std::thread receiver_thread_;
    std::atomic<bool> running_;
    
    // Methods
    void load_parameters();
    void parse_rigid_bodies();
    void receiver_loop();
    
public:
    OptiTrackListener();
    ~OptiTrackListener();
};

} // namespace optitrack_listener_cpp

#endif // OPTITRACK_LISTENER_CPP_OPTITRACK_LISTENER_HPP