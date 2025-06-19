#include "optitrack_listener_cpp/optitrack_listener.hpp"

#include <sstream>

namespace optitrack_listener_cpp {

OptiTrackListener::OptiTrackListener() 
    : Node("optitrack_listener"), running_(true) {
    
    load_parameters();
    parse_rigid_bodies();
    
    // Initialize TF broadcaster if enabled
    if (publish_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    }
    
    // Initialize receiver
    receiver_ = std::make_unique<OptiTrackReceiver>();
    
    // Start receiver thread
    receiver_thread_ = std::thread(&OptiTrackListener::receiver_loop, this);
    
    RCLCPP_INFO(get_logger(), "OptiTrack listener initialized");
    RCLCPP_INFO(get_logger(), "Fixed frame: %s", fixed_frame_.c_str());
    RCLCPP_INFO(get_logger(), "Local interface: %s", local_interface_.c_str());
    RCLCPP_INFO(get_logger(), "Publishing TF: %s", publish_tf_ ? "yes" : "no");
}

OptiTrackListener::~OptiTrackListener() {
    running_ = false;
    
    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }
    
    RCLCPP_INFO(get_logger(), "OptiTrack listener shutdown");
}

void OptiTrackListener::load_parameters() {
    // Declare and get parameters
    declare_parameter("local_interface", "192.168.1.4");
    declare_parameter("fixed_frame", "world");
    declare_parameter("rigid_object_list", "");
    declare_parameter("publish_tf", false);
    
    get_parameter("local_interface", local_interface_);
    get_parameter("fixed_frame", fixed_frame_);
    get_parameter("rigid_object_list", rigid_object_list_);
    get_parameter("publish_tf", publish_tf_);
}

void OptiTrackListener::parse_rigid_bodies() {
    if (rigid_object_list_.empty()) {
        RCLCPP_WARN(get_logger(), "No rigid_object_list specified");
        return;
    }
    
    // Parse comma-separated list of rigid body names
    std::stringstream ss(rigid_object_list_);
    std::string name;
    int index = 0;
    
    while (std::getline(ss, name, ',')) {
        // Trim whitespace
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        
        if (!name.empty()) {
            // Declare parameters for this trackable
            declare_parameter("trackables." + name + ".id", 0);
            declare_parameter("trackables." + name + ".name", "trackable");
            
            int id = get_parameter("trackables." + name + ".id").as_int();
            std::string trackable_name = get_parameter("trackables." + name + ".name").as_string();
            
            RCLCPP_INFO(get_logger(), "Configured: %s -> ID: %d, Name: %s", 
                        name.c_str(), id, trackable_name.c_str());
            
            // Store mappings
            id_to_name_[id] = trackable_name;
            id_to_publisher_index_[id] = index;
            
            // Create publisher
            auto publisher = create_publisher<geometry_msgs::msg::PoseStamped>(
                "optitrack/" + trackable_name, 1);
            publishers_.push_back(publisher);
            
            index++;
        }
    }
    
    RCLCPP_INFO(get_logger(), "Configured %zu trackables", publishers_.size());
}

void OptiTrackListener::receiver_loop() {
    if (!receiver_->connect()) {
        RCLCPP_ERROR(get_logger(), "Failed to connect to OptiTrack");
        return;
    }
    
    RCLCPP_INFO(get_logger(), "Connected to OptiTrack multicast stream");
    
    OptiTrackPacket packet;
    bool first_packet = true;
    
    while (running_.load() && rclcpp::ok()) {
        if (!receiver_->receive_packet(packet)) {
            continue;
        }
        
        // Handle sender data to get version
        if (packet.type == OptiTrackPacket::SENDER_DATA && first_packet) {
            RCLCPP_INFO(get_logger(), "Received NatNet version info");
            first_packet = false;
            continue;
        }
        // Process frame data
        if (packet.type == OptiTrackPacket::FRAME_DATA) {
            auto current_time = this->now();
            for (const auto& rigid_body : packet.rigid_bodies) {
                // Check if we're tracking this body
                auto it = id_to_publisher_index_.find(rigid_body.id);
                if (it == id_to_publisher_index_.end()) {
                    RCLCPP_WARN_ONCE(get_logger(), "Received data for unconfigured body ID: %d", rigid_body.id);
                    continue;
                }
                
                // Create pose message
                geometry_msgs::msg::PoseStamped pose_msg;
                pose_msg.header.stamp = current_time;
                pose_msg.header.frame_id = fixed_frame_;
                
                // Set position
                pose_msg.pose.position.x = rigid_body.position[0];
                pose_msg.pose.position.y = rigid_body.position[1];
                pose_msg.pose.position.z = rigid_body.position[2];
                
                // Set orientation (quaternion)
                pose_msg.pose.orientation.x = rigid_body.orientation[0];
                pose_msg.pose.orientation.y = rigid_body.orientation[1];
                pose_msg.pose.orientation.z = rigid_body.orientation[2];
                pose_msg.pose.orientation.w = rigid_body.orientation[3];
                
                // Publish pose
                publishers_[it->second]->publish(pose_msg);
                
                // Publish TF if enabled
                if (publish_tf_) {
                    geometry_msgs::msg::TransformStamped transform;
                    transform.header = pose_msg.header;
                    transform.child_frame_id = id_to_name_[rigid_body.id];
                    transform.transform.translation.x = pose_msg.pose.position.x;
                    transform.transform.translation.y = pose_msg.pose.position.y;
                    transform.transform.translation.z = pose_msg.pose.position.z;
                    transform.transform.rotation = pose_msg.pose.orientation;
                    
                    tf_broadcaster_->sendTransform(transform);
                }
            }
        }
    }
    
    receiver_->disconnect();
}

} // namespace optitrack_listener_cpp