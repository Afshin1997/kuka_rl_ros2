#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>  // Added for OptiTrack
#include <lbr_fri_idl/msg/lbr_joint_position_command.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <mutex>  // Added for thread-safe access

using std::placeholders::_1;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

class JointStateNode : public rclcpp::Node {
public:
    JointStateNode()
    : Node("joint_state_node"), recording_(false) {
        using QoS = rclcpp::QoS;
        auto qos = QoS(rclcpp::KeepLast(1)).reliable();

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/lbr/joint_states", qos, std::bind(&JointStateNode::joint_states_callback, this, _1));

        ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/lbr/state/pose", qos, std::bind(&JointStateNode::ee_pose_callback, this, _1));

        ee_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/lbr/state/twist", qos, std::bind(&JointStateNode::ee_vel_callback, this, _1));

        // OptiTrack ball position subscription
        ball_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/optitrack/ball_marker", qos, std::bind(&JointStateNode::ball_pose_callback, this, _1));

        joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
            "/lbr/command/joint_position", 10);
            
        RCLCPP_INFO(this->get_logger(), "Node initialized with OptiTrack ball tracking (no filtering)");
    }

    void publish_joint_commands(const std::vector<std::vector<double>> &target_positions, int rate_hz = 100) {
        recording_ = true;
        rclcpp::Rate rate(rate_hz);
        
        RCLCPP_INFO(this->get_logger(), "Starting to publish raw joint commands...");
        
        for (const auto &joint_pos : target_positions) {
            // Store the original commands for analysis
            published_commands_.push_back(joint_pos);
            
            // Create and publish message directly with raw data
            auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
            std::copy(joint_pos.begin(), joint_pos.end(), msg.joint_position.begin());
            joint_ref_pub_->publish(msg);
            
            // Log current ball position periodically
            if (published_commands_.size() % 100 == 0) {  // Log every 100 iterations
                std::lock_guard<std::mutex> lock(ball_mutex_);
                RCLCPP_INFO(this->get_logger(), "Ball position: x=%.3f, y=%.3f, z=%.3f", 
                           current_ball_pos_.x, current_ball_pos_.y, current_ball_pos_.z);
            }
            
            rate.sleep();
        }

        recording_ = false;
        RCLCPP_INFO(this->get_logger(), "Finished publishing all raw targets");
    }

    void save_data(const std::string &output_dir) {
        fs::create_directories(output_dir);
        
        save_vector(received_joint_pos_, output_dir + "/ft_received_joint_pos_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
        save_vector(received_joint_vel_, output_dir + "/ft_received_joint_vel_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
        save_vector(received_joint_eff_, output_dir + "/ft_received_joint_effort_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
        save_vector(received_ee_pos_, output_dir + "/ft_received_ee_pos_np.csv", 
                    "pos_X,pos_Y,pos_Z");
        
        save_vector(received_ee_orient_, output_dir + "/ft_received_ee_orient_np.csv", 
                    "or_w,or_x,or_y,or_z");
        
        save_vector(received_ee_lin_vel_, output_dir + "/ft_received_ee_lin_vel_np.csv", 
                    "lin_vel_X,lin_vel_Y,lin_vel_Z");
        
        // Save the published commands
        save_vector(published_commands_, output_dir + "/ft_published_commands_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
        // Save ball position data
        save_vector(received_ball_pos_, output_dir + "/ft_ball_positions_np.csv", 
                    "ball_x,ball_y,ball_z");
        
        save_vector(received_ball_orient_, output_dir + "/ft_ball_orientations_np.csv", 
                    "ball_qw,ball_qx,ball_qy,ball_qz");
    }
    
    // Get current ball position (thread-safe)
    geometry_msgs::msg::Point get_current_ball_position() {
        std::lock_guard<std::mutex> lock(ball_mutex_);
        return current_ball_pos_;
    }
    
    // Get current ball orientation (thread-safe)
    geometry_msgs::msg::Quaternion get_current_ball_orientation() {
        std::lock_guard<std::mutex> lock(ball_mutex_);
        return current_ball_orient_;
    }

private:
    bool recording_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr ee_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ee_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ball_pose_sub_;  // OptiTrack subscription
    rclcpp::Publisher<lbr_fri_idl::msg::LBRJointPositionCommand>::SharedPtr joint_ref_pub_;

    std::vector<std::vector<double>> received_joint_pos_, received_joint_vel_, received_joint_eff_;
    std::vector<std::vector<double>> received_ee_pos_, received_ee_orient_, received_ee_lin_vel_;
    std::vector<std::vector<double>> published_commands_;
    std::vector<std::vector<double>> received_ball_pos_, received_ball_orient_;  // Ball tracking data
    
    // Current ball state (thread-safe access)
    std::mutex ball_mutex_;
    geometry_msgs::msg::Point current_ball_pos_;
    geometry_msgs::msg::Quaternion current_ball_orient_;

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (!recording_) return;
        std::vector<std::string> desired_order = {"lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7"};
        std::vector<double> pos(7), vel(7), eff(7);
        for (size_t i = 0; i < desired_order.size(); ++i) {
            auto it = std::find(msg->name.begin(), msg->name.end(), desired_order[i]);
            if (it != msg->name.end()) {
                size_t idx = std::distance(msg->name.begin(), it);
                pos[i] = msg->position[idx];
                vel[i] = msg->velocity[idx];
                eff[i] = msg->effort[idx];
            }
        }
        received_joint_pos_.push_back(pos);
        received_joint_vel_.push_back(vel);
        received_joint_eff_.push_back(eff);
    }

    void ee_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        if (!recording_) return;
        received_ee_pos_.push_back({msg->position.x, msg->position.y, msg->position.z});
        received_ee_orient_.push_back({msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z});
    }

    void ee_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        if (!recording_) return;
        received_ee_lin_vel_.push_back({msg->linear.x, msg->linear.y, msg->linear.z});
    }
    
    // OptiTrack ball pose callback
    void ball_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // Update current ball position (thread-safe)
        {
            std::lock_guard<std::mutex> lock(ball_mutex_);
            current_ball_pos_ = msg->pose.position;
            current_ball_orient_ = msg->pose.orientation;
        }
        
        // Record ball data if recording is active
        if (recording_) {
            received_ball_pos_.push_back({msg->pose.position.x, msg->pose.position.y, msg->pose.position.z});
            received_ball_orient_.push_back({msg->pose.orientation.w, msg->pose.orientation.x, 
                                           msg->pose.orientation.y, msg->pose.orientation.z});
        }
    }

    void save_vector(const std::vector<std::vector<double>> &data, 
                    const std::string &filename, 
                    const std::string &header) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write header
        file << header << "\n";

        // Write data
        for (const auto &row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i != row.size() - 1)
                    file << ",";
            }
            file << "\n";
        }

        file.close();
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointStateNode>();

    // Example: Load a hardcoded file
    std::string input_csv = "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/ft_idealpd.csv"; // Replace with real path
    std::ifstream file(input_csv);
    if (!file.is_open()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to open file: %s", input_csv.c_str());
        return 1;
    }
    
    std::vector<std::vector<double>> joint_data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;
        int col = 0;
        while (std::getline(ss, cell, ',')) {
            if (col >= 14 && col < 21) {
                try {
                    // Trim whitespace
                    cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
                    if (!cell.empty()) {
                        row.push_back(std::stod(cell));
                    }
                } catch (const std::exception &e) {
                    RCLCPP_ERROR(node->get_logger(), "Error parsing column %d: %s", col, e.what());
                }
            }
            col++;
        }
        if (row.size() == 7) {
            joint_data.push_back(row);
        }
    }

    double sample_rate_hz = 100.0;  // Control rate

    std::thread pub_thread([&]() {
        // Wait a moment to ensure OptiTrack subscription is active
        std::this_thread::sleep_for(1s);
        
        // Get initial ball position
        auto initial_ball_pos = node->get_current_ball_position();
        RCLCPP_INFO(node->get_logger(), "Initial ball position: x=%.3f, y=%.3f, z=%.3f", 
                   initial_ball_pos.x, initial_ball_pos.y, initial_ball_pos.z);
        
        node->publish_joint_commands(joint_data, sample_rate_hz);
        node->save_data("/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/ft/idealpd");  // Replace with your output dir
        rclcpp::shutdown();
    });

    rclcpp::spin(node);
    pub_thread.join();
    return 0;
}