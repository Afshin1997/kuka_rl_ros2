// #include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <geometry_msgs/msg/pose.hpp>
// #include <geometry_msgs/msg/twist.hpp>
// #include <lbr_fri_idl/msg/lbr_joint_position_command.hpp>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <chrono>
// #include <filesystem>

// using std::placeholders::_1;
// using namespace std::chrono_literals;
// namespace fs = std::filesystem;

// class JointStateNode : public rclcpp::Node {
// public:
//     JointStateNode()
//     : Node("joint_state_node"), recording_(false) {
//         using QoS = rclcpp::QoS;
//         auto qos = QoS(rclcpp::KeepLast(1)).reliable();

//         joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
//             "/lbr/joint_states", qos, std::bind(&JointStateNode::joint_states_callback, this, _1));

//         ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
//             "/lbr/state/pose", qos, std::bind(&JointStateNode::ee_pose_callback, this, _1));

//         ee_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
//             "/lbr/state/twist", qos, std::bind(&JointStateNode::ee_vel_callback, this, _1));

//         joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
//             "/lbr/command/joint_position", 10);
//     }

//     void publish_joint_commands(const std::vector<std::vector<double>> &target_positions, int rate_hz = 200) {
//         recording_ = true;
//         rclcpp::Rate rate(rate_hz);

//         for (const auto &joint_pos : target_positions) {
//             auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
//             std::copy(joint_pos.begin(), joint_pos.end(), msg.joint_position.begin());
//             joint_ref_pub_->publish(msg);
//             rate.sleep();
//         }

//         recording_ = false;
//         RCLCPP_INFO(this->get_logger(), "Finished publishing all targets");
//     }

//     void save_data(const std::string &output_dir) {
//         fs::create_directories(output_dir);
        
//         save_vector(received_joint_pos_, output_dir + "/ft_received_joint_pos_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_joint_vel_, output_dir + "/ft_received_joint_vel_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_joint_eff_, output_dir + "/ft_received_joint_effort_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_ee_pos_, output_dir + "/ft_received_ee_pos_np.csv", 
//                     "pos_X,pos_Y,pos_Z");
        
//         save_vector(received_ee_orient_, output_dir + "/ft_received_ee_orient_np.csv", 
//                     "or_w,or_x,or_y,or_z");
        
//         save_vector(received_ee_lin_vel_, output_dir + "/ft_received_ee_lin_vel_np.csv", 
//                     "lin_vel_X,lin_vel_Y,lin_vel_Z");
//     }
    

// private:
//     bool recording_;
//     rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr ee_pose_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ee_vel_sub_;
//     rclcpp::Publisher<lbr_fri_idl::msg::LBRJointPositionCommand>::SharedPtr joint_ref_pub_;

//     std::vector<std::vector<double>> received_joint_pos_, received_joint_vel_, received_joint_eff_;
//     std::vector<std::vector<double>> received_ee_pos_, received_ee_orient_, received_ee_lin_vel_;

//     void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
//         if (!recording_) return;
//         std::vector<std::string> desired_order = {"lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7"};
//         std::vector<double> pos(7), vel(7), eff(7);
//         for (size_t i = 0; i < desired_order.size(); ++i) {
//             auto it = std::find(msg->name.begin(), msg->name.end(), desired_order[i]);
//             if (it != msg->name.end()) {
//                 size_t idx = std::distance(msg->name.begin(), it);
//                 pos[i] = msg->position[idx];
//                 vel[i] = msg->velocity[idx];
//                 eff[i] = msg->effort[idx];
//             }
//         }
//         received_joint_pos_.push_back(pos);
//         received_joint_vel_.push_back(vel);
//         received_joint_eff_.push_back(eff);
//     }

//     void ee_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
//         if (!recording_) return;
//         received_ee_pos_.push_back({msg->position.x, msg->position.y, msg->position.z});
//         received_ee_orient_.push_back({msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z});
//     }

//     void ee_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
//         if (!recording_) return;
//         received_ee_lin_vel_.push_back({msg->linear.x, msg->linear.y, msg->linear.z});
//     }

//     void save_vector(const std::vector<std::vector<double>> &data, 
//                     const std::string &filename, 
//                     const std::string &header) {
//         std::ofstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Failed to open file: " << filename << std::endl;
//             return;
//         }

//         // Write header
//         file << header << "\n";

//         // Write data
//         for (const auto &row : data) {
//             for (size_t i = 0; i < row.size(); ++i) {
//                 file << row[i];
//                 if (i != row.size() - 1)
//                     file << ",";
//             }
//             file << "\n";
//         }

//         file.close();
//     }


// };

// int main(int argc, char *argv[]) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<JointStateNode>();

//     // Example: Load a hardcoded file
//     std::string input_csv = "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/ft_idealpd.csv"; // Replace with real path
//     std::ifstream file(input_csv);
//     if (!file.is_open()) {
//         RCLCPP_ERROR(node->get_logger(), "Failed to open file: %s", input_csv.c_str());
//         return 1;
//     }
//     // std::getline(file, line);
//     std::vector<std::vector<double>> joint_data;
//     std::string line;

//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::vector<double> row;
//         std::string cell;
//         int col = 0;
//         while (std::getline(ss, cell, ',')) {
//             if (col >= 14 && col < 21) {
//                 try {
//                     // Trim whitespace
//                     cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
//                     if (!cell.empty()) {
//                         row.push_back(std::stod(cell));
//                     }
//                 } catch (const std::exception &e) {
//                     RCLCPP_ERROR(node->get_logger(), "Error parsing column %d: %s", col, e.what());
//                 }
//             }
//             col++;
//         }
//         if (row.size() == 7) {
//             joint_data.push_back(row);
//         }
//     }

//     std::thread pub_thread([&]() {
//         node->publish_joint_commands(joint_data);
//         node->save_data("/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/ft/idealpd_2");  // Replace with your output dir
//         rclcpp::shutdown();
//     });

//     rclcpp::spin(node);
//     pub_thread.join();
//     return 0;
// }

// #include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <geometry_msgs/msg/pose.hpp>
// #include <geometry_msgs/msg/twist.hpp>
// #include <lbr_fri_idl/msg/lbr_joint_position_command.hpp>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <chrono>
// #include <filesystem>
// #include <cmath>
// #include <torch/torch.h>

// using std::placeholders::_1;
// using namespace std::chrono_literals;
// namespace fs = std::filesystem;

// // Low Pass Filter implementation
// class LowPassFilter {
// public:
//     LowPassFilter(double cutoff_hz, double sample_rate_hz, torch::Tensor initial_state = torch::Tensor()) {
//         double dt = 1.0 / sample_rate_hz;
//         alpha_ = std::exp(-2.0 * M_PI * cutoff_hz * dt);
        
//         if (initial_state.numel() == 0) {
//             // Default to scalar
//             prev_ = torch::zeros(1);
//         } else {
//             // Clone to ensure we have our own copy
//             prev_ = initial_state.clone().to(torch::kFloat64);
//         }
//     }
    
//     torch::Tensor operator()(const torch::Tensor& x) {
//         auto x_float = x.to(torch::kFloat64);
//         auto y = alpha_ * prev_ + (1.0 - alpha_) * x_float;
//         prev_ = y.clone(); // Store a copy to avoid reference issues
//         return y;
//     }
    
// private:
//     double alpha_;
//     torch::Tensor prev_;
// };

// class JointStateNode : public rclcpp::Node {
// public:
//     JointStateNode()
//     : Node("joint_state_node"), recording_(false) {
//         using QoS = rclcpp::QoS;
//         auto qos = QoS(rclcpp::KeepLast(1)).reliable();

//         joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
//             "/lbr/joint_states", qos, std::bind(&JointStateNode::joint_states_callback, this, _1));

//         ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
//             "/lbr/state/pose", qos, std::bind(&JointStateNode::ee_pose_callback, this, _1));

//         ee_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
//             "/lbr/state/twist", qos, std::bind(&JointStateNode::ee_vel_callback, this, _1));

//         joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
//             "/lbr/command/joint_position", 10);
//     }

//     // Initialize the low-pass filter with proper parameters
//     void init_filter(double cutoff_hz, double sample_rate_hz, const std::vector<double>& initial_state = {}) {
//         torch::Tensor initial;
//         if (!initial_state.empty()) {
//             initial = torch::tensor(initial_state).to(torch::kFloat64);
//         }
//         filter_ = std::make_unique<LowPassFilter>(cutoff_hz, sample_rate_hz, initial);
//         RCLCPP_INFO(this->get_logger(), "Low-pass filter initialized with cutoff: %.2f Hz, sample rate: %.2f Hz", 
//                    cutoff_hz, sample_rate_hz);
//     }

//     void publish_joint_commands(const std::vector<std::vector<double>> &target_positions, int rate_hz = 200) {
//         recording_ = true;
//         rclcpp::Rate rate(rate_hz);
        
//         // Initialize filter if not already done
//         if (!filter_) {
//             if (!target_positions.empty()) {
//                 init_filter(10.0, rate_hz, target_positions[0]); // Default 10Hz cutoff if not specified
//             } else {
//                 RCLCPP_ERROR(this->get_logger(), "Empty target positions, cannot initialize filter");
//                 return;
//             }
//         }
        
//         RCLCPP_INFO(this->get_logger(), "Starting to publish filtered joint commands...");
        
//         for (const auto &joint_pos : target_positions) {
//             // Apply the filter to the joint positions
//             auto tensor_input = torch::tensor(joint_pos).to(torch::kFloat64);
//             auto filtered_tensor = (*filter_)(tensor_input);
            
//             // Convert filtered tensor back to std::vector
//             auto filtered_pos = std::vector<double>(filtered_tensor.data_ptr<double>(), 
//                                                  filtered_tensor.data_ptr<double>() + filtered_tensor.numel());
            
//             // Store both original and filtered positions for analysis
//             original_commands_.push_back(joint_pos);
//             filtered_commands_.push_back(filtered_pos);
            
//             // Create and publish message
//             auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
//             std::copy(filtered_pos.begin(), filtered_pos.end(), msg.joint_position.begin());
//             // joint_ref_pub_->publish(msg);
            
//             rate.sleep();
//         }

//         recording_ = false;
//         RCLCPP_INFO(this->get_logger(), "Finished publishing all filtered targets");
//     }

//     void save_data(const std::string &output_dir) {
//         fs::create_directories(output_dir);
        
//         save_vector(received_joint_pos_, output_dir + "/ft_received_joint_pos_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_joint_vel_, output_dir + "/ft_received_joint_vel_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_joint_eff_, output_dir + "/ft_received_joint_effort_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_ee_pos_, output_dir + "/ft_received_ee_pos_np.csv", 
//                     "pos_X,pos_Y,pos_Z");
        
//         save_vector(received_ee_orient_, output_dir + "/ft_received_ee_orient_np.csv", 
//                     "or_w,or_x,or_y,or_z");
        
//         save_vector(received_ee_lin_vel_, output_dir + "/ft_received_ee_lin_vel_np.csv", 
//                     "lin_vel_X,lin_vel_Y,lin_vel_Z");
        
//         // Save the original and filtered commands for comparison
//         save_vector(original_commands_, output_dir + "/ft_original_commands_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(filtered_commands_, output_dir + "/ft_filtered_commands_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
//     }
    

// private:
//     bool recording_;
//     rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr ee_pose_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ee_vel_sub_;
//     rclcpp::Publisher<lbr_fri_idl::msg::LBRJointPositionCommand>::SharedPtr joint_ref_pub_;
//     std::unique_ptr<LowPassFilter> filter_;

//     std::vector<std::vector<double>> received_joint_pos_, received_joint_vel_, received_joint_eff_;
//     std::vector<std::vector<double>> received_ee_pos_, received_ee_orient_, received_ee_lin_vel_;
//     std::vector<std::vector<double>> original_commands_, filtered_commands_;

//     void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
//         if (!recording_) return;
//         std::vector<std::string> desired_order = {"lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7"};
//         std::vector<double> pos(7), vel(7), eff(7);
//         for (size_t i = 0; i < desired_order.size(); ++i) {
//             auto it = std::find(msg->name.begin(), msg->name.end(), desired_order[i]);
//             if (it != msg->name.end()) {
//                 size_t idx = std::distance(msg->name.begin(), it);
//                 pos[i] = msg->position[idx];
//                 vel[i] = msg->velocity[idx];
//                 eff[i] = msg->effort[idx];
//             }
//         }
//         received_joint_pos_.push_back(pos);
//         received_joint_vel_.push_back(vel);
//         received_joint_eff_.push_back(eff);
//     }

//     void ee_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
//         if (!recording_) return;
//         received_ee_pos_.push_back({msg->position.x, msg->position.y, msg->position.z});
//         received_ee_orient_.push_back({msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z});
//     }

//     void ee_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
//         if (!recording_) return;
//         received_ee_lin_vel_.push_back({msg->linear.x, msg->linear.y, msg->linear.z});
//     }

//     void save_vector(const std::vector<std::vector<double>> &data, 
//                     const std::string &filename, 
//                     const std::string &header) {
//         std::ofstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Failed to open file: " << filename << std::endl;
//             return;
//         }

//         // Write header
//         file << header << "\n";

//         // Write data
//         for (const auto &row : data) {
//             for (size_t i = 0; i < row.size(); ++i) {
//                 file << row[i];
//                 if (i != row.size() - 1)
//                     file << ",";
//             }
//             file << "\n";
//         }

//         file.close();
//     }
// };

// int main(int argc, char *argv[]) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<JointStateNode>();

//     // Example: Load a hardcoded file
//     std::string input_csv = "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/ft_idealpd.csv"; // Replace with real path
//     std::ifstream file(input_csv);
//     if (!file.is_open()) {
//         RCLCPP_ERROR(node->get_logger(), "Failed to open file: %s", input_csv.c_str());
//         return 1;
//     }
    
//     std::vector<std::vector<double>> joint_data;
//     std::string line;

//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::vector<double> row;
//         std::string cell;
//         int col = 0;
//         while (std::getline(ss, cell, ',')) {
//             if (col >= 21 && col < 28) {
//                 try {
//                     // Trim whitespace
//                     cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
//                     if (!cell.empty()) {
//                         row.push_back(std::stod(cell));
//                     }
//                 } catch (const std::exception &e) {
//                     RCLCPP_ERROR(node->get_logger(), "Error parsing column %d: %s", col, e.what());
//                 }
//             }
//             col++;
//         }
//         if (row.size() == 7) {
//             joint_data.push_back(row);
//         }
//     }

//     // Initialize the low-pass filter with cutoff frequency and sample rate
//     // You can adjust the cutoff frequency to change filtering behavior
//     double cutoff_hz = 5.0;  // 5 Hz cutoff frequency
//     double sample_rate_hz = 200.0;  // Assuming 200 Hz control rate
    
//     if (!joint_data.empty()) {
//         node->init_filter(cutoff_hz, sample_rate_hz, joint_data[0]);
//     }

//     std::thread pub_thread([&]() {
//         node->publish_joint_commands(joint_data, sample_rate_hz);
//         node->save_data("/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/ft/idealpd_6");  // Replace with your output dir
//         rclcpp::shutdown();
//     });

//     rclcpp::spin(node);
//     pub_thread.join();
//     return 0;
// }


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
#include <torch/torch.h>
#include <mutex>  // Added for thread-safe access

using std::placeholders::_1;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Low Pass Filter implementation
class LowPassFilter {
public:
    LowPassFilter(double cutoff_hz, double sample_rate_hz, torch::Tensor initial_state = torch::Tensor()) {
        double dt = 1.0 / sample_rate_hz;
        alpha_ = std::exp(-2.0 * M_PI * cutoff_hz * dt);
        
        if (initial_state.numel() == 0) {
            // Default to scalar
            prev_ = torch::zeros(1);
        } else {
            // Clone to ensure we have our own copy
            prev_ = initial_state.clone().to(torch::kFloat64);
        }
    }
    
    torch::Tensor operator()(const torch::Tensor& x) {
        auto x_float = x.to(torch::kFloat64);
        auto y = alpha_ * prev_ + (1.0 - alpha_) * x_float;
        prev_ = y.clone(); // Store a copy to avoid reference issues
        return y;
    }
    
private:
    double alpha_;
    torch::Tensor prev_;
};

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
            "/optitrack/ball", qos, std::bind(&JointStateNode::ball_pose_callback, this, _1));

        joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
            "/lbr/command/joint_position", 10);
            
        RCLCPP_INFO(this->get_logger(), "Node initialized with OptiTrack ball tracking");
    }

    // Initialize the low-pass filter with proper parameters
    void init_filter(double cutoff_hz, double sample_rate_hz, const std::vector<double>& initial_state = {}) {
        torch::Tensor initial;
        if (!initial_state.empty()) {
            initial = torch::tensor(initial_state).to(torch::kFloat64);
        }
        filter_ = std::make_unique<LowPassFilter>(cutoff_hz, sample_rate_hz, initial);
        RCLCPP_INFO(this->get_logger(), "Low-pass filter initialized with cutoff: %.2f Hz, sample rate: %.2f Hz", 
                   cutoff_hz, sample_rate_hz);
    }

    void publish_joint_commands(const std::vector<std::vector<double>> &target_positions, int rate_hz = 200) {
        recording_ = true;
        rclcpp::Rate rate(rate_hz);
        
        // Initialize filter if not already done
        if (!filter_) {
            if (!target_positions.empty()) {
                init_filter(10.0, rate_hz, target_positions[0]); // Default 10Hz cutoff if not specified
            } else {
                RCLCPP_ERROR(this->get_logger(), "Empty target positions, cannot initialize filter");
                return;
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Starting to publish filtered joint commands...");
        
        for (const auto &joint_pos : target_positions) {
            // Apply the filter to the joint positions
            auto tensor_input = torch::tensor(joint_pos).to(torch::kFloat64);
            auto filtered_tensor = (*filter_)(tensor_input);
            
            // Convert filtered tensor back to std::vector
            auto filtered_pos = std::vector<double>(filtered_tensor.data_ptr<double>(), 
                                                 filtered_tensor.data_ptr<double>() + filtered_tensor.numel());
            
            // Store both original and filtered positions for analysis
            original_commands_.push_back(joint_pos);
            filtered_commands_.push_back(filtered_pos);
            
            // Create and publish message
            auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
            std::copy(filtered_pos.begin(), filtered_pos.end(), msg.joint_position.begin());
            joint_ref_pub_->publish(msg);
            
            // Log current ball position periodically
            if (filtered_commands_.size() % 10 == 0) {  // Log every 100 iterations
                std::lock_guard<std::mutex> lock(ball_mutex_);
                RCLCPP_INFO(this->get_logger(), "Ball position: x=%.3f, y=%.3f, z=%.3f", 
                           current_ball_pos_.x, current_ball_pos_.y, current_ball_pos_.z);
            }
            
            rate.sleep();
        }

        recording_ = false;
        RCLCPP_INFO(this->get_logger(), "Finished publishing all filtered targets");
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
        
        // Save the original and filtered commands for comparison
        save_vector(original_commands_, output_dir + "/ft_original_commands_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
        save_vector(filtered_commands_, output_dir + "/ft_filtered_commands_np.csv", 
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
    std::unique_ptr<LowPassFilter> filter_;

    std::vector<std::vector<double>> received_joint_pos_, received_joint_vel_, received_joint_eff_;
    std::vector<std::vector<double>> received_ee_pos_, received_ee_orient_, received_ee_lin_vel_;
    std::vector<std::vector<double>> original_commands_, filtered_commands_;
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

    // Initialize the low-pass filter with cutoff frequency and sample rate
    // You can adjust the cutoff frequency to change filtering behavior
    double cutoff_hz = 5.0;  // 5 Hz cutoff frequency
    double sample_rate_hz = 200.0;  // Assuming 200 Hz control rate
    
    if (!joint_data.empty()) {
        node->init_filter(cutoff_hz, sample_rate_hz, joint_data[0]);
    }

    std::thread pub_thread([&]() {
        // Wait a moment to ensure OptiTrack subscription is active
        std::this_thread::sleep_for(1s);
        
        // Get initial ball position
        auto initial_ball_pos = node->get_current_ball_position();
        RCLCPP_INFO(node->get_logger(), "Initial ball position: x=%.3f, y=%.3f, z=%.3f", 
                   initial_ball_pos.x, initial_ball_pos.y, initial_ball_pos.z);
        
        node->publish_joint_commands(joint_data, sample_rate_hz);
        node->save_data("/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/ft/idealpd_14");  // Replace with your output dir
        rclcpp::shutdown();
    });

    rclcpp::spin(node);
    pub_thread.join();
    return 0;
}


// #include <rclcpp/rclcpp.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <geometry_msgs/msg/pose.hpp>
// #include <geometry_msgs/msg/twist.hpp>
// #include <geometry_msgs/msg/pose_stamped.hpp>
// #include <lbr_fri_idl/msg/lbr_joint_position_command.hpp>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <chrono>
// #include <filesystem>
// #include <cmath>
// #include <torch/torch.h>
// #include <mutex>
// #include <algorithm>
// #include <sstream>

// using std::placeholders::_1;
// using namespace std::chrono_literals;
// namespace fs = std::filesystem;

// // Savitzky-Golay Filter implementation for responsive joint command filtering
// class RealTimeSavitzkyGolay {
// public:
//     RealTimeSavitzkyGolay(int window_length = 11, 
//                           int polyorder = 3, 
//                           int deriv = 0, 
//                           double delta = 1.0,
//                           torch::Tensor initial_values = torch::Tensor()) {
        
//         // Ensure window length is odd and reasonable
//         if (window_length % 2 == 0) {
//             window_length += 1;
//             std::cout << "Warning: window_length adjusted to odd value: " << window_length << std::endl;
//         }
        
//         // Ensure minimum window length
//         if (window_length < 5) {
//             window_length = 5;
//             std::cout << "Warning: window_length too small, set to 5" << std::endl;
//         }
        
//         // Check polyorder against window length
//         if (polyorder >= window_length) {
//             polyorder = window_length - 1;
//             std::cout << "Warning: polyorder too large, reduced to " << polyorder << std::endl;
//         }
        
//         window_length_ = window_length;
//         polyorder_ = polyorder;
        
//         std::cout << "Initializing SG filter with window_length=" << window_length_ 
//                   << ", polyorder=" << polyorder_ << std::endl;
        
//         // Precompute coefficients
//         coeffs_ = compute_savgol_coeffs_smooth(window_length_, polyorder_);
        
//         // Initialize buffer
//         if (initial_values.numel() > 0) {
//             bool is_batched = initial_values.dim() > 1;
            
//             if (is_batched) {
//                 if (initial_values.size(0) >= window_length_) {
//                     buffer_ = initial_values.slice(0, initial_values.size(0) - window_length_, initial_values.size(0)).clone();
//                 } else {
//                     auto first_value = initial_values[0].unsqueeze(0);
//                     auto padding = first_value.repeat({window_length_ - initial_values.size(0), 1});
//                     buffer_ = torch::cat({padding, initial_values}, 0);
//                 }
//             } else {
//                 if (initial_values.size(0) >= window_length_) {
//                     buffer_ = initial_values.slice(0, initial_values.size(0) - window_length_, initial_values.size(0)).clone();
//                 } else {
//                     float first_val = initial_values.size(0) > 0 ? initial_values[0].item<float>() : 0.0f;
//                     auto padding = torch::full({window_length_ - initial_values.size(0)}, first_val);
//                     buffer_ = torch::cat({padding, initial_values}, 0);
//                 }
//             }
//         } else {
//             buffer_ = torch::zeros({window_length_});
//             buffer_initialized_ = false;
//         }
        
//         current_idx_ = 0;
//     }
    
//     torch::Tensor operator()(const torch::Tensor& x) {
//         // Initialize buffer properly on first call if not initialized
//         if (!buffer_initialized_) {
//             if (x.dim() > 0 && x.size(0) > 1) {
//                 buffer_ = torch::zeros({window_length_, x.size(0)}).to(x.device(), x.dtype());
//             }
//             buffer_initialized_ = true;
//         }
        
//         // Update buffer with new value (circular buffer approach)
//         if (x.dim() == 0) {
//             buffer_[current_idx_] = x.item<float>();
//         } else {
//             buffer_[current_idx_] = x;
//         }
        
//         current_idx_ = (current_idx_ + 1) % window_length_;
        
//         // Rearrange buffer to get correct temporal order for filtering
//         torch::Tensor ordered_buffer;
//         if (current_idx_ == 0) {
//             ordered_buffer = buffer_;
//         } else {
//             ordered_buffer = torch::cat({
//                 buffer_.slice(0, current_idx_, window_length_),
//                 buffer_.slice(0, 0, current_idx_)
//             }, 0);
//         }
        
//         // Apply precomputed coefficients to get filtered value
//         torch::Tensor filtered_value;
        
//         if (ordered_buffer.dim() == 1) {
//             filtered_value = (ordered_buffer * coeffs_).sum();
//         } else {
//             filtered_value = torch::zeros_like(x);
//             for (int i = 0; i < ordered_buffer.size(1); i++) {
//                 filtered_value[i] = (ordered_buffer.select(1, i) * coeffs_).sum();
//             }
//         }
        
//         return filtered_value;
//     }
    
// private:
//     int window_length_;
//     int polyorder_;
//     torch::Tensor coeffs_;
//     torch::Tensor buffer_;
//     int current_idx_;
//     bool buffer_initialized_ = true;
    
//     torch::Tensor compute_savgol_coeffs_smooth(int window_length, int polyorder) {
//         int half_window = window_length / 2;
        
//         std::cout << "Computing SG coeffs with window_length=" << window_length 
//                   << ", polyorder=" << polyorder << ", half_window=" << half_window << std::endl;
        
//         auto options = torch::TensorOptions().dtype(torch::kFloat64);  // Use double precision
//         torch::Tensor x = torch::arange(-half_window, half_window + 1, options);
//         torch::Tensor vander = torch::zeros({window_length, polyorder + 1}, options);
        
//         // Build Vandermonde matrix
//         for (int i = 0; i <= polyorder; i++) {
//             vander.select(1, i) = torch::pow(x, i);
//         }
        
//         std::cout << "Vandermonde matrix shape: " << vander.sizes() << std::endl;
        
//         // Use normal equations approach: (V^T V) coeffs = V^T b
//         // where b is a unit vector selecting the center point
//         torch::Tensor b = torch::zeros({window_length}, options);
//         b[half_window] = 1.0;  // Center point
        
//         // Compute V^T V and V^T b
//         torch::Tensor vtv = torch::matmul(vander.transpose(0, 1), vander);
//         torch::Tensor vtb = torch::matmul(vander.transpose(0, 1), b);
        
//         std::cout << "VTV shape: " << vtv.sizes() << ", VTB shape: " << vtb.sizes() << std::endl;
        
//         // Solve the system (V^T V) coeffs = V^T b
//         torch::Tensor poly_coeffs = torch::linalg_solve(vtv, vtb);
        
//         // The SG coefficients are V * poly_coeffs
//         torch::Tensor sg_coeffs = torch::matmul(vander, poly_coeffs);
        
//         std::cout << "SG coefficients shape: " << sg_coeffs.sizes() << std::endl;
//         return sg_coeffs.to(torch::kFloat32);  // Convert back to float for efficiency
//     }
// };

// class JointStateNode : public rclcpp::Node {
// public:
//     JointStateNode()
//     : Node("joint_state_node"), recording_(false) {
//         using QoS = rclcpp::QoS;
//         auto qos = QoS(rclcpp::KeepLast(1)).reliable();

//         // Declare parameters for Savitzky-Golay filter
//         this->declare_parameter<int>("joint_command_sg_window", 11);
//         this->declare_parameter<int>("joint_command_sg_polyorder", 3);
//         this->declare_parameter<double>("sample_rate_hz", 200.0);

//         // Subscribers
//         joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
//             "/lbr/joint_states", qos, std::bind(&JointStateNode::joint_states_callback, this, _1));

//         ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
//             "/lbr/state/pose", qos, std::bind(&JointStateNode::ee_pose_callback, this, _1));

//         ee_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
//             "/lbr/state/twist", qos, std::bind(&JointStateNode::ee_vel_callback, this, _1));

//         // OptiTrack ball position subscription
//         ball_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
//             "/optitrack/ball", qos, std::bind(&JointStateNode::ball_pose_callback, this, _1));

//         // Publisher
//         joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
//             "/lbr/command/joint_position", 10);

//         // Get parameters
//         sample_rate_hz_ = this->get_parameter("sample_rate_hz").as_double();
//         dt_ = 1.0 / sample_rate_hz_;
        
//         int joint_cmd_sg_window = this->get_parameter("joint_command_sg_window").as_int();
//         int joint_cmd_sg_polyorder = this->get_parameter("joint_command_sg_polyorder").as_int();
            
//         RCLCPP_INFO(this->get_logger(), "Node initialized with OptiTrack ball tracking");
//         RCLCPP_INFO(this->get_logger(), "Joint command SG filter - Window: %d, Polyorder: %d", 
//                    joint_cmd_sg_window, joint_cmd_sg_polyorder);
//     }

//     // Initialize the Savitzky-Golay filter for joint commands
//     void init_sg_filter(int window_length, int polyorder, const std::vector<double>& initial_state = {}) {
//         torch::Tensor initial;
//         if (!initial_state.empty()) {
//             initial = torch::tensor(initial_state).to(torch::kFloat32);
//         }
//         sg_filter_ = std::make_unique<RealTimeSavitzkyGolay>(
//             window_length, polyorder, 0, dt_, initial);
        
//         RCLCPP_INFO(this->get_logger(), 
//                    "Savitzky-Golay filter initialized - Window: %d, Polyorder: %d", 
//                    window_length, polyorder);
//     }

//     void publish_joint_commands(const std::vector<std::vector<double>> &target_positions, int rate_hz = 200) {
//         recording_ = true;
//         rclcpp::Rate rate(rate_hz);
        
//         // Get SG filter parameters
//         int joint_cmd_sg_window = this->get_parameter("joint_command_sg_window").as_int();
//         int joint_cmd_sg_polyorder = this->get_parameter("joint_command_sg_polyorder").as_int();
        
//         // Initialize SG filter if not already done
//         if (!sg_filter_) {
//             if (!target_positions.empty()) {
//                 init_sg_filter(joint_cmd_sg_window, joint_cmd_sg_polyorder, target_positions[0]);
//             } else {
//                 RCLCPP_ERROR(this->get_logger(), "Empty target positions, cannot initialize SG filter");
//                 return;
//             }
//         }
        
//         RCLCPP_INFO(this->get_logger(), "Starting to publish SG-filtered joint commands...");
        
//         for (const auto &joint_pos : target_positions) {
//             // Apply the Savitzky-Golay filter to the joint positions
//             auto tensor_input = torch::tensor(joint_pos).to(torch::kFloat32);
//             auto filtered_tensor = (*sg_filter_)(tensor_input);
            
//             // Convert filtered tensor back to std::vector
//             auto filtered_pos = std::vector<double>(filtered_tensor.data_ptr<float>(), 
//                                                  filtered_tensor.data_ptr<float>() + filtered_tensor.numel());
            
//             // Store both original and filtered positions for analysis
//             original_commands_.push_back(joint_pos);
//             filtered_commands_.push_back(filtered_pos);
            
//             // Create and publish message
//             auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
//             std::copy(filtered_pos.begin(), filtered_pos.end(), msg.joint_position.begin());
//             joint_ref_pub_->publish(msg);
            
//             // Log current ball position periodically
//             if (filtered_commands_.size() % 50 == 0) {
//                 std::lock_guard<std::mutex> lock(ball_mutex_);
//                 RCLCPP_INFO(this->get_logger(), "Ball position: x=%.3f, y=%.3f, z=%.3f", 
//                            current_ball_pos_.x, current_ball_pos_.y, current_ball_pos_.z);
//             }
            
//             rate.sleep();
//         }

//         recording_ = false;
//         RCLCPP_INFO(this->get_logger(), "Finished publishing all SG-filtered targets");
//     }

//     void save_data(const std::string &output_dir) {
//         fs::create_directories(output_dir);
        
//         save_vector(received_joint_pos_, output_dir + "/ft_received_joint_pos_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_joint_vel_, output_dir + "/ft_received_joint_vel_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_joint_eff_, output_dir + "/ft_received_joint_effort_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(received_ee_pos_, output_dir + "/ft_received_ee_pos_np.csv", 
//                     "pos_X,pos_Y,pos_Z");
        
//         save_vector(received_ee_orient_, output_dir + "/ft_received_ee_orient_np.csv", 
//                     "or_w,or_x,or_y,or_z");
        
//         save_vector(received_ee_lin_vel_, output_dir + "/ft_received_ee_lin_vel_np.csv", 
//                     "lin_vel_X,lin_vel_Y,lin_vel_Z");
        
//         // Save the original and SG-filtered commands for comparison
//         save_vector(original_commands_, output_dir + "/ft_original_commands_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         save_vector(filtered_commands_, output_dir + "/ft_sg_filtered_commands_np.csv", 
//                     "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        
//         // Save ball position data
//         save_vector(received_ball_pos_, output_dir + "/ft_ball_positions_np.csv", 
//                     "ball_x,ball_y,ball_z");
        
//         save_vector(received_ball_orient_, output_dir + "/ft_ball_orientations_np.csv", 
//                     "ball_qw,ball_qx,ball_qy,ball_qz");
//     }
    
//     // Get current ball position (thread-safe)
//     geometry_msgs::msg::Point get_current_ball_position() {
//         std::lock_guard<std::mutex> lock(ball_mutex_);
//         return current_ball_pos_;
//     }
    
//     // Get current ball orientation (thread-safe)
//     geometry_msgs::msg::Quaternion get_current_ball_orientation() {
//         std::lock_guard<std::mutex> lock(ball_mutex_);
//         return current_ball_orient_;
//     }

// private:
//     bool recording_;
//     double sample_rate_hz_;
//     double dt_;
    
//     // ROS2 communication
//     rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr ee_pose_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ee_vel_sub_;
//     rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ball_pose_sub_;
//     rclcpp::Publisher<lbr_fri_idl::msg::LBRJointPositionCommand>::SharedPtr joint_ref_pub_;
    
//     // Savitzky-Golay filter for joint commands
//     std::unique_ptr<RealTimeSavitzkyGolay> sg_filter_;

//     // Data storage
//     std::vector<std::vector<double>> received_joint_pos_, received_joint_vel_, received_joint_eff_;
//     std::vector<std::vector<double>> received_ee_pos_, received_ee_orient_, received_ee_lin_vel_;
//     std::vector<std::vector<double>> original_commands_, filtered_commands_;
//     std::vector<std::vector<double>> received_ball_pos_, received_ball_orient_;
    
//     // Current ball state (thread-safe access)
//     std::mutex ball_mutex_;
//     geometry_msgs::msg::Point current_ball_pos_;
//     geometry_msgs::msg::Quaternion current_ball_orient_;

//     void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
//         if (!recording_) return;
        
//         std::vector<std::string> desired_order = {"lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7"};
//         std::vector<double> pos(7), vel(7), eff(7);
        
//         for (size_t i = 0; i < desired_order.size(); ++i) {
//             auto it = std::find(msg->name.begin(), msg->name.end(), desired_order[i]);
//             if (it != msg->name.end()) {
//                 size_t idx = std::distance(msg->name.begin(), it);
//                 pos[i] = msg->position[idx];
//                 vel[i] = msg->velocity[idx];
//                 eff[i] = msg->effort[idx];
//             }
//         }
        
//         received_joint_pos_.push_back(pos);
//         received_joint_vel_.push_back(vel);
//         received_joint_eff_.push_back(eff);
//     }

//     void ee_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
//         if (!recording_) return;
        
//         received_ee_pos_.push_back({msg->position.x, msg->position.y, msg->position.z});
//         received_ee_orient_.push_back({msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z});
//     }

//     void ee_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
//         if (!recording_) return;
        
//         received_ee_lin_vel_.push_back({msg->linear.x, msg->linear.y, msg->linear.z});
//     }
    
//     // OptiTrack ball pose callback
//     void ball_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
//         // Update current ball position (thread-safe)
//         {
//             std::lock_guard<std::mutex> lock(ball_mutex_);
//             current_ball_pos_ = msg->pose.position;
//             current_ball_orient_ = msg->pose.orientation;
//         }
        
//         // Record ball data if recording is active
//         if (recording_) {
//             received_ball_pos_.push_back({msg->pose.position.x, msg->pose.position.y, msg->pose.position.z});
//             received_ball_orient_.push_back({msg->pose.orientation.w, msg->pose.orientation.x, 
//                                            msg->pose.orientation.y, msg->pose.orientation.z});
//         }
//     }

//     void save_vector(const std::vector<std::vector<double>> &data, 
//                     const std::string &filename, 
//                     const std::string &header) {
//         std::ofstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Failed to open file: " << filename << std::endl;
//             return;
//         }

//         // Write header
//         file << header << "\n";

//         // Write data
//         for (const auto &row : data) {
//             for (size_t i = 0; i < row.size(); ++i) {
//                 file << row[i];
//                 if (i != row.size() - 1)
//                     file << ",";
//             }
//             file << "\n";
//         }

//         file.close();
//     }

//     // Helper function to check if a string is a valid number
//     bool is_number(const std::string& str) {
//         try {
//             std::stod(str);
//             return true;
//         } catch (const std::exception&) {
//             return false;
//         }
//     }
// };

// int main(int argc, char *argv[]) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<JointStateNode>();

//     // Load joint trajectory from CSV file
//     std::string input_csv = "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/ft_idealpd.csv";
//     std::ifstream file(input_csv);
//     if (!file.is_open()) {
//         RCLCPP_ERROR(node->get_logger(), "Failed to open file: %s", input_csv.c_str());
//         return 1;
//     }
    
//     std::vector<std::vector<double>> joint_data;
//     std::string line;
//     int line_number = 0;
//     bool skip_header = true;  // Flag to skip the first line if it's a header

//     // Parse CSV file to extract joint positions (columns 21-27)
//     while (std::getline(file, line)) {
//         line_number++;
        
//         // Skip empty lines
//         if (line.empty()) continue;
        
//         // Skip the first line if it looks like a header
//         if (skip_header && line_number == 1) {
//             // Check if the first few cells contain non-numeric data
//             std::stringstream ss(line);
//             std::string cell;
//             bool has_non_numeric = false;
//             int check_count = 0;
//             while (std::getline(ss, cell, ',') && check_count < 5) {
//                 cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
//                 if (!cell.empty() && !std::isdigit(cell[0]) && cell[0] != '-' && cell[0] != '+' && cell[0] != '.') {
//                     has_non_numeric = true;
//                     break;
//                 }
//                 check_count++;
//             }
//             if (has_non_numeric) {
//                 RCLCPP_INFO(node->get_logger(), "Skipping header line: %s", line.substr(0, 50).c_str());
//                 continue;
//             }
//         }
//         skip_header = false;
        
//         std::stringstream ss(line);
//         std::vector<double> row;
//         std::string cell;
//         int col = 0;
        
//         while (std::getline(ss, cell, ',')) {
//             if (col >= 21 && col < 28) {
//                 try {
//                     // Trim whitespace
//                     cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
//                     if (!cell.empty()) {
//                         double value = std::stod(cell);
//                         row.push_back(value);
//                     }
//                 } catch (const std::exception &e) {
//                     RCLCPP_WARN(node->get_logger(), "Error parsing line %d, column %d ('%s'): %s", 
//                                line_number, col, cell.c_str(), e.what());
//                     // Skip this line if we can't parse the joint data
//                     row.clear();
//                     break;
//                 }
//             }
//             col++;
//         }
        
//         if (row.size() == 7) {
//             joint_data.push_back(row);
//         } else if (!row.empty()) {
//             RCLCPP_WARN(node->get_logger(), "Line %d: Expected 7 joint values, got %zu", 
//                        line_number, row.size());
//         }
//     }

//     RCLCPP_INFO(node->get_logger(), "Loaded %zu joint trajectory points", joint_data.size());

//     if (joint_data.empty()) {
//         RCLCPP_ERROR(node->get_logger(), "No valid joint data loaded from CSV file");
//         return 1;
//     }

//     // Run the trajectory execution in a separate thread
//     std::thread pub_thread([&]() {
//         // Wait a moment to ensure OptiTrack subscription is active
//         std::this_thread::sleep_for(1s);
        
//         // Get initial ball position
//         auto initial_ball_pos = node->get_current_ball_position();
//         RCLCPP_INFO(node->get_logger(), "Initial ball position: x=%.3f, y=%.3f, z=%.3f", 
//                    initial_ball_pos.x, initial_ball_pos.y, initial_ball_pos.z);
        
//         // Execute the trajectory with SG filtering
//         double sample_rate_hz = 200.0;
//         node->publish_joint_commands(joint_data, sample_rate_hz);
        
//         // Save all collected data
//         node->save_data("/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/ft/idealpd_sg");
        
//         RCLCPP_INFO(node->get_logger(), "Trajectory execution completed. Data saved.");
//         rclcpp::shutdown();
//     });

//     // Spin the node to handle callbacks
//     rclcpp::spin(node);
//     pub_thread.join();
    
//     return 0;
// }