#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <lbr_fri_idl/msg/lbr_joint_position_command.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include <Eigen/Geometry>

#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <mutex>
#include <cmath>
#include <numeric>

using std::placeholders::_1;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Forward declarations
class RealTimeSavitzkyGolay;
class CSVReader;
class DeploymentPolicy;
class JointStateNode;

// CSVReader class for loading data
class CSVReader {
public:
    static std::vector<std::vector<float>> read(const std::string& filename) {
        std::vector<std::vector<float>> data;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        // Skip header if exists
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                // Trim whitespace
                cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
                
                try {
                    if (!cell.empty()) {
                        row.push_back(std::stof(cell));
                    }
                } catch (const std::exception& e) {
                    // Handle conversion errors
                    row.push_back(0.0f);
                }
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }

        return data;
    }
};

// RealTimeSavitzkyGolay class for signal smoothing
class RealTimeSavitzkyGolay {
public:
    RealTimeSavitzkyGolay(int window_length = 9, 
                          int polyorder = 3, 
                          int deriv = 0, 
                          double delta = 1.0,
                          torch::Tensor initial_values = torch::Tensor()) {
        
        // Ensure window length is odd
        if (window_length % 2 == 0) {
            window_length += 1;
            std::cout << "Warning: window_length adjusted to odd value: " << window_length << std::endl;
        }
        
        // Check polyorder against window length
        if (polyorder >= window_length) {
            polyorder = window_length - 1;
            std::cout << "Warning: polyorder too large, reduced to " << polyorder << std::endl;
        }
        
        window_length_ = window_length;
        polyorder_ = polyorder;
        
        // Precompute coefficients
        coeffs_ = compute_savgol_coeffs_smooth(window_length_, polyorder_);
        
        // Initialize buffer
        if (initial_values.numel() > 0) {
            // Check if initial_values is 1D or batch of vectors
            bool is_batched = initial_values.dim() > 1;
            
            // Create appropriately shaped buffer
            if (is_batched) {
                if (initial_values.size(0) >= window_length_) {
                    // Use the most recent values
                    buffer_ = initial_values.slice(0, initial_values.size(0) - window_length_, initial_values.size(0)).clone();
                } else {
                    // Pad with first value
                    auto first_value = initial_values[0].unsqueeze(0);
                    auto padding = first_value.repeat({window_length_ - initial_values.size(0), 1});
                    buffer_ = torch::cat({padding, initial_values}, 0);
                }
            } else {
                // Handle 1D tensor case
                if (initial_values.size(0) >= window_length_) {
                    buffer_ = initial_values.slice(0, initial_values.size(0) - window_length_, initial_values.size(0)).clone();
                } else {
                    float first_val = initial_values.size(0) > 0 ? initial_values[0].item<float>() : 0.0f;
                    auto padding = torch::full({window_length_ - initial_values.size(0)}, first_val);
                    buffer_ = torch::cat({padding, initial_values}, 0);
                }
            }
        } else {
            // Default buffer shape depends on input later
            buffer_ = torch::zeros({window_length_});
            buffer_initialized_ = false;
        }
        
        current_idx_ = 0;
    }
    
    torch::Tensor operator()(const torch::Tensor& x) {
        // Initialize buffer properly on first call if not initialized
        if (!buffer_initialized_) {
            // Check input dimension to properly set up buffer
            if (x.dim() > 0 && x.size(0) > 1) {
                // Input is a vector, resize buffer to match
                buffer_ = torch::zeros({window_length_, x.size(0)}).to(x.device(), x.dtype());
            }
            buffer_initialized_ = true;
        }
        
        // Update buffer with new value (circular buffer approach)
        if (x.dim() == 0) {
            // Handle scalar input
            buffer_[current_idx_] = x.item<float>();
        } else {
            // Handle vector input
            buffer_[current_idx_] = x;
        }
        
        current_idx_ = (current_idx_ + 1) % window_length_;
        
        // Rearrange buffer to get correct temporal order for filtering
        torch::Tensor ordered_buffer;
        if (current_idx_ == 0) {
            // If we're at the start of the buffer, no need to rearrange
            ordered_buffer = buffer_;
        } else {
            // Concatenate the two parts of the circular buffer
            ordered_buffer = torch::cat({
                buffer_.slice(0, current_idx_, window_length_),
                buffer_.slice(0, 0, current_idx_)
            }, 0);
        }
        
        // Apply precomputed coefficients to get filtered value
        torch::Tensor filtered_value;
        
        if (ordered_buffer.dim() == 1) {
            // If buffer contains scalars
            filtered_value = (ordered_buffer * coeffs_).sum();
        } else {
            // If buffer contains vectors - apply filtering to each dimension
            filtered_value = torch::zeros_like(x);
            for (int i = 0; i < ordered_buffer.size(1); i++) {
                filtered_value[i] = (ordered_buffer.select(1, i) * coeffs_).sum();
            }
        }
        
        return filtered_value;
    }
    
private:
    int window_length_;
    int polyorder_;
    torch::Tensor coeffs_;
    torch::Tensor buffer_;
    int current_idx_;
    bool buffer_initialized_ = true;  // Set to false if buffer shape is unknown initially
    
    torch::Tensor compute_savgol_coeffs_smooth(int window_length, int polyorder) {
        // For the smoothing case (deriv=0), we implement a basic version
        int half_window = window_length / 2;
        
        // Create Vandermonde matrix - using correct C++ syntax for torch::arange
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor x = torch::arange(-half_window, half_window + 1, options);
        torch::Tensor vander = torch::zeros({window_length, polyorder + 1});
        
        for (int i = 0; i <= polyorder; i++) {
            vander.select(1, i) = torch::pow(x, i);
        }
        
        // Compute QR decomposition to solve the least squares problem
        auto qr = torch::linalg_qr(vander);
        auto q = std::get<0>(qr);
        auto r = std::get<1>(qr);
        
        // Target vector for smoothing (deriv=0) is [1, 0, 0, ...] to extract constant term
        torch::Tensor target = torch::zeros({polyorder + 1});
        target[0] = 1.0;
        
        // Solve R*x = Q^T * target
        torch::Tensor qtb = torch::matmul(q.transpose(0, 1), target);
        
        // Back substitution to solve R*coeffs = qtb
        torch::Tensor coeffs = torch::zeros({polyorder + 1});
        for (int k = polyorder; k >= 0; k--) {
            coeffs[k] = qtb[k];
            for (int j = k + 1; j <= polyorder; j++) {
                coeffs[k] -= coeffs[j] * r.index({k, j});
            }
            coeffs[k] /= r.index({k, k});
        }
        
        // Compute filter coefficients
        torch::Tensor sg_coeffs = torch::matmul(vander, coeffs);
        
        return sg_coeffs;
    }
};

// DeploymentPolicy class for loading and running the policy
class DeploymentPolicy {
public:
    DeploymentPolicy(const std::string& checkpoint_path) {
        // Load exported TorchScript model
        try {
            model = torch::jit::load(checkpoint_path);
            model.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }
    }

    torch::Tensor get_action(const torch::Tensor& raw_obs) {
        // Ensure CPU tensor conversion
        auto obs_tensor = raw_obs.unsqueeze(0).to(torch::kCPU).to(torch::kFloat32);
        
        torch::NoGradGuard no_grad;
        auto action = model.forward({obs_tensor}).toTensor();
        
        return action.squeeze(0);
    }

private:
    torch::jit::Module model;
};

// Main JointStateNode class
class JointStateNode : public rclcpp::Node {
public:
    JointStateNode()
    : Node("joint_state_node"), t_(0), recording_(false) {
        using QoS = rclcpp::QoS;
        auto qos = QoS(rclcpp::KeepLast(1)).reliable();

        // Declare parameters
        this->declare_parameter<std::string>("model_path", 
            "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/policy.pt");
        this->declare_parameter<int>("sg_window_length", 9);
        this->declare_parameter<int>("sg_polyorder", 3);

        // Hyperparameters matching Python script
        dt_ = 1.0 / 200.0;
        tennis_ball_pos_scale_ = 0.25;
        lin_vel_scale_ = 0.15;
        dof_vel_scale_ = 0.31;

        // Joint limits
        robot_dof_lower_limits_ = torch::tensor({-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543});
        robot_dof_upper_limits_ = torch::tensor({2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543});

        robot_dof_lower_limits_np_ = robot_dof_lower_limits_.clone();
        robot_dof_upper_limits_np_ = robot_dof_upper_limits_.clone();

        // Final target and throwing parameters
        final_target_pos_ = torch::tensor({-0.65, -0.4, 0.55});
        throwing_pos_ = torch::tensor({0.2, -0.35, 0.9});
        throwing_vel_ = torch::tensor({2.5, 0.0, 0.5});

        // Load observation data
        load_observation_data();

        // Initialize policy
        std::string model_path = this->get_parameter("model_path").as_string();
        policy_ = std::make_unique<DeploymentPolicy>(model_path);

        // Initialize action history and other buffers
        action_history_ = torch::zeros({2, 7});
        action_logits_ = torch::zeros(7);
        pure_action_ = torch::zeros(7);
        robot_dof_targets_ = torch::zeros(7);
        
        // Initialize Savitzky-Golay filter for action smoothing
        int sg_window_length = this->get_parameter("sg_window_length").as_int();
        int sg_polyorder = this->get_parameter("sg_polyorder").as_int();
        // action_filter_ = std::make_unique<RealTimeSavitzkyGolay>(
        //     sg_window_length, sg_polyorder, 0, dt_, torch::zeros(7));

        // Subscribers
        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/lbr/joint_states", qos, std::bind(&JointStateNode::joint_states_callback, this, _1));

        ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/lbr/state/pose", qos, std::bind(&JointStateNode::ee_pose_callback, this, _1));

        ee_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/lbr/state/twist", qos, std::bind(&JointStateNode::ee_vel_callback, this, _1));

        // Publisher
        joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
            "/lbr/command/joint_position", 1);

        // Timer for action computation
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_), 
            std::bind(&JointStateNode::compute_action_and_publish, this)
        );

        RCLCPP_INFO(this->get_logger(), "JointStateNode initialized with Savitzky-Golay filter parameters: window=%d, polyorder=%d",
                   sg_window_length, sg_polyorder);
    }

    void load_observation_data() {
        // Load CSV data 
        std::string input_file_path = "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/ft_idealpd.csv";
        auto data = CSVReader::read(input_file_path);
        
        // Extract data from CSV 
        for (const auto& row : data) {
            // Ensure row has at least 68 columns to prevent out-of-bounds access
            if (row.size() < 68) {
                std::cerr << "Skipping row with insufficient columns" << std::endl;
                continue;
            }
            
            // Tennis ball position (columns 42-44)
            std::vector<float> position_row = {
                row[42], 
                row[43], 
                row[44]
            };
            tennisball_pos_.push_back(position_row);
            
            // Tennis ball linear velocity (columns 45-47)
            std::vector<float> velocity_row = {
                row[45], 
                row[46], 
                row[47]
            };
            tennisball_lin_vel_.push_back(velocity_row);
            
        }
    }

    // Helper function to convert PyTorch tensor to std::vector<double>
    std::vector<double> tensor_to_vector(const torch::Tensor& tensor) {
        std::vector<double> result;
        result.reserve(tensor.numel());
        
        // Ensure tensor is on CPU and contiguous
        auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
        
        // Access the tensor data
        if (cpu_tensor.dtype() == torch::kFloat32) {
            float* data_ptr = cpu_tensor.data_ptr<float>();
            for (int i = 0; i < cpu_tensor.numel(); i++) {
                result.push_back(static_cast<double>(data_ptr[i]));
            }
        } else if (cpu_tensor.dtype() == torch::kFloat64) {
            double* data_ptr = cpu_tensor.data_ptr<double>();
            for (int i = 0; i < cpu_tensor.numel(); i++) {
                result.push_back(data_ptr[i]);
            }
        } else {
            // For other types, use item access (slower but more general)
            for (int i = 0; i < cpu_tensor.numel(); i++) {
                result.push_back(cpu_tensor.flatten()[i].item<double>());
            }
        }
        
        return result;
    }

    void compute_action_and_publish() {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if we have all necessary data
        if (joint_positions_obs_.numel() == 0 || joint_velocities_obs_.numel() == 0 ||
            ee_pose_.numel() == 0 || ee_orientation_.numel() == 0 || 
            ee_vel_.numel() == 0 || ee_angular_vel_.numel() == 0) {
            return;
        }

        // Check time index
        if (t_ >= tennisball_pos_.size()) {
            RCLCPP_WARN(this->get_logger(), "Time index exceeded data length. Shutting down node.");
            rclcpp::shutdown();
            return;
        }

        // Scale joint observations (similar to Python script)
        auto dof_pos_scaled_obs = 2.0 * (joint_positions_obs_ - robot_dof_lower_limits_np_) / 
            (robot_dof_upper_limits_np_ - robot_dof_lower_limits_np_) - 1.0;
        auto dof_vel_scaled_obs = joint_velocities_obs_ * dof_vel_scale_;

        // Extract current time step data
        torch::Tensor tennisball_pos_obs = torch::tensor(tennisball_pos_[t_]) * tennis_ball_pos_scale_;
        torch::Tensor tennisball_lin_vel_obs = torch::tensor(tennisball_lin_vel_[t_]) * lin_vel_scale_;
        torch::Tensor ee_lin_vel_scaled = ee_vel_ * lin_vel_scale_;

        // Prepare actions
        torch::Tensor action = action_logits_.numel() == 0 ? 
            torch::ones(7) : action_logits_;
        torch::Tensor pure_action = pure_action_.numel() == 0 ? 
            torch::zeros(7) : pure_action_;


        // Update action history
        action_history_ = torch::roll(action_history_, -1, 0);
        action_history_[-1] = action_logits_;


        // Prepare observation tensor
        torch::Tensor observations = torch::cat({
            pure_action_,                    // 7
            action_history_.flatten(),      // 14 (2x7)
            dof_pos_scaled_obs.flatten(),   // 7
            dof_vel_scaled_obs.flatten(),   // 7
            tennisball_pos_obs.flatten(),   // 3
            tennisball_lin_vel_obs.flatten(), // 3
            ee_lin_vel_scaled.flatten(),    // 3
            ee_orientation_.flatten(),      // 4
        });

        // Sanity check observation size
        assert(observations.size(0) == 48);

        // Get action from policy
        torch::Tensor raw_action = policy_->get_action(observations);
        
        // Apply Savitzky-Golay filter to the action
        // action_logits_ = (*action_filter_)(raw_action);
        action_logits_ = raw_action; // For now, use raw_action directly
        pure_action_ = raw_action.clone();
        
        // Store the raw (unfiltered) action for debugging
        raw_action_history_.push_back(tensor_to_vector(raw_action));
        filtered_action_history_.push_back(tensor_to_vector(action_logits_));

        // Compute new targets
        torch::Tensor targets = joint_positions_obs_ + (action_logits_ * 0.1);
        robot_dof_targets_ = torch::clamp(
            targets,
            0.975 * robot_dof_lower_limits_,
            0.975 * robot_dof_upper_limits_
        );

        // Save outputs for logging/debugging using tensor_to_vector helper
        joint_targets_.push_back(tensor_to_vector(targets));
        joint_poses_.push_back(tensor_to_vector(joint_positions_obs_));
        joint_velocities_.push_back(tensor_to_vector(joint_velocities_obs_));
        ee_poses_.push_back(tensor_to_vector(ee_pose_));
        ee_orientations_.push_back(tensor_to_vector(ee_orientation_));
        ee_velocities_.push_back(tensor_to_vector(ee_vel_));

        // Publish joint commands
        auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
        
        // Use tensor_to_vector instead of tolist()
        auto robot_dof_targets_vec = tensor_to_vector(robot_dof_targets_);
        std::copy(robot_dof_targets_vec.begin(), robot_dof_targets_vec.end(), msg.joint_position.begin());
        joint_ref_pub_->publish(msg);

        t_++;
    }

    void save_output(const std::vector<std::vector<double>>& outputs, 
                     const std::string& output_file_path, 
                     const std::string& header = "") {
        std::ofstream file(output_file_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", output_file_path.c_str());
            return;
        }

        // Write header if provided
        if (!header.empty()) {
            file << header << "\n";
        }

        // Write data
        for (const auto& row : outputs) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i != row.size() - 1)
                    file << ",";
            }
            file << "\n";
        }

        file.close();
        RCLCPP_INFO(this->get_logger(), "Saved output to: %s", output_file_path.c_str());
    }

    void destroy_node() {
        // Prepare output directory
        std::filesystem::path script_dir = std::filesystem::path(__FILE__).parent_path();
        std::string output_dir = "/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/tm/idealpd4";
        std::filesystem::create_directories(output_dir);

        // Save various outputs
        save_output(joint_targets_, output_dir + "/tm_received_joint_target_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        save_output(joint_poses_, output_dir + "/tm_received_joint_pos_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        save_output(joint_velocities_, output_dir + "/tm_received_joint_vel_np.csv", 
                    "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
        save_output(ee_poses_, output_dir + "/tm_received_ee_pos_np.csv", 
                    "pos_X,pos_Y,pos_Z");
        save_output(ee_orientations_, output_dir + "/tm_received_ee_orientation_np.csv", 
                    "or_w,or_x,or_y,or_z");
        save_output(ee_velocities_, output_dir + "/tm_received_ee_vel_np.csv", 
                    "lin_vel_X,lin_vel_Y,lin_vel_Z");
        
        // Save additional filter-related outputs
        save_output(raw_action_history_, output_dir + "/tm_raw_actions_np.csv", 
                    "action_0,action_1,action_2,action_3,action_4,action_5,action_6");
        save_output(filtered_action_history_, output_dir + "/tm_filtered_actions_sg_np.csv", 
                    "action_0,action_1,action_2,action_3,action_4,action_5,action_6");

        RCLCPP_INFO(this->get_logger(), "Model outputs saved on shutdown.");
    }

private:
    // Joint state callback
    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::string> desired_order = {
            "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", 
            "lbr_A5", "lbr_A6", "lbr_A7"
        };

        // Extract joint positions and velocities in correct order
        std::vector<double> positions(7, 0.0);
        std::vector<double> velocities(7, 0.0);

        for (size_t i = 0; i < desired_order.size(); ++i) {
            auto it = std::find(msg->name.begin(), msg->name.end(), desired_order[i]);
            if (it != msg->name.end()) {
                size_t idx = std::distance(msg->name.begin(), it);
                positions[i] = msg->position[idx];
                velocities[i] = msg->velocity[idx];
            }
        }

        // Convert to torch tensors
        joint_positions_obs_ = torch::tensor(positions);
        joint_velocities_obs_ = torch::tensor(velocities);

        // Initialize robot_dof_targets if not set
        if (robot_dof_targets_.numel() == 0) {
            robot_dof_targets_ = torch::tensor(positions);
        }
    }

    // Pose callback
    void ee_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Store end effector pose and orientation
        ee_pose_ = torch::tensor({msg->position.x, msg->position.y, msg->position.z});
        ee_orientation_ = torch::tensor({
            msg->orientation.w, 
            msg->orientation.x, 
            msg->orientation.y, 
            msg->orientation.z
        });
    }

    // Velocity callback
    void ee_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Store linear and angular velocities
        ee_vel_ = torch::tensor({msg->linear.x, msg->linear.y, msg->linear.z});
        ee_angular_vel_ = torch::tensor({msg->angular.x, msg->angular.y, msg->angular.z});
    }

    // Member variables
    // Time and recording
    int t_;
    bool recording_;

    // Synchronization
    std::mutex mutex_;

    // ROS2 Communication
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr ee_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ee_vel_sub_;
    rclcpp::Publisher<lbr_fri_idl::msg::LBRJointPositionCommand>::SharedPtr joint_ref_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Hyperparameters
    double dt_;
    double tennis_ball_pos_scale_;
    double lin_vel_scale_;
    double dof_vel_scale_;

    // Savitzky-Golay filter
    std::unique_ptr<RealTimeSavitzkyGolay> action_filter_;

    // Observation data
    std::vector<std::vector<float>> tennisball_pos_;
    std::vector<std::vector<float>> tennisball_lin_vel_;

    // Tensors for observations and targets
    torch::Tensor joint_positions_obs_;
    torch::Tensor joint_velocities_obs_;
    torch::Tensor ee_pose_;
    torch::Tensor ee_orientation_;
    torch::Tensor ee_vel_;
    torch::Tensor ee_angular_vel_;

    // Policy and action-related tensors
    std::unique_ptr<DeploymentPolicy> policy_;
    torch::Tensor robot_dof_lower_limits_;
    torch::Tensor robot_dof_upper_limits_;
    torch::Tensor robot_dof_lower_limits_np_;
    torch::Tensor robot_dof_upper_limits_np_;
    torch::Tensor final_target_pos_;
    torch::Tensor throwing_pos_;
    torch::Tensor throwing_vel_;

    torch::Tensor action_history_;
    torch::Tensor action_logits_;
    torch::Tensor pure_action_;
    torch::Tensor robot_dof_targets_;

    // Output storage for logging/debugging
    std::vector<std::vector<double>> joint_targets_;
    std::vector<std::vector<double>> joint_poses_;
    std::vector<std::vector<double>> joint_velocities_;
    std::vector<std::vector<double>> ee_poses_;
    std::vector<std::vector<double>> ee_orientations_;
    std::vector<std::vector<double>> ee_velocities_;
    std::vector<std::vector<double>> raw_action_history_;
    std::vector<std::vector<double>> filtered_action_history_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<JointStateNode>();
        rclcpp::spin(node);
        node->destroy_node();
    } catch (const std::exception& e) {
        std::cerr << "Error in JointStateNode: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error in JointStateNode" << std::endl;
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}