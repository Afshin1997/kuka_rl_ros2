#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>  // For OptiTrack
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
#include <deque>
#include "json.hpp" // nlohmann/json library - install with: apt-get install nlohmann-json3-dev

using json = nlohmann::json;

using std::placeholders::_1;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

// Forward declarations
class DeploymentPolicy;
class JointStateNode;

class PolynomialJointPredictor {
private:
    struct JointModel {
        std::vector<double> scaler_means;
        std::vector<double> scaler_scales;
        std::vector<std::vector<int>> powers;
        std::vector<double> coefficients;
        double intercept;
        int degree;
        int n_features;
    };
    
    std::vector<JointModel> joint_models;
    bool is_loaded;
    
public:
    PolynomialJointPredictor() : is_loaded(false) {}
    
    /**
     * Load polynomial models from JSON file
     * 
     * @param json_file Path to the exported JSON file
     * @return true if successful, false otherwise
     */
    bool loadFromJSON(const std::string& json_file) {
        try {
            std::ifstream file(json_file);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open file " << json_file << std::endl;
                return false;
            }
            
            json data;
            file >> data;
            
            // Clear existing models
            joint_models.clear();
            joint_models.resize(7);
            
            // Load each joint model
            for (int joint_idx = 0; joint_idx < 7; joint_idx++) {
                std::string joint_key = "joint_" + std::to_string(joint_idx);
                
                if (!data["joints"].contains(joint_key)) {
                    std::cerr << "Error: Missing " << joint_key << " in JSON" << std::endl;
                    return false;
                }
                
                auto& joint_data = data["joints"][joint_key];
                JointModel& model = joint_models[joint_idx];
                
                // Load scaling parameters
                model.scaler_means = joint_data["scaling"]["means"].get<std::vector<double>>();
                model.scaler_scales = joint_data["scaling"]["scales"].get<std::vector<double>>();
                
                // Load polynomial parameters
                model.powers = joint_data["polynomial"]["powers"].get<std::vector<std::vector<int>>>();
                model.degree = joint_data["degree"].get<int>();
                model.n_features = joint_data["n_features"].get<int>();
                
                // Load regression parameters
                model.coefficients = joint_data["regression"]["coefficients"].get<std::vector<double>>();
                model.intercept = joint_data["regression"]["intercept"].get<double>();
                
                // Validate sizes
                if (model.scaler_means.size() != 7 || model.scaler_scales.size() != 7) {
                    std::cerr << "Error: Invalid scaler size for joint " << joint_idx << std::endl;
                    return false;
                }
                
                if (model.powers.size() != model.coefficients.size()) {
                    std::cerr << "Error: Powers and coefficients size mismatch for joint " << joint_idx << std::endl;
                    return false;
                }
            }
            
            is_loaded = true;
            std::cout << "✓ Model loaded successfully from " << json_file << std::endl;
            printModelInfo();
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading JSON: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * Compute a single polynomial feature
     * Formula: feature = product(input[i]^power[i] for i in range(7))
     */
    double computePolynomialFeature(const std::array<double, 7>& scaled_inputs, 
                                   const std::vector<int>& powers) const {
        double result = 1.0;
        
        for (int i = 0; i < 7; i++) {
            for (int p = 0; p < powers[i]; p++) {
                result *= scaled_inputs[i];
            }
        }
        
        return result;
    }
    
    /**
     * Predict joint position for a single joint
     * 
     * @param set_targets Array of 7 input values (set_target_0 to set_target_6)
     * @param joint_idx Joint index (0-6)
     * @return Predicted joint position
     */
    double predictSingleJoint(const std::array<double, 7>& set_targets, int joint_idx) const {
        if (!is_loaded) {
            throw std::runtime_error("Model not loaded. Call loadFromJSON() first.");
        }
        
        if (joint_idx < 0 || joint_idx >= 7) {
            throw std::invalid_argument("Joint index must be 0-6");
        }
        
        const auto& model = joint_models[joint_idx];
        
        // Step 1: Scale inputs
        auto scaled_inputs = scaleInputs(set_targets, joint_idx);
        
        // Step 2: Start with intercept
        double result = model.intercept;
        
        // Step 3: Add polynomial terms
        for (size_t i = 0; i < model.powers.size(); i++) {
            double feature = computePolynomialFeature(scaled_inputs, model.powers[i]);
            result += model.coefficients[i] * feature;
        }
        
        return result;
    }
    
    
    /**
     * Predict all joint positions
     * 
     * @param set_targets Array of 7 input values (set_target_0 to set_target_6)
     * @return Array of 7 predicted joint positions (joint_pos_0 to joint_pos_6)
     */
    std::array<double, 7> predict(const std::array<double, 7>& set_targets) const {
        std::array<double, 7> joint_positions;
        
        for (int joint_idx = 0; joint_idx < 7; joint_idx++) {
            joint_positions[joint_idx] = predictSingleJoint(set_targets, joint_idx);
        }
        
        return joint_positions;
    }
    
};



auto joint_positions = predictor.predict(set_targets);

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

// DeploymentPolicy class for loading and running the policy
class DeploymentPolicy {
public:
    DeploymentPolicy(const std::string& checkpoint_path) {
        try {
            model = torch::jit::load(checkpoint_path);
            model.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }
    }

    torch::Tensor get_action(const torch::Tensor& raw_obs) {
        auto obs_tensor = raw_obs.unsqueeze(0).to(torch::kCPU).to(torch::kFloat32);
        
        torch::NoGradGuard no_grad;
        auto action = model.forward({obs_tensor}).toTensor();
        
        return action.squeeze(0);
    }

private:
    torch::jit::Module model;
};

class JointStateNode : public rclcpp::Node {
public:
    JointStateNode()
    : Node("joint_state_node"), t_(0), recording_(false), recording_active_(false) {
        
        using QoS = rclcpp::QoS;
        auto qos = QoS(rclcpp::KeepLast(10)).reliable();

        // Declare parameters
        this->declare_parameter<std::string>("model_path", 
            "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/policy.pt");
        this->declare_parameter<std::string>("output_dir", 
            "/home/user/kuka_rl_ros2/src/catch_and_throw/output_files/tm/idealpd");
        this->declare_parameter<std::string>("PolynomialModelPath", 
            "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/polynomial_models.json");
        this->declare_parameter<double>("max_ball_x_position", 3.0);  // Maximum X position for command publishing
        this->declare_parameter<double>("min_ball_z_position", -1.0);  // Minimum Z position for command publishing
        this->declare_parameter<double>("ball_vel_ema_alpha", 1.0);  // EMA smoothing factor for ball velocity
        this->declare_parameter<int>("joint_vel_window_size", 5);  // SMA smoothing window for joint velocites
        this->declare_parameter<bool>("fake_joint_state", 5);  // Fake joint state for testing
        this->declare_parameter<bool>("fake_ball_state", 5);  // Fake ball state for testing

    

        // Initialize and clean output directory
        
        output_dir_ = this->get_parameter("output_dir").as_string();
        std::string polynomial_model_path_ = this->get_parameter("PolynomialModelPath").as_string();
        max_ball_x_position_ = this->get_parameter("max_ball_x_position").as_double();
        min_ball_z_position_ = this->get_parameter("min_ball_z_position").as_double();
        ball_vel_ema_alpha_ = this->get_parameter("ball_vel_ema_alpha").as_double();
        joint_vel_window_size_ = this->get_parameter("joint_vel_window_size").as_int();

        std::string model_path = this->get_parameter("model_path").as_string();
        policy_ = std::make_unique<DeploymentPolicy>(model_path);

        fake_joint_state_ = this->get_parameter("fake_joint_state").as_bool();
        fake_ball_state_ = this->get_parameter("fake_ball_state").as_bool();

        setup_output_directory();

        // Hyperparameters matching Python script
        dt_ = dt_nn;
        tennis_ball_pos_scale_ = 0.25;
        lin_vel_scale_ = 0.15;
        dof_vel_scale_ = 0.31;
        action_scale_ = 0.1;  // From the Python script: targets = self.robot_dof_pos + self.actions * 0.1

        // Joint limits
        robot_dof_lower_limits_ = torch::tensor({-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543});
        robot_dof_upper_limits_ = torch::tensor({2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543});

        robot_dof_lower_limits_np_ = robot_dof_lower_limits_.clone();
        robot_dof_upper_limits_np_ = robot_dof_upper_limits_.clone();

        // Initialize action history and other buffers
        action_history_ = torch::zeros({2, 7});
        robot_dof_targets_ = torch::zeros(7);

        // Initialize ball velocity EMA
        tennisball_lin_vel_world_ema_ = torch::zeros(3);
        ball_vel_ema_initialized_ = false;

        // Initialize robot base position (assuming robot base is at origin)
        robot_base_pos_ = torch::tensor({0.0, 0.0, 0.0});

        //Initialize interpolation vectors dimension
        q_interp_init.resize(7);
        q_interp_final.resize(7);
        q_interp_ref.resize(7);
        q_cmd.resize(7);
        initial_positions.resize(7);


        // Initialize to PolynomialJointPredictor
        predictor = PolynomialJointPredictor();
        predictor.loadFromJSON(polynomial_model_path_)

        // Subscribers
        joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/lbr/joint_states", qos, std::bind(&JointStateNode::joint_states_callback, this, _1));

        ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/lbr/state/pose", qos, std::bind(&JointStateNode::ee_pose_callback, this, _1));

        ee_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/lbr/state/twist", qos, std::bind(&JointStateNode::ee_vel_callback, this, _1));

        // OptiTrack ball position subscription
        ball_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "optitrack/ball_marker", qos, std::bind(&JointStateNode::ball_pose_callback, this, _1));

        // Publisher
        joint_ref_pub_ = this->create_publisher<lbr_fri_idl::msg::LBRJointPositionCommand>(
            "/lbr/command/joint_position", 10);

        //Fake trajectory reading
        load_observation_data();

        // Timer for action computation
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_), 
            std::bind(&JointStateNode::compute_action_and_publish, this)
        );

        // RCLCPP_INFO(this->get_logger(), "JointStateNode initialized - Output directory prepared");


    }

    // Setup output directory - remove old and create new
    void setup_output_directory() {
        try {
            // Remove the entire idealpd directory if it exists
            if (fs::exists(output_dir_)) {
                fs::remove_all(output_dir_);
                // RCLCPP_INFO(this->get_logger(), "Removed previous output directory: %s", output_dir_.c_str());
            }
            
            // Create fresh directory
            fs::create_directories(output_dir_);
            // RCLCPP_INFO(this->get_logger(), "Created new output directory: %s", output_dir_.c_str());
            
        } catch (const fs::filesystem_error& e) {
            RCLCPP_ERROR(this->get_logger(), "Filesystem error: %s", e.what());
            throw;
        }
    }

    // Transform position from OptiTrack frame to world frame
    torch::Tensor transform_to_world_frame(const torch::Tensor& pos_optitrack) {
        double X = -pos_optitrack[2].item<double>() + 0.7;
        double Y = -pos_optitrack[0].item<double>() + 0.0;
        double Z = pos_optitrack[1].item<double>() + 0.5106 + 0.05375; //flange eight + frame offset
        return torch::tensor({X, Y, Z});
    }

    // Transform velocity from OptiTrack frame to world frame
    torch::Tensor transform_velocity_to_world_frame(const torch::Tensor& vel_optitrack) {
        double vx = -vel_optitrack[2].item<double>();
        double vy = -vel_optitrack[0].item<double>();
        double vz = vel_optitrack[1].item<double>();
        return torch::tensor({vx, vy, vz});
    }

    // Apply SMA filter on the joint velocities
    torch::Tensor apply_moving_average_joint_vel(const torch::Tensor& new_joint_vel) {
        // Add new velocity to history
        joint_vel_history_.push_back(new_joint_vel.clone());
        
        // Keep only the last JOINT_VEL_WINDOW_SIZE elements
        if (joint_vel_history_.size() > joint_vel_window_size_) {
            joint_vel_history_.pop_front();
        }
        
        // Calculate moving average
        if (joint_vel_history_.size() == 1) {
            // For the first measurement, return the value itself
            return new_joint_vel.clone();
        }
        
        // Stack all velocities in history and compute mean
        std::vector<torch::Tensor> vel_stack;
        for (const auto& vel : joint_vel_history_) {
            vel_stack.push_back(vel);
        }
        
        torch::Tensor stacked_vels = torch::stack(vel_stack, 0);  // Shape: [history_size, 7]
        torch::Tensor averaged_vel = torch::mean(stacked_vels, 0);  // Shape: [7]
        
        return averaged_vel;
    }

    // Check if ball X position is within threshold
    bool is_ball_x_within_threshold() {
        if (tennisball_pos_world_.numel() == 0) {
            return false;  // Return false if no ball data
        }
        
        double ball_x = tennisball_pos_world_[0].item<double>();
        double ball_z = tennisball_pos_world_[2].item<double>();

        if (fake_ball_state_){
            return true;  // If fake ball state, always return true
        }

        else {
            return (ball_x <= max_ball_x_position_); // Check if ball X position is within the threshold
        }
    }

    // Get current ball X position
    double get_ball_x_position() {
        if (tennisball_pos_world_.numel() == 0) {
            return std::numeric_limits<double>::max();  // Return large value if no data
        }
        return tennisball_pos_world_[0].item<double>();
    }

    // Helper function to convert PyTorch tensor to std::vector<double>
    std::vector<double> tensor_to_vector(const torch::Tensor& tensor) {
        std::vector<double> result;
        result.reserve(tensor.numel());
        
        auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
        
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
            for (int i = 0; i < cpu_tensor.numel(); i++) {
                result.push_back(cpu_tensor.flatten()[i].item<double>());
            }
        }
        
        return result;
    }

    void load_observation_data() {
        // Load CSV data 
        std::string input_file_path = "/home/user/kuka_rl_ros2/src/catch_and_throw/input_files/idealpd/ft_idealpd_1.csv";
        auto data = CSVReader::read(input_file_path);
        
        // Extract data from CSV 
        for (const auto& row : data) {
            // Ensure row has at least 68 columns to prevent out-of-bounds access
            // if (row.size() < 68) {
            //     std::cerr << "Skipping row with insufficient columns" << std::endl;
            //     continue;
            // }
            
            // Tennis ball position (columns 42-44)
            std::vector<float> position_row = {
                row[35], 
                row[36], 
                row[37]
            };
            tennisball_pos_csv.push_back(position_row);
            
            // Tennis ball linear velocity (columns 45-47)
            std::vector<float> velocity_row = {
                row[38], 
                row[39], 
                row[40]
            };
            tennisball_lin_vel_csv.push_back(velocity_row);

                        // Tennis ball linear velocity (columns 45-47)
            std::vector<double> joint_pos_row = {
                row[21], 
                row[22], 
                row[23],
                row[24],
                row[25],
                row[26],
                row[27]
            };
            joint_recorded_pos_csv.push_back(joint_pos_row);
            
            
        }
    }

    void compute_action_and_publish() {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if we have all necessary data
        if (joint_positions_obs_.numel() == 0 || joint_velocities_obs_.numel() == 0 ||
            ee_pose_.numel() == 0 || ee_orientation_.numel() == 0 || 
            ee_vel_.numel() == 0 || ee_angular_vel_.numel() == 0 ||
            tennisball_pos_world_.numel() == 0 || tennisball_lin_vel_world_ema_.numel() == 0) {
            return;
        }

        // Apply moving average filter to joint velocities
        joint_vel_smoothed_ = apply_moving_average_joint_vel(joint_velocities_obs_);    

        // Scale joint observations
        auto dof_pos_scaled_obs = 2.0 * (joint_positions_obs_ - robot_dof_lower_limits_np_) / 
            (robot_dof_upper_limits_np_ - robot_dof_lower_limits_np_) - 1.0;
        auto dof_vel_scaled_obs = joint_vel_smoothed_ * dof_vel_scale_;


        if (fake_ball_state_) {
            if (t_ >= tennisball_pos_csv.size()) {
            RCLCPP_WARN(this->get_logger(), "Time index exceeded data length. Shutting down node.");
            rclcpp::shutdown();
            return;
        }
        torch::Tensor tennisball_pos_obs = torch::tensor(tennisball_pos_csv[t_])* tennis_ball_pos_scale_;
        torch::Tensor tennisball_lin_vel_obs = torch::tensor(tennisball_lin_vel_csv[t_]) * lin_vel_scale_;
        // RCLCPP_INFO(this->get_logger(), "Tenisball Position x: %.3f, y: %.3f, z: %.3f", tennisball_pos_obs[0].item<double>(), 
        //     tennisball_pos_obs[1].item<double>(), tennisball_pos_obs[2].item<double>());
        // RCLCPP_INFO(this->get_logger(), "Tenisball Velocity x: %.3f, y: %.3f, z: %.3f", tennisball_lin_vel_obs[0].item<double>(), 
        //     tennisball_lin_vel_obs[1].item<double>(), tennisball_lin_vel_obs[2].item<double>());
        }

        else{
            Scale observations (using world frame data with EMA filtered ball velocity)
            torch::Tensor tennisball_pos_obs = tennisball_pos_world_ * tennis_ball_pos_scale_;
            torch::Tensor tennisball_lin_vel_obs = tennisball_lin_vel_world_ema_ * lin_vel_scale_;  // Use EMA filtered velocity
        }

        torch::Tensor ee_lin_vel_scaled = ee_vel_ * lin_vel_scale_;


        // Get current action (use last action if this is the first iteration)
        torch::Tensor current_action = action_history_.numel() == 0 ? 
            torch::zeros(7) : action_history_[-1];

        // Update action history
        action_history_ = torch::roll(action_history_, -1, 0);
        action_history_[-1] = current_action;

        // Compute offset point
        torch::Tensor ee_pos_for_obs;
        if (ee_pose_.numel() > 0 && ee_orientation_.numel() > 0) {
            ee_pos_for_obs = compute_offset_point(ee_pose_, ee_orientation_);
        } else {
            ee_pos_for_obs = torch::tensor({0.530416, -0.4544952, 0.693097});
        }

        // Prepare observation tensor based on the reduced observation space
        // From Python: obs_actor with 41 elements
        torch::Tensor observations = torch::cat({
            action_history_.flatten(),                   // 14 - self.action_history_actor.flatten(start_dim=1) (2x7)
            dof_pos_scaled_obs.flatten(),               // 7  - self.dof_pos_scaled
            dof_vel_scaled_obs.flatten(),               // 7  - self.joint_vel * self.cfg.dof_velocity_scale
            tennisball_pos_obs.flatten(),               // 3  - self.tennisball_pos * self.cfg.tennis_ball_pos_scale
            tennisball_lin_vel_obs.flatten(),           // 3  - self.tennisball_lin_vel * self.cfg.lin_vel_scale (NOW EMA FILTERED)
            ee_lin_vel_scaled.flatten(),                // 3  - self.end_effector_lin_vel * self.cfg.lin_vel_scale
            ee_orientation_.flatten()                   // 4  - self.end_effector_rot
        });

        // Sanity check observation size (should be 41 for the reduced observation space)
        assert(observations.size(0) == 41);

        // Get action from policy
        torch::Tensor raw_action = policy_->get_action(observations);
        
        // Store the action for next iteration
        current_action = raw_action.clone();

        previous_joint_pos = robot_dof_targets_.clone();

        // Compute new targets
        torch::Tensor targets = joint_positions_obs_ + (raw_action * action_scale_);
        robot_dof_targets_ = torch::clamp(
            targets,
            0.975 * robot_dof_lower_limits_,
            0.975 * robot_dof_upper_limits_
        );

        // Check ball X position and decide whether to publish commands
        double ball_x = get_ball_x_position();
        double ball_z = tennisball_pos_world_[2].item<double>();
        bool should_publish = is_ball_x_within_threshold();

        // RCLCPP_INFO(this->get_logger(), "Started recording - Robot is active (ball X: %.3f <= %.3f), (ball Z: %.3f >= %.3f), (ball Y: %.3f)", 
        //                ball_x, max_ball_x_position_, 
        //                ball_z, min_ball_z_position_,
        //                tennisball_pos_world_[1].item<double>()); 

        // Recording logic: start recording when robot starts moving, stop when it stops
        if (should_publish && !recording_active_) {
            recording_active_ = true;
            RCLCPP_INFO(this->get_logger(), "Started recording - Robot is active (ball X: %.3f <= %.3f), (ball Z: %.3f >= %.3f), (ball Y: %.3f)", 
                       ball_x, max_ball_x_position_, 
                       ball_z, min_ball_z_position_,
                       tennisball_pos_world_[1].item<double>()); 
        } else if (!should_publish && recording_active_) {
            recording_active_ = false;
            RCLCPP_INFO(this->get_logger(), "Stopped recording - Robot stopped (ball X: %.3f > %.3f), (ball Z: %.3f < %.3f, (ball Y: %.3f))", 
                       ball_x, max_ball_x_position_, 
                       ball_z, min_ball_z_position_,
                    tennisball_pos_world_[1].item<double>());
        } 

        // Only record data when recording is active (robot is moving)
        if (recording_active_) {
            // Save outputs for logging/debugging
            joint_targets_.push_back(tensor_to_vector(targets));
            joint_poses_.push_back(tensor_to_vector(joint_positions_obs_));
            joint_velocities_.push_back(tensor_to_vector(joint_velocities_obs_));
            discrete_joint_velocities_smoothed_.push_back(tensor_to_vector(joint_vel_smoothed_));  // Smoothed velocity
            ee_poses_.push_back(tensor_to_vector(ee_pos_for_obs));
            ee_orientations_.push_back(tensor_to_vector(ee_orientation_));
            ee_velocities_.push_back(tensor_to_vector(ee_vel_));
            ball_positions_.push_back(tensor_to_vector(tennisball_pos_world_));
            ball_velocities_raw_.push_back(tensor_to_vector(tennisball_lin_vel_world_));  // Raw velocities
            ball_velocities_ema_.push_back(tensor_to_vector(tennisball_lin_vel_world_ema_));  // EMA filtered velocities
            raw_actions_.push_back(tensor_to_vector(raw_action));
            ball_x_positions_.push_back({ball_x});
            command_published_.push_back({should_publish ? 1.0 : 0.0});
        }

        // Update target reference which will be then interpolated
        if (fake_joint_state_) {
            q_interp_ref = joint_recorded_pos_csv[t_];
            // std::cout << "\n";
            // std::cout << "q_reference (to be interpolated) at sample t: " << t_ << "\n";
            // std::cout << q_interp_ref << "\n";
        }

        else {
            q_interp_ref = tensor_to_vector(robot_dof_targets_);
        }
                

        //Update the flag for new reference to trigger interpolation in the other thread
        updated_ref = true;
        
        //At first iteration, run the other thread to command the robot
        if(!first_ref && first_joint_cb){
            RCLCPP_INFO(this->get_logger(), "First iteration, starting robot command thread.");
            q_interp_final = initial_positions;
            first_ref = true;     
            timer_robot = this->create_wall_timer(std::chrono::duration<double>(dt_robot), std::bind(&JointStateNode::robot_command_callback, this));
        }        

        t_++;
    }

    torch::Tensor compute_offset_point(const torch::Tensor& position, const torch::Tensor& orientation) {
        // Compute rotation matrix from quaternion
        Eigen::Quaterniond quat(
            orientation[0].item<double>(), 
            orientation[1].item<double>(), 
            orientation[2].item<double>(), 
            orientation[3].item<double>()
        );
        Eigen::Matrix3d rot_matrix = quat.normalized().toRotationMatrix();

        // Offset vector (along z-axis)
        Eigen::Vector3d offset_local(0, 0, 0.05375);

        // Compute global offset
        Eigen::Vector3d global_offset = rot_matrix * offset_local;

        // Compute new point position
        return torch::tensor({
            position[0].item<double>() + global_offset(0),
            position[1].item<double>() + global_offset(1),
            position[2].item<double>() + global_offset(2)
        });
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
    }

    ~JointStateNode(){
        RCLCPP_INFO(this->get_logger(), "Destructor called - Saving recorded data...");
        if (!joint_targets_.empty()) {
            // RCLCPP_INFO(this->get_logger(), "Saving recorded data (%zu samples)...", joint_targets_.size());
            
            // Save various outputs
            save_output(joint_targets_, output_dir_ + "/tm_received_joint_target_np.csv", 
                        "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
            save_output(joint_poses_, output_dir_ + "/tm_received_joint_pos_np.csv", 
                        "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
            save_output(joint_velocities_, output_dir_ + "/tm_received_joint_vel_np.csv", 
                        "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
            save_output(discrete_joint_velocities_smoothed_, output_dir_ + "/tm_discrete_joint_vel_smoothed_np.csv", 
                        "joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6");
            save_output(ee_poses_, output_dir_ + "/tm_received_ee_pos_np.csv", 
                        "pos_X,pos_Y,pos_Z");
            save_output(ee_orientations_, output_dir_ + "/tm_received_ee_orientation_np.csv", 
                        "or_w,or_x,or_y,or_z");
            save_output(ee_velocities_, output_dir_ + "/tm_received_ee_vel_np.csv", 
                        "lin_vel_X,lin_vel_Y,lin_vel_Z");
            
            // Save ball tracking data (world frame)
            save_output(ball_positions_, output_dir_ + "/tm_ball_positions_world_np.csv", 
                        "ball_x,ball_y,ball_z");
            save_output(ball_velocities_raw_, output_dir_ + "/tm_ball_velocities_raw_world_np.csv", 
                        "ball_vx,ball_vy,ball_vz");
            save_output(ball_velocities_ema_, output_dir_ + "/tm_ball_velocities_ema_world_np.csv", 
                        "ball_vx_ema,ball_vy_ema,ball_vz_ema");
            
            // Save action data
            save_output(raw_actions_, output_dir_ + "/tm_raw_actions_np.csv", 
                        "action_0,action_1,action_2,action_3,action_4,action_5,action_6");
            
            // Save distance and command status
            save_output(ball_x_positions_, output_dir_ + "/tm_ball_x_positions_np.csv", 
                        "ball_x_position");
            save_output(command_published_, output_dir_ + "/tm_command_published_np.csv", 
                        "command_published");

            RCLCPP_INFO(this->get_logger(), "Model outputs saved successfully.");
        } 

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
        if(!first_joint_cb){
            first_joint_cb = true;
            initial_positions = positions;
        }

        // Convert to torch tensors
        joint_positions_obs_ = torch::tensor(positions);
        joint_velocities_obs_ = torch::tensor(velocities);

        // Initialize robot_dof_targets if not set
        if (robot_dof_targets_.numel() == 0) {
            robot_dof_targets_ = torch::tensor(positions);
        }

        // Initialize previous_joint_pos if not set
        if (previous_joint_pos.numel() == 0) {
            previous_joint_pos = torch::tensor(positions);
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

    void robot_command_callback(){

        //If the pose is updated from neural network, update init and final joint, and perform linear interpolation
        if(updated_ref){
            q_interp_init = q_interp_final;
            q_interp_final = q_interp_ref;
            n = 0;
            updated_ref = false;
        }

        if(n<n_interp){n++;}
        for(int i = 0; i < 7; i++){
            q_cmd[i] = q_interp_init[i] + double(n)/double(n_interp) * (q_interp_final[i] - q_interp_init[i]);
        }

        // std::cout << "q_init: " << q_interp_init << "\n" << 
        //         "q_final: " << q_interp_final << "\n" <<
        //         "q_cmd: " << q_cmd << "\n" <<
        //         "n: " << n << "\n" <<
        //         "n_interp: " << n_interp << std::endl << std::endl;

        // Check ball X position and decide whether to publish commands
        double ball_x = get_ball_x_position();
        bool should_publish = is_ball_x_within_threshold();

        // Publish joint commands only if ball is within distance threshold
        
        
        if (should_publish) {
            //Command the robot with the interpolated command
            auto msg = lbr_fri_idl::msg::LBRJointPositionCommand();
            msg.set__joint_position((std::array<double, 7>){
                q_cmd[0], q_cmd[1], q_cmd[2], 
                q_cmd[3], q_cmd[4], q_cmd[5], 
                q_cmd[6]
            });
            joint_ref_pub_->publish(msg);
        }
    }

    // OptiTrack ball pose callback
    void ball_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto current_time = this->now();
        
        // Get raw OptiTrack position
        torch::Tensor raw_ball_pos = torch::tensor({
            msg->pose.position.x, 
            msg->pose.position.y, 
            msg->pose.position.z
        });

        // Transform to world frame
        torch::Tensor new_ball_pos_world = transform_to_world_frame(raw_ball_pos);

        // Calculate velocity if we have previous position and time
        if (tennisball_pos_world_.numel() > 0 /*&& last_ball_time_.nanoseconds() > 0*/) {
            //double dt = (current_time - last_ball_time_).seconds();
            
            torch::Tensor raw_velocity_world = (new_ball_pos_world - tennisball_pos_world_) / dt_;
            
            // Store raw velocity
            tennisball_lin_vel_world_ = raw_velocity_world;
            
            // Apply EMA filtering to ball velocity
            if (!ball_vel_ema_initialized_) {
                // Initialize EMA with the first velocity measurement
                tennisball_lin_vel_world_ema_ = tennisball_lin_vel_world_.clone();
                ball_vel_ema_initialized_ = true;
            } else {
                // EMA formula: filtered_value = alpha * new_value + (1 - alpha) * previous_filtered_value
                tennisball_lin_vel_world_ema_ = ball_vel_ema_alpha_ * tennisball_lin_vel_world_ + 
                                                (1.0 - ball_vel_ema_alpha_) * tennisball_lin_vel_world_ema_;
            }
            
            // Store time for next calculation
            //last_ball_time_ = current_time;
            
        } else {
            // Initialize velocity to zero for first measurement
            tennisball_lin_vel_world_ = torch::zeros(3);
            tennisball_lin_vel_world_ema_ = torch::zeros(3);
            //last_ball_time_ = current_time;
        }

        // Update ball position in world frame
        tennisball_pos_world_ = new_ball_pos_world;

        // RCLCPP_INFO(this->get_logger(), "(ball X: %.3f), (ball Y: %.3f), (ball Z: %.3f)", 
        //                tennisball_pos_world_[0].item<double>(), 
        //                tennisball_pos_world_[1].item<double>(),
        //                tennisball_pos_world_[2].item<double>()); 
    }

    // Member variables
    // Time and recording
    int t_;
    bool recording_;
    bool recording_active_;  // New flag to track active recording state
    std::string output_dir_; // Store output directory path

    // Synchronization
    std::mutex mutex_;

    // ROS2 Communication
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr ee_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ee_vel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ball_pose_sub_;
    rclcpp::Publisher<lbr_fri_idl::msg::LBRJointPositionCommand>::SharedPtr joint_ref_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    //Interpolation
    bool updated_ref = false;
    bool first_ref = false;
    rclcpp::TimerBase::SharedPtr timer_robot;
    std::vector<double> q_interp_init;
    std::vector<double> q_interp_final;
    std::vector<double> q_interp_ref;
    std::vector<double> q_cmd;
    int n = 0;
    float dt_nn = 0.01;
    float dt_robot = 0.005;
    int n_interp = floor(dt_nn/dt_robot);
    std::vector<double> initial_positions;
    bool first_joint_cb = false;

    // Hyperparameters
    double dt_;
    double tennis_ball_pos_scale_;
    double lin_vel_scale_;
    double dof_vel_scale_;
    double action_scale_;
    double max_ball_x_position_;
    double min_ball_z_position_;  // Minimum Z position for command publishing
    double ball_vel_ema_alpha_;  // EMA smoothing factor for ball velocity
    int joint_vel_window_size_;  // Window size for joint velocity moving average
    bool fake_joint_state_;  // Flag to indicate if joint states are fake (from CSV)
    bool fake_ball_state_;  // Flag to indicate if ball tracking is fake (from CSV)

    // Ball tracking (world frame)
    torch::Tensor tennisball_pos_world_;
    torch::Tensor tennisball_lin_vel_world_;  // Raw velocity
    torch::Tensor tennisball_lin_vel_world_ema_;  // EMA filtered velocity
    bool ball_vel_ema_initialized_;
    rclcpp::Time last_ball_time_;

    //Fake trajectory from csv
    std::vector<std::vector<float>> tennisball_pos_csv;
    std::vector<std::vector<float>> tennisball_lin_vel_csv;

    //Fake joint trajectory
    std::vector<std::vector<double>> joint_recorded_pos_csv;

    // Robot base position
    torch::Tensor robot_base_pos_;

    // Tensors for observations and targets
    torch::Tensor joint_positions_obs_;
    torch::Tensor joint_velocities_obs_;
    torch::Tensor ee_pose_;
    torch::Tensor ee_pos_for_obs;
    torch::Tensor ee_orientation_;
    torch::Tensor ee_vel_;
    torch::Tensor ee_angular_vel_;
    torch::Tensor previous_joint_pos;  // To compute joint velocities

    std::deque<torch::Tensor> joint_vel_history_;
    torch::Tensor joint_vel_smoothed_;

    // Policy and action-related tensors
    std::unique_ptr<DeploymentPolicy> policy_;
    torch::Tensor robot_dof_lower_limits_;
    torch::Tensor robot_dof_upper_limits_;
    torch::Tensor robot_dof_lower_limits_np_;
    torch::Tensor robot_dof_upper_limits_np_;

    torch::Tensor action_history_;
    torch::Tensor robot_dof_targets_;

    // Output storage for logging/debugging
    std::vector<std::vector<double>> joint_targets_;
    std::vector<std::vector<double>> joint_poses_;
    std::vector<std::vector<double>> joint_velocities_;
    // std::vector<std::vector<double>> discrete_joint_velocities_;
    std::vector<std::vector<double>> discrete_joint_velocities_smoothed_;  // Smoothed joint velocities
    std::vector<std::vector<double>> ee_poses_;
    std::vector<std::vector<double>> ee_orientations_;
    std::vector<std::vector<double>> ee_velocities_;
    std::vector<std::vector<double>> ball_positions_;
    std::vector<std::vector<double>> ball_velocities_raw_;  // Raw ball velocities
    std::vector<std::vector<double>> ball_velocities_ema_;  // EMA filtered ball velocities
    std::vector<std::vector<double>> raw_actions_;
    std::vector<std::vector<double>> ball_x_positions_;
    std::vector<std::vector<double>> command_published_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<JointStateNode>();
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        
        // Set up signal handler for graceful shutdown
        signal(SIGINT, [](int) {
            rclcpp::shutdown();
        });
        
        // This blocks until shutdown is called
        executor.spin();
        
        // Clean shutdown - let smart pointers and destructors handle cleanup
        node.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in JointStateNode: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    } catch (...) {
        std::cerr << "Unknown error in JointStateNode" << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    
    return 0;
}