#include "catch_and_throw/experiment_manager.hpp"
#include <algorithm>

ExperimentManager::ExperimentManager(std::shared_ptr<rclcpp::Node> node) 
    : node_(node), start_time_(std::chrono::system_clock::now()) {
    // Initialize empty
}

std::string ExperimentManager::initialize_experiment(const std::string& base_output_dir) {
    try {
        // Use provided base directory or get from parameter
        std::string base_dir = base_output_dir.empty() ? 
            get_string_parameter("output_dir", "/tmp/experiments") : base_output_dir;
        
        // Generate unique directory name
        std::string unique_dir_name = generate_unique_directory_name();
        output_dir_ = base_dir + "/" + unique_dir_name;
        
        // Remove directory if it exists (shouldn't happen with timestamp)
        if (fs::exists(output_dir_)) {
            fs::remove_all(output_dir_);
            RCLCPP_WARN(node_->get_logger(), "Removed existing output directory: %s", output_dir_.c_str());
        }
        
        // Create directory
        fs::create_directories(output_dir_);
        RCLCPP_INFO(node_->get_logger(), "Created experiment directory: %s", output_dir_.c_str());
        
        // Save all metadata
        save_run_parameters();
        save_source_code();
        save_build_info();
        
        return output_dir_;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to initialize experiment: %s", e.what());
        throw;
    }
}

std::string ExperimentManager::generate_unique_directory_name() {
    // Get current timestamp with milliseconds
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream timestamp_ss;
    timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    timestamp_ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    std::string timestamp = timestamp_ss.str();

    // Build parameter-based directory name
    std::stringstream dir_name;
    dir_name << "run_" << timestamp;
    
    // Add key parameters to the directory name
    bool fake_joint_state = get_bool_parameter("fake_joint_state", false);
    bool fake_ball_state = get_bool_parameter("fake_ball_state", false);
    bool use_polynomial_regressor = get_bool_parameter("use_polynomial_regressor", false);
    bool c_q_sim = get_bool_parameter("commanded_joint_poses_simulation", false);
    bool a_q_sim = get_bool_parameter("achievable_joint_poses_simulation", false);
    
    // Joint state mode
    if (fake_joint_state) {
        if (c_q_sim) {
            dir_name << "_cmd_joints";
        } else if (a_q_sim) {
            dir_name << "_ach_joints";
        } else {
            dir_name << "_fake_joints";
        }
    } else {
        dir_name << "_real_joints";
    }
    
    // Ball tracking mode
    if (fake_ball_state) {
        dir_name << "_fake_ball";
    } else {
        dir_name << "_real_ball";
    }
    
    // Regressor type
    if (use_polynomial_regressor) {
        dir_name << "_poly_reg";
    } else {
        dir_name << "_direct_nn";
    }
    
    // Ball threshold parameters
    double max_ball_x = get_double_parameter("max_ball_x_position", 3.0);
    double min_ball_z = get_double_parameter("min_ball_z_position", -1.0);
    dir_name << "_ballX" << std::fixed << std::setprecision(1) << max_ball_x;
    dir_name << "_ballZ" << std::fixed << std::setprecision(1) << min_ball_z;
    
    // EMA and smoothing parameters (only if different from defaults)
    double ball_vel_ema_alpha = get_double_parameter("ball_vel_ema_alpha", 1.0);
    if (ball_vel_ema_alpha != 1.0) {
        dir_name << "_ema" << std::fixed << std::setprecision(2) << ball_vel_ema_alpha;
    }
    
    int joint_vel_window_size = get_int_parameter("joint_vel_window_size", 5);
    if (joint_vel_window_size != 5) {
        dir_name << "_jvw" << joint_vel_window_size;
    }

    return dir_name.str();
}

void ExperimentManager::save_source_code() {
    try {
        std::string source_file_path = find_source_file();
        
        if (source_file_path.empty()) {
            RCLCPP_WARN(node_->get_logger(), "Could not find source file to backup");
            return;
        }
        
        // Copy source file to output directory
        std::string dest_path = output_dir_ + "/source_code.cpp";
        fs::copy_file(source_file_path, dest_path, fs::copy_options::overwrite_existing);
        
        RCLCPP_INFO(node_->get_logger(), "Saved source code from %s to: %s", 
                   source_file_path.c_str(), dest_path.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(node_->get_logger(), "Failed to save source code: %s", e.what());
    }
}

std::string ExperimentManager::find_source_file() {
    // Try to find the source file in various locations
    std::vector<std::string> possible_paths;
    
    // 1. Explicit path if set
    if (!source_file_path_.empty()) {
        possible_paths.push_back(source_file_path_);
    }
    
    // 2. Parameter-specified path
    std::string param_path = get_string_parameter("source_file_path", "");
    if (!param_path.empty()) {
        possible_paths.push_back(param_path);
    }
    
    // 3. Common ROS2 workspace locations
    std::string executable_path = get_executable_path();
    if (!executable_path.empty()) {
        // Try to find workspace root and common source locations
        possible_paths.push_back(executable_path + ".cpp");
        
        // Common ROS2 patterns
        size_t build_pos = executable_path.find("/build/");
        if (build_pos != std::string::npos) {
            std::string workspace_root = executable_path.substr(0, build_pos);
            possible_paths.push_back(workspace_root + "/src/catch_and_throw/src/joint_state_node_trained_bouncing.cpp");
            possible_paths.push_back(workspace_root + "/src/*/src/*.cpp");
        }
        
        // Try relative to executable
        std::string exec_dir = executable_path.substr(0, executable_path.find_last_of('/'));
        possible_paths.push_back(exec_dir + "/../src/joint_state_node_trained_bouncing.cpp");
        possible_paths.push_back(exec_dir + "/../../src/joint_state_node_trained_bouncing.cpp");
    }
    
    // 4. Hardcoded common locations (adjust as needed)
    possible_paths.push_back("/home/user/kuka_rl_ros2/src/catch_and_throw/src/joint_state_node_trained_bouncing.cpp");
    possible_paths.push_back("./joint_state_node_trained_bouncing.cpp");
    possible_paths.push_back("../src/joint_state_node_trained_bouncing.cpp");
    
    // Search for existing file
    for (const auto& path : possible_paths) {
        if (fs::exists(path) && fs::is_regular_file(path)) {
            return path;
        }
    }
    
    // Log searched paths if not found
    RCLCPP_WARN(node_->get_logger(), "Source file not found. Searched paths:");
    for (const auto& path : possible_paths) {
        RCLCPP_WARN(node_->get_logger(), "  - %s", path.c_str());
    }
    
    return "";
}

void ExperimentManager::save_build_info() {
    try {
        json build_info;
        
        // Compilation timestamp
        build_info["compilation_date"] = __DATE__;
        build_info["compilation_time"] = __TIME__;
        
        // Source file info
        std::string source_path = find_source_file();
        build_info["source_file_path"] = source_path;
        build_info["executable_path"] = get_executable_path();
        
        // Get source file modification time
        if (!source_path.empty() && fs::exists(source_path)) {
            auto file_time = fs::last_write_time(source_path);
            auto time_t = std::chrono::duration_cast<std::chrono::seconds>(
                file_time.time_since_epoch()).count();
            build_info["source_file_modified"] = time_t;
        }
        
        // Compiler info
        #ifdef __GNUC__
            build_info["compiler"] = "GCC";
            build_info["compiler_version"] = std::to_string(__GNUC__) + "." + 
                                           std::to_string(__GNUC_MINOR__) + "." + 
                                           std::to_string(__GNUC_PATCHLEVEL__);
        #elif defined(__clang__)
            build_info["compiler"] = "Clang";
            build_info["compiler_version"] = std::to_string(__clang_major__) + "." + 
                                           std::to_string(__clang_minor__) + "." + 
                                           std::to_string(__clang_patchlevel__);
        #else
            build_info["compiler"] = "Unknown";
        #endif
        
        // C++ standard
        #if __cplusplus == 201703L
            build_info["cpp_standard"] = "C++17";
        #elif __cplusplus == 202002L
            build_info["cpp_standard"] = "C++20";
        #elif __cplusplus == 201402L
            build_info["cpp_standard"] = "C++14";
        #elif __cplusplus == 201103L
            build_info["cpp_standard"] = "C++11";
        #else
            build_info["cpp_standard"] = "Unknown (" + std::to_string(__cplusplus) + ")";
        #endif
        
        // Build type
        #ifdef NDEBUG
            build_info["build_type"] = "Release";
        #else
            build_info["build_type"] = "Debug";
        #endif
        
        // Runtime info
        build_info["experiment_start_time"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            start_time_.time_since_epoch()).count();
        
        // Save to file
        std::string build_info_file = output_dir_ + "/build_info.json";
        std::ofstream file(build_info_file);
        file << build_info.dump(2);
        file.close();
        
        RCLCPP_INFO(node_->get_logger(), "Saved build info to: %s", build_info_file.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(node_->get_logger(), "Failed to save build info: %s", e.what());
    }
}

void ExperimentManager::save_run_parameters() {
    try {
        json run_params;
        
        // Basic experiment info
        run_params["experiment_timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            start_time_.time_since_epoch()).count();
        run_params["node_name"] = node_->get_name();
        
        // Extract all declared parameters automatically
        auto param_names = node_->list_parameters({}, 0).names;
        json auto_params;
        
        for (const auto& param_name : param_names) {
            try {
                auto param = node_->get_parameter(param_name);
                switch (param.get_type()) {
                    case rclcpp::ParameterType::PARAMETER_BOOL:
                        auto_params[param_name] = param.as_bool();
                        break;
                    case rclcpp::ParameterType::PARAMETER_INTEGER:
                        auto_params[param_name] = param.as_int();
                        break;
                    case rclcpp::ParameterType::PARAMETER_DOUBLE:
                        auto_params[param_name] = param.as_double();
                        break;
                    case rclcpp::ParameterType::PARAMETER_STRING:
                        auto_params[param_name] = param.as_string();
                        break;
                    default:
                        auto_params[param_name] = param.value_to_string();
                        break;
                }
            } catch (const std::exception& e) {
                RCLCPP_DEBUG(node_->get_logger(), "Could not extract parameter %s: %s", 
                           param_name.c_str(), e.what());
            }
        }
        
        run_params["ros_parameters"] = auto_params;
        
        // Add custom parameters
        if (!custom_parameters_.empty()) {
            run_params["custom_parameters"] = custom_parameters_;
        }
        
        // Save to file
        std::string params_file = output_dir_ + "/run_parameters.json";
        std::ofstream file(params_file);
        file << run_params.dump(2);
        file.close();
        
        RCLCPP_INFO(node_->get_logger(), "Saved run parameters to: %s", params_file.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(node_->get_logger(), "Failed to save run parameters: %s", e.what());
    }
}

void ExperimentManager::save_csv(const std::vector<std::vector<double>>& data, 
                                const std::string& filename, 
                                const std::string& header) {
    if (output_dir_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "Experiment not initialized. Call initialize_experiment() first.");
        return;
    }
    
    std::string full_path = output_dir_ + "/" + filename;
    std::ofstream file(full_path);
    
    if (!file.is_open()) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to open file: %s", full_path.c_str());
        return;
    }

    // Write header if provided
    if (!header.empty()) {
        file << header << "\n";
    }

    // Write data
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i != row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    RCLCPP_DEBUG(node_->get_logger(), "Saved CSV data to: %s", full_path.c_str());
}

void ExperimentManager::generate_experiment_summary(const json& additional_info) {
    try {
        json summary;
        
        auto end_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_);
        
        summary["experiment_duration_seconds"] = duration.count();
        summary["start_time"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            start_time_.time_since_epoch()).count();
        summary["end_time"] = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time.time_since_epoch()).count();
        
        // List all files in output directory
        std::vector<std::string> output_files;
        for (const auto& entry : fs::directory_iterator(output_dir_)) {
            if (entry.is_regular_file()) {
                output_files.push_back(entry.path().filename().string());
            }
        }
        summary["output_files"] = output_files;
        
        // Add any additional information
        if (!additional_info.empty()) {
            summary["additional_info"] = additional_info;
        }
        
        // Save summary
        std::string summary_file = output_dir_ + "/experiment_summary.json";
        std::ofstream file(summary_file);
        file << summary.dump(2);
        file.close();
        
        RCLCPP_INFO(node_->get_logger(), "Generated experiment summary: %s", summary_file.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_WARN(node_->get_logger(), "Failed to generate experiment summary: %s", e.what());
    }
}

std::string ExperimentManager::get_executable_path() {
    try {
        char path[1024];
        ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
        if (len != -1) {
            path[len] = '\0';
            return std::string(path);
        }
    } catch (...) {
        // Silent fallback
    }
    return "";
}

void ExperimentManager::set_source_file_path(const std::string& source_path) {
    source_file_path_ = source_path;
}

// Parameter helper implementations
bool ExperimentManager::get_bool_parameter(const std::string& param_name, bool default_value) {
    try {
        return node_->get_parameter(param_name).as_bool();
    } catch (...) {
        return default_value;
    }
}

double ExperimentManager::get_double_parameter(const std::string& param_name, double default_value) {
    try {
        return node_->get_parameter(param_name).as_double();
    } catch (...) {
        return default_value;
    }
}

int ExperimentManager::get_int_parameter(const std::string& param_name, int default_value) {
    try {
        return node_->get_parameter(param_name).as_int();
    } catch (...) {
        return default_value;
    }
}

std::string ExperimentManager::get_string_parameter(const std::string& param_name, const std::string& default_value) {
    try {
        return node_->get_parameter(param_name).as_string();
    } catch (...) {
        return default_value;
    }
}