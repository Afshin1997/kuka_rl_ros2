// Add this method to the JointStateNode class - insert after the setup_output_directory() method

private:
    /**
     * Generate a unique output directory name based on key parameters
     * This creates a descriptive folder name that includes the important configuration values
     */
    std::string generate_unique_output_directory() {
        // Get current timestamp for uniqueness
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream timestamp_ss;
        timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        timestamp_ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
        std::string timestamp = timestamp_ss.str();

        // Create parameter-based directory name
        std::stringstream dir_name;
        dir_name << "run_" << timestamp;
        
        // Add key parameters to the directory name
        if (fake_joint_state_) {
            if (c_q_sim_) {
                dir_name << "_cmd_joints";
            } else if (a_q_sim_) {
                dir_name << "_ach_joints";
            } else {
                dir_name << "_fake_joints";
            }
        } else {
            dir_name << "_real_joints";
        }
        
        if (fake_ball_state_) {
            dir_name << "_fake_ball";
        } else {
            dir_name << "_real_ball";
        }
        
        if (use_polynomial_regressor_) {
            dir_name << "_poly_reg";
        } else {
            dir_name << "_direct_nn";
        }
        
        // Add ball threshold parameters
        dir_name << "_ballX" << std::fixed << std::setprecision(1) << max_ball_x_position_;
        dir_name << "_ballZ" << std::fixed << std::setprecision(1) << min_ball_z_position_;
        
        // Add EMA and smoothing parameters
        if (ball_vel_ema_alpha_ != 1.0) {
            dir_name << "_ema" << std::fixed << std::setprecision(2) << ball_vel_ema_alpha_;
        }
        
        if (joint_vel_window_size_ != 5) {
            dir_name << "_jvw" << joint_vel_window_size_;
        }

        return dir_name.str();
    }

    /**
     * Modified setup_output_directory to use unique naming
     */
    void setup_output_directory() {
        try {
            // Get base output directory from parameter
            std::string base_output_dir = this->get_parameter("output_dir").as_string();
            
            // Generate unique subdirectory name
            std::string unique_dir_name = generate_unique_output_directory();
            
            // Combine base path with unique directory name
            output_dir_ = base_output_dir + "/" + unique_dir_name;
            
            // Remove the specific run directory if it exists (shouldn't happen with timestamp)
            if (fs::exists(output_dir_)) {
                fs::remove_all(output_dir_);
                RCLCPP_WARN(this->get_logger(), "Removed existing output directory: %s", output_dir_.c_str());
            }
            
            // Create fresh directory
            fs::create_directories(output_dir_);
            RCLCPP_INFO(this->get_logger(), "Created unique output directory: %s", output_dir_.c_str());
            
            // Save the parameters used for this run
            save_run_parameters();
            
            // Save the source code
            save_source_code();
            
        } catch (const fs::filesystem_error& e) {
            RCLCPP_ERROR(this->get_logger(), "Filesystem error: %s", e.what());
            throw;
        }
    }

    /**
     * Copy the source code file to the output directory
     */
    void save_source_code() {
        try {
            // Get the executable path
            std::string executable_path = get_executable_path();
            
            if (executable_path.empty()) {
                RCLCPP_WARN(this->get_logger(), "Could not determine executable path for source code backup");
                return;
            }
            
            // Try to find the source file in common locations
            std::vector<std::string> possible_source_paths = {
                // If you know the exact path to your source file, add it here:
                "/home/user/kuka_rl_ros2/src/catch_and_throw/src/joint_state_node.cpp",
                // Add more possible paths as needed
                executable_path + ".cpp",  // Sometimes works
                executable_path.substr(0, executable_path.find_last_of('/')) + "/../src/joint_state_node.cpp"
            };
            
            // You can also add the current file path as a parameter
            std::string source_file_param;
            if (this->has_parameter("source_file_path")) {
                source_file_param = this->get_parameter("source_file_path").as_string();
                if (!source_file_param.empty()) {
                    possible_source_paths.insert(possible_source_paths.begin(), source_file_param);
                }
            }
            
            std::string source_file_path;
            for (const auto& path : possible_source_paths) {
                if (fs::exists(path)) {
                    source_file_path = path;
                    break;
                }
            }
            
            if (source_file_path.empty()) {
                RCLCPP_WARN(this->get_logger(), "Could not find source file to backup. Searched paths:");
                for (const auto& path : possible_source_paths) {
                    RCLCPP_WARN(this->get_logger(), "  - %s", path.c_str());
                }
                return;
            }
            
            // Copy source file to output directory
            std::string dest_path = output_dir_ + "/source_code.cpp";
            fs::copy_file(source_file_path, dest_path, fs::copy_options::overwrite_existing);
            
            RCLCPP_INFO(this->get_logger(), "Saved source code from %s to: %s", 
                       source_file_path.c_str(), dest_path.c_str());
            
            // Also save compilation info
            save_compilation_info(source_file_path);
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to save source code: %s", e.what());
        }
    }
    
    /**
     * Get the path of the current executable
     */
    std::string get_executable_path() {
        try {
            char path[1024];
            ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
            if (len != -1) {
                path[len] = '\0';
                return std::string(path);
            }
        } catch (...) {
            // Fallback methods can be added here
        }
        return "";
    }
    
    /**
     * Save compilation and build information
     */
    void save_compilation_info(const std::string& source_path) {
        try {
            json build_info;
            
            // Compilation timestamp
            build_info["compilation_date"] = __DATE__;
            build_info["compilation_time"] = __TIME__;
            
            // Source file info
            build_info["source_file_path"] = source_path;
            build_info["executable_path"] = get_executable_path();
            
            // Get source file modification time
            if (fs::exists(source_path)) {
                auto file_time = fs::last_write_time(source_path);
                auto time_t = std::chrono::duration_cast<std::chrono::seconds>(
                    file_time.time_since_epoch()).count();
                build_info["source_file_modified"] = time_t;
            }
            
            // Compiler info (if available through preprocessor)
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
                build_info["cpp_standard"] = "Unknown";
            #endif
            
            // Build type (if available)
            #ifdef NDEBUG
                build_info["build_type"] = "Release";
            #else
                build_info["build_type"] = "Debug";
            #endif
            
            // Save to file
            std::string build_info_file = output_dir_ + "/build_info.json";
            std::ofstream file(build_info_file);
            file << build_info.dump(2);
            file.close();
            
            RCLCPP_INFO(this->get_logger(), "Saved build info to: %s", build_info_file.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to save build info: %s", e.what());
        }
    }

    /**
     * Save the parameters used for this run to a JSON file
     */
    void save_run_parameters() {
        try {
            json run_params;
            
            // Core parameters
            run_params["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Model and data paths
            run_params["model_path"] = this->get_parameter("model_path").as_string();
            run_params["polynomial_model_path"] = this->get_parameter("PolynomialModelPath").as_string();
            
            // Simulation flags
            run_params["fake_joint_state"] = fake_joint_state_;
            run_params["fake_ball_state"] = fake_ball_state_;
            run_params["use_polynomial_regressor"] = use_polynomial_regressor_;
            run_params["commanded_joint_poses_simulation"] = c_q_sim_;
            run_params["achievable_joint_poses_simulation"] = a_q_sim_;
            
            // Thresholds and limits
            run_params["max_ball_x_position"] = max_ball_x_position_;
            run_params["min_ball_z_position"] = min_ball_z_position_;
            
            // Filtering parameters
            run_params["ball_vel_ema_alpha"] = ball_vel_ema_alpha_;
            run_params["joint_vel_window_size"] = joint_vel_window_size_;
            
            // Timing parameters
            run_params["dt_nn"] = dt_nn;
            run_params["dt_robot"] = dt_robot;
            run_params["n_interp"] = n_interp;
            
            // Scaling factors
            run_params["tennis_ball_pos_scale"] = tennis_ball_pos_scale_;
            run_params["lin_vel_scale"] = lin_vel_scale_;
            run_params["dof_vel_scale"] = dof_vel_scale_;
            run_params["action_scale"] = action_scale_;
            
            // Save to file
            std::string params_file = output_dir_ + "/run_parameters.json";
            std::ofstream file(params_file);
            file << run_params.dump(2);  // Pretty print with 2-space indentation
            file.close();
            
            RCLCPP_INFO(this->get_logger(), "Saved run parameters to: %s", params_file.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to save run parameters: %s", e.what());
        }
    }

// Also add these includes at the top of the file if not already present:
// #include <iomanip>
// #include <sstream>
// #include <unistd.h>  // For readlink

// Additionally, you may want to add this parameter declaration in your constructor:
// this->declare_parameter<std::string>("source_file_path", "");  // Optional: explicit source file path