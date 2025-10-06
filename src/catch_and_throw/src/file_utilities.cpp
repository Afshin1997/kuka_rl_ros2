#include "catch_and_throw/file_utilities.h"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

FileUtilities::FileUtilities(rclcpp::Logger logger) : logger_(logger) {}

std::vector<std::vector<float>> FileUtilities::readCSV(const std::string& filename) {
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

void FileUtilities::setupOutputDirectory(const std::string& output_dir) {
    try {
        // Remove the entire directory if it exists
        if (fs::exists(output_dir)) {
            fs::remove_all(output_dir);
        }
        
        // Create fresh directory
        fs::create_directories(output_dir);
        
    } catch (const fs::filesystem_error& e) {
        RCLCPP_ERROR(logger_, "Filesystem error: %s", e.what());
        throw;
    }
}

std::string FileUtilities::createOutputDirSuffix(bool use_poly_regressor,
                                               double ema_alpha,
                                               int joint_vel_window_size,
                                               bool fake_ball) {
    std::ostringstream oss;
    
    // Parameter 1: Polynomial regressor usage
    if (use_poly_regressor) {
        oss << "poly";
    } else {
        oss << "nn";
    }
    
    // Parameter 2: Ball velocity EMA alpha (format to 2 decimal places)
    oss << "_ema" << std::fixed << std::setprecision(2) << ema_alpha;
    
    // Parameter 3: Joint Velocity Window Size
    oss << "_jvws" << joint_vel_window_size;
    
    // Parameter 4: Ball state source
    if (fake_ball) {
        oss << "_simball";
    } else {
        oss << "_realball";
    }
    
    // Parameter 5: Timestamp (day, hour, minute)
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    // Format: _D[day]H[hour]M[minute]
    oss << "_D" << std::setfill('0') << std::setw(2) << tm.tm_mday
        << "H" << std::setfill('0') << std::setw(2) << tm.tm_hour
        << "M" << std::setfill('0') << std::setw(2) << tm.tm_min
        << "S" << std::setfill('0') << std::setw(2) << tm.tm_sec;
    
    return oss.str();
}

void FileUtilities::copySourceCodeToOutput(const std::string& source_file_path, 
                                         const std::string& output_dir) {
    try {
        // Create the destination file path
        std::string dest_file_path = output_dir + "/source_code.txt";
        
        // Check if source file exists
        if (!fs::exists(source_file_path)) {
            RCLCPP_WARN(logger_, "Source file not found: %s", source_file_path.c_str());
            return;
        }
        
        // Copy the file
        fs::copy_file(source_file_path, dest_file_path, fs::copy_options::overwrite_existing);
        
        // RCLCPP_INFO(logger_, "Source code copied to: %s", dest_file_path.c_str());
        
    } catch (const fs::filesystem_error& e) {
        RCLCPP_ERROR(logger_, "Failed to copy source code: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Error copying source code: %s", e.what());
    }
}

void FileUtilities::saveExperimentMetadata(const std::string& output_dir,
                                         bool use_polynomial_regressor,
                                         double ball_vel_ema_alpha,
                                         bool fake_ball_state,
                                         bool fake_joint_state,
                                         double max_ball_x_position,
                                         double min_ball_x_position,
                                         double min_ball_z_position,
                                         int joint_vel_window_size,
                                         bool c_q_sim,
                                         bool a_q_sim,
                                         double joint_position_tau,
                                         double dt_nn,
                                         double dt_robot,
                                         double tennis_ball_pos_scale,
                                         double lin_vel_scale,
                                         double dof_vel_scale,
                                         double action_scale) {
    try {
        std::string metadata_file = output_dir + "/experiment_metadata.txt";
        std::ofstream file(metadata_file);
        
        if (!file.is_open()) {
            RCLCPP_ERROR(logger_, "Failed to create metadata file: %s", metadata_file.c_str());
            return;
        }
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        file << "=== EXPERIMENT METADATA ===" << std::endl;
        file << "Timestamp: " << std::ctime(&time_t);
        file << std::endl;
        
        file << "=== KEY PARAMETERS ===" << std::endl;
        file << "use_polynomial_regressor: " << (use_polynomial_regressor ? "true" : "false") << std::endl;
        file << "ball_vel_ema_alpha: " << ball_vel_ema_alpha << std::endl;
        file << "fake_ball_state: " << (fake_ball_state ? "true" : "false") << std::endl;
        file << "fake_joint_state: " << (fake_joint_state ? "true" : "false") << std::endl;
        file << "max_ball_x_position: " << max_ball_x_position << std::endl;
        file << "min_ball_x_position: " << min_ball_x_position << std::endl;
        file << "min_ball_z_position: " << min_ball_z_position << std::endl;
        file << "joint_vel_window_size: " << joint_vel_window_size << std::endl;
        file << "commanded_joint_poses_simulation: " << (c_q_sim ? "true" : "false") << std::endl;
        file << "achievable_joint_poses_simulation: " << (a_q_sim ? "true" : "false") << std::endl;
        file << std::endl;
        
        file << "=== HYPERPARAMETERS ===" << std::endl;
        file << "dt_nn: " << dt_nn << std::endl;
        file << "dt_robot: " << dt_robot << std::endl;
        file << "tennis_ball_pos_scale: " << tennis_ball_pos_scale << std::endl;
        file << "lin_vel_scale: " << lin_vel_scale << std::endl;
        file << "dof_vel_scale: " << dof_vel_scale << std::endl;
        file << "action_scale: " << action_scale << std::endl;
        file << std::endl;

        file << "=== ROS NODE PARAMETERS ===" << std::endl;
        file << "joint_position_tau: " << joint_position_tau << std::endl;
        
        file.close();
        // RCLCPP_INFO(logger_, "Experiment metadata saved to: %s", metadata_file.c_str());
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Failed to save experiment metadata: %s", e.what());
    }
}

void FileUtilities::saveConsolidatedOutput(const std::string& output_file_path,
                                         const std::vector<std::vector<double>>& joint_pure_targets,
                                         const std::vector<std::vector<double>>& joint_targets,
                                         const std::vector<std::vector<double>>& joint_poses,
                                         const std::vector<std::vector<double>>& joint_velocities,
                                         const std::vector<std::vector<double>>& discrete_joint_velocities_smoothed,
                                         const std::vector<std::vector<double>>& ee_poses,
                                         const std::vector<std::vector<double>>& ee_orientations,
                                         const std::vector<std::vector<double>>& ee_velocities,
                                         const std::vector<std::vector<double>>& ee_velocities_offset,
                                         const std::vector<std::vector<double>>& ball_positions,
                                         const std::vector<std::vector<double>>& ball_velocities_raw,
                                         const std::vector<std::vector<double>>& ball_velocities_ema,
                                         const std::vector<std::vector<double>>& raw_actions,
                                         const std::vector<std::vector<double>>& command_published) {
    
    std::ofstream file(output_file_path);
    if (!file.is_open()) {
        RCLCPP_ERROR(logger_, "Failed to open consolidated output file: %s", output_file_path.c_str());
        return;
    }
    
    // Write comprehensive header
    file << "pure_target_joint_0,pure_target_joint_1,pure_target_joint_2,pure_target_joint_3,pure_target_joint_4,pure_target_joint_5,pure_target_joint_6,"
         << "target_joint_0,target_joint_1,target_joint_2,target_joint_3,target_joint_4,target_joint_5,target_joint_6,"
         << "pos_joint_0,pos_joint_1,pos_joint_2,pos_joint_3,pos_joint_4,pos_joint_5,pos_joint_6,"
         << "vel_joint_0,vel_joint_1,vel_joint_2,vel_joint_3,vel_joint_4,vel_joint_5,vel_joint_6,"
         << "vel_smooth_joint_0,vel_smooth_joint_1,vel_smooth_joint_2,vel_smooth_joint_3,vel_smooth_joint_4,vel_smooth_joint_5,vel_smooth_joint_6,"
         << "ee_pos_x,ee_pos_y,ee_pos_z,"
         << "ee_or_w,ee_or_x,ee_or_y,ee_or_z,"
         << "eep_vel_x,eep_vel_y,eep_vel_z,"
         << "ee_vel_offset_x,ee_vel_offset_y,ee_vel_offset_z,"
         << "ball_pos_x,ball_pos_y,ball_pos_z,"
         << "ball_raw_vx,ball_raw_vy,ball_raw_vz,"
         << "ball_ema_vx,ball_ema_vy,ball_ema_vz,"
         << "action_0,action_1,action_2,action_3,action_4,action_5,action_6,"
         << "command_published\n";
    
    // Get the number of rows (assuming all vectors have the same length)
    size_t num_rows = joint_targets.size();
    
    // Write data rows
    for (size_t i = 0; i < num_rows; ++i) {
        // Joint pure targets (7 values)
        for (size_t j = 0; j < 7; ++j) {
            file << joint_pure_targets[i][j] << ",";
        }
        
        // Joint targets (7 values)
        for (size_t j = 0; j < 7; ++j) {
            file << joint_targets[i][j] << ",";
        }
        
        // Joint positions (7 values)
        for (size_t j = 0; j < 7; ++j) {
            file << joint_poses[i][j] << ",";
        }
        
        // Joint velocities (7 values)
        for (size_t j = 0; j < 7; ++j) {
            file << joint_velocities[i][j] << ",";
        }
        
        // Smoothed joint velocities (7 values)
        for (size_t j = 0; j < 7; ++j) {
            file << discrete_joint_velocities_smoothed[i][j] << ",";
        }
        
        // EE position (3 values)
        for (size_t j = 0; j < 3; ++j) {
            file << ee_poses[i][j] << ",";
        }
        
        // EE orientation (4 values)
        for (size_t j = 0; j < 4; ++j) {
            file << ee_orientations[i][j] << ",";
        }
        
        // EE velocity (3 values)
        for (size_t j = 0; j < 3; ++j) {
            file << ee_velocities[i][j] << ",";
        }
        
        // EE velocity offset point (3 values)
        for (size_t j = 0; j < 3; ++j) {
            file << ee_velocities_offset[i][j] << ",";
        }
        
        // Ball position (3 values)
        for (size_t j = 0; j < 3; ++j) {
            file << ball_positions[i][j] << ",";
        }
        
        // Ball velocity raw (3 values)
        for (size_t j = 0; j < 3; ++j) {
            file << ball_velocities_raw[i][j] << ",";
        }
        
        // Ball velocity EMA (3 values)
        for (size_t j = 0; j < 3; ++j) {
            file << ball_velocities_ema[i][j] << ",";
        }
        
        // Raw actions (7 values)
        for (size_t j = 0; j < 7; ++j) {
            file << raw_actions[i][j] << ",";
        }
        
        // Command published (1 value) - no comma after last value
        file << command_published[i][0] << "\n";
    }
    
    file.close();
    // RCLCPP_INFO(logger_, "Consolidated data saved to: %s (%zu rows)", output_file_path.c_str(), num_rows);
}

void FileUtilities::saveOutput(const std::vector<std::vector<double>>& outputs,
                             const std::string& output_file_path,
                             const std::string& header) {
    std::ofstream file(output_file_path);
    if (!file.is_open()) {
        RCLCPP_ERROR(logger_, "Failed to open file: %s", output_file_path.c_str());
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