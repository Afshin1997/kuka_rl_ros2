#ifndef FILE_UTILITIES_H
#define FILE_UTILITIES_H

#include <string>
#include <vector>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>

namespace fs = std::filesystem;

class FileUtilities {
public:
    // Constructor takes logger for error reporting
    FileUtilities(rclcpp::Logger logger);
    
    // CSV Reading
    static std::vector<std::vector<float>> readCSV(const std::string& filename);

    void saveConsolidatedOutput(const std::string& output_file_path,
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
                           const std::vector<std::vector<double>>& command_published);
    
    // Directory management
    void setupOutputDirectory(const std::string& output_dir);
    std::string createOutputDirSuffix(bool use_poly_regressor,
                                     double ema_alpha,
                                     int joint_vel_window_size,
                                     bool fake_ball);
    
    // File operations
    void copySourceCodeToOutput(const std::string& source_file_path, 
                               const std::string& output_dir);
    
    // Metadata saving
    void saveExperimentMetadata(const std::string& output_dir,
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
                               double action_scale);
    
    // Data output
    void saveOutput(const std::vector<std::vector<double>>& outputs,
                   const std::string& output_file_path,
                   const std::string& header = "");

private:
    rclcpp::Logger logger_;
};

#endif // FILE_UTILITIES_H