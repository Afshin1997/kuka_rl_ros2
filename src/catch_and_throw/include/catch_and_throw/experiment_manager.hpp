#ifndef EXPERIMENT_MANAGER_HPP
#define EXPERIMENT_MANAGER_HPP

#include <rclcpp/rclcpp.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

/**
 * @brief Utility class for managing experiment outputs, directories, and metadata
 * 
 * This class handles:
 * - Creating unique output directories based on parameters
 * - Saving source code and build information
 * - Saving run parameters and configuration
 * - CSV output utilities
 */
class ExperimentManager {
public:
    /**
     * @brief Constructor
     * @param node Shared pointer to the ROS2 node for parameter access and logging
     */
    explicit ExperimentManager(std::shared_ptr<rclcpp::Node> node);

    /**
     * @brief Initialize the experiment directory and save metadata
     * @param base_output_dir Base directory for all experiments
     * @return The created unique output directory path
     */
    std::string initialize_experiment(const std::string& base_output_dir = "");

    /**
     * @brief Save a 2D vector of data to CSV file
     * @param data 2D vector containing the data to save
     * @param filename Filename (without path, will be saved in experiment directory)
     * @param header Optional CSV header string
     */
    void save_csv(const std::vector<std::vector<double>>& data, 
                  const std::string& filename, 
                  const std::string& header = "");

    /**
     * @brief Get the current experiment output directory
     * @return Full path to the experiment directory
     */
    std::string get_output_directory() const { return output_dir_; }

    /**
     * @brief Add a custom parameter to be saved in run_parameters.json
     * @param key Parameter name
     * @param value Parameter value (supports various types)
     */
    template<typename T>
    void add_custom_parameter(const std::string& key, const T& value);

    /**
     * @brief Set the source file path explicitly (alternative to auto-detection)
     * @param source_path Full path to the source file
     */
    void set_source_file_path(const std::string& source_path);

    /**
     * @brief Generate experiment summary at the end
     * @param additional_info Any additional information to include in summary
     */
    void generate_experiment_summary(const json& additional_info = json{});

private:
    std::shared_ptr<rclcpp::Node> node_;
    std::string output_dir_;
    std::string source_file_path_;
    json custom_parameters_;
    std::chrono::system_clock::time_point start_time_;

    // Core functionality methods
    std::string generate_unique_directory_name();
    void save_source_code();
    void save_build_info();
    void save_run_parameters();
    std::string get_executable_path();
    std::string find_source_file();

    // Parameter extraction helpers
    template<typename T>
    T get_parameter_safe(const std::string& param_name, const T& default_value);
    
    bool get_bool_parameter(const std::string& param_name, bool default_value = false);
    double get_double_parameter(const std::string& param_name, double default_value = 0.0);
    int get_int_parameter(const std::string& param_name, int default_value = 0);
    std::string get_string_parameter(const std::string& param_name, const std::string& default_value = "");
};

// Template implementation for add_custom_parameter
template<typename T>
void ExperimentManager::add_custom_parameter(const std::string& key, const T& value) {
    custom_parameters_[key] = value;
}

#endif // EXPERIMENT_MANAGER_HPP