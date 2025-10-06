#ifndef POLYNOMIAL_JOINT_PREDICTOR_H
#define POLYNOMIAL_JOINT_PREDICTOR_H

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class PolynomialJointPredictor {
private:
    struct JointModel {
        std::vector<double> scaler_means;   // Size 14: 7 joint_pos + 7 set_targets
        std::vector<double> scaler_scales;  // Size 14: 7 joint_pos + 7 set_targets
        std::vector<std::vector<int>> powers;  // Each power vector has 14 elements
        std::vector<double> coefficients;
        double intercept;
        int degree;
        int n_features;
    };
    
    std::vector<JointModel> joint_models;
    bool is_loaded;
    
    // Updated to handle 14 inputs
    std::vector<double> scaleInputs(const std::array<double, 14>& inputs, int joint_idx) const;
    
    // Updated to handle 14 inputs
    double computePolynomialFeature(const std::array<double, 14>& scaled_inputs,
                                  const std::vector<int>& powers) const;

public:
    PolynomialJointPredictor();
    
    // Load model from JSON file
    bool loadFromJSON(const std::string& json_file);
    
    // Print model information
    void printModelInfo() const;
    
    // Predict single joint position for next timestep
    // Input: current joint positions (7) + set targets (7)
    double predictSingleJoint(const std::array<double, 7>& current_joint_pos,
                            const std::array<double, 7>& set_targets, 
                            int joint_idx) const;
    
    // Predict all joint positions for next timestep
    // Input: current joint positions (7) + set targets (7)
    // Output: predicted joint positions for next timestep (7)
    std::array<double, 7> predict(const std::array<double, 7>& current_joint_pos,
                                const std::array<double, 7>& set_targets) const;
    
    // Alternative predict function that takes a single 14-element array
    std::array<double, 7> predict(const std::array<double, 14>& inputs) const;
    
    // Get model info
    bool isLoaded() const { return is_loaded; }
    int getJointDegree(int joint_idx) const;
    int getJointFeatureCount(int joint_idx) const;
};

#endif // POLYNOMIAL_JOINT_PREDICTOR_H