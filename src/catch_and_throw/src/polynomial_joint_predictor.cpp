#include "catch_and_throw/polynomial_joint_predictor.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

PolynomialJointPredictor::PolynomialJointPredictor() : is_loaded(false) {}

void PolynomialJointPredictor::printModelInfo() const {
    if (!is_loaded) {
        std::cout << "Model not loaded!" << std::endl;
        return;
    }
    
    std::cout << "=== Polynomial Joint Predictor Model Info ===" << std::endl;
    std::cout << "Sequential Model: joint_pos[t] + set_target[t] -> joint_pos[t+1]" << std::endl;
    std::cout << "Input dimensions: 14 (7 joint positions + 7 set targets)" << std::endl;
    std::cout << "Output dimensions: 7 (next joint positions)" << std::endl;
    std::cout << std::endl;
    
    for (int i = 0; i < 7; i++) {
        const auto& model = joint_models[i];
        std::cout << "Joint " << i << ":" << std::endl;
        std::cout << "  Degree: " << model.degree << std::endl;
        std::cout << "  Features: " << model.n_features << std::endl;
        std::cout << "  Polynomial terms: " << model.powers.size() << std::endl;
        std::cout << "  Intercept: " << model.intercept << std::endl;
        std::cout << "  Scaler dimensions: " << model.scaler_means.size() << std::endl;
    }
    std::cout << "=============================================" << std::endl;
}

std::vector<double> PolynomialJointPredictor::scaleInputs(const std::array<double, 14>& inputs, int joint_idx) const {
    if (!is_loaded || joint_idx < 0 || joint_idx >= 7) {
        throw std::runtime_error("Invalid joint index or model not loaded");
    }
    
    const auto& model = joint_models[joint_idx];
    
    // Validate scaler dimensions
    if (model.scaler_means.size() != 14 || model.scaler_scales.size() != 14) {
        throw std::runtime_error("Scaler dimensions mismatch. Expected 14, got " + 
                               std::to_string(model.scaler_means.size()));
    }
    
    std::vector<double> scaled(14);
    
    for (int i = 0; i < 14; i++) {
        if (model.scaler_scales[i] == 0) {
            throw std::runtime_error("Zero scale value at index " + std::to_string(i));
        }
        scaled[i] = (inputs[i] - model.scaler_means[i]) / model.scaler_scales[i];
    }
    
    return scaled;
}

bool PolynomialJointPredictor::loadFromJSON(const std::string& json_file) {
    try {
        std::ifstream file(json_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << json_file << std::endl;
            return false;
        }
        
        json data;
        file >> data;
        
        // Verify model type
        if (data.contains("model_info")) {
            auto& model_info = data["model_info"];
            if (model_info.contains("input_features")) {
                int input_features = model_info["input_features"].get<int>();
                if (input_features != 14) {
                    std::cerr << "Error: Model expects " << input_features 
                             << " input features, but this implementation requires 14" << std::endl;
                    return false;
                }
            }
        }
        
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
            
            // Validate scaling parameters are for 14 inputs
            if (model.scaler_means.size() != 14 || model.scaler_scales.size() != 14) {
                std::cerr << "Error: Expected 14 scaling parameters for joint " << joint_idx 
                         << ", got " << model.scaler_means.size() << std::endl;
                return false;
            }
            
            // Load polynomial parameters
            model.powers = joint_data["polynomial"]["powers"].get<std::vector<std::vector<int>>>();
            model.degree = joint_data["degree"].get<int>();
            model.n_features = joint_data["n_features"].get<int>();
            
            // Validate powers dimensions
            for (const auto& power_vec : model.powers) {
                if (power_vec.size() != 14) {
                    std::cerr << "Error: Power vector should have 14 elements for joint " 
                             << joint_idx << ", got " << power_vec.size() << std::endl;
                    return false;
                }
            }
            
            // Load regression parameters
            model.coefficients = joint_data["regression"]["coefficients"].get<std::vector<double>>();
            model.intercept = joint_data["regression"]["intercept"].get<double>();
            
            // Validate sizes
            if (model.powers.size() != model.coefficients.size()) {
                std::cerr << "Error: Powers and coefficients size mismatch for joint " << joint_idx << std::endl;
                return false;
            }
            
            std::cout << "Loaded joint " << joint_idx << " model successfully" << std::endl;
        }
        
        is_loaded = true;
        std::cout << "All models loaded successfully!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading JSON: " << e.what() << std::endl;
        return false;
    }
}

double PolynomialJointPredictor::computePolynomialFeature(const std::array<double, 14>& scaled_inputs, 
                                                          const std::vector<int>& powers) const {
    if (powers.size() != 14) {
        throw std::runtime_error("Powers vector must have 14 elements");
    }
    
    double result = 1.0;
    
    for (int i = 0; i < 14; i++) {
        if (powers[i] > 0) {
            result *= std::pow(scaled_inputs[i], powers[i]);
        }
    }
    
    return result;
}

double PolynomialJointPredictor::predictSingleJoint(const std::array<double, 7>& current_joint_pos,
                                                   const std::array<double, 7>& set_targets, 
                                                   int joint_idx) const {
    if (!is_loaded) {
        throw std::runtime_error("Model not loaded. Call loadFromJSON() first.");
    }
    
    if (joint_idx < 0 || joint_idx >= 7) {
        throw std::invalid_argument("Joint index must be 0-6");
    }
    
    // Combine inputs into single 14-element array
    std::array<double, 14> combined_inputs;
    for (int i = 0; i < 7; i++) {
        combined_inputs[i] = current_joint_pos[i];      // First 7: current joint positions
        combined_inputs[i + 7] = set_targets[i];        // Next 7: set targets
    }
    
    const auto& model = joint_models[joint_idx];
    
    // Step 1: Scale inputs
    auto scaled_inputs_vec = scaleInputs(combined_inputs, joint_idx);
    std::array<double, 14> scaled_inputs;
    std::copy(scaled_inputs_vec.begin(), scaled_inputs_vec.end(), scaled_inputs.begin());
    
    // Step 2: Start with intercept
    double result = model.intercept;
    
    // Step 3: Add polynomial terms
    for (size_t i = 0; i < model.powers.size(); i++) {
        double feature = computePolynomialFeature(scaled_inputs, model.powers[i]);
        result += model.coefficients[i] * feature;
    }
    
    return result;
}

std::array<double, 7> PolynomialJointPredictor::predict(const std::array<double, 7>& current_joint_pos,
                                                       const std::array<double, 7>& set_targets) const {
    std::array<double, 7> next_joint_positions;
    
    for (int joint_idx = 0; joint_idx < 7; joint_idx++) {
        next_joint_positions[joint_idx] = predictSingleJoint(current_joint_pos, set_targets, joint_idx);
    }
    
    return next_joint_positions;
}

std::array<double, 7> PolynomialJointPredictor::predict(const std::array<double, 14>& inputs) const {
    if (!is_loaded) {
        throw std::runtime_error("Model not loaded. Call loadFromJSON() first.");
    }
    
    // Extract current joint positions and set targets
    std::array<double, 7> current_joint_pos;
    std::array<double, 7> set_targets;
    
    for (int i = 0; i < 7; i++) {
        current_joint_pos[i] = inputs[i];
        set_targets[i] = inputs[i + 7];
    }
    
    return predict(current_joint_pos, set_targets);
}

int PolynomialJointPredictor::getJointDegree(int joint_idx) const {
    if (!is_loaded || joint_idx < 0 || joint_idx >= 7) {
        throw std::runtime_error("Invalid joint index or model not loaded");
    }
    return joint_models[joint_idx].degree;
}

int PolynomialJointPredictor::getJointFeatureCount(int joint_idx) const {
    if (!is_loaded || joint_idx < 0 || joint_idx >= 7) {
        throw std::runtime_error("Invalid joint index or model not loaded");
    }
    return joint_models[joint_idx].n_features;
}