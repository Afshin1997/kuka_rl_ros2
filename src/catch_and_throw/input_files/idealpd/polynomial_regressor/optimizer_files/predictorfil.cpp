#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <iomanip>
#include "json.hpp" // nlohmann/json library - install with: apt-get install nlohmann-json3-dev

using json = nlohmann::json;

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
     * Scale input features using StandardScaler parameters
     * Formula: scaled = (input - mean) / scale
     */
    std::array<double, 7> scaleInputs(const std::array<double, 7>& inputs, int joint_idx) const {
        std::array<double, 7> scaled;
        const auto& model = joint_models[joint_idx];
        
        for (int i = 0; i < 7; i++) {
            scaled[i] = (inputs[i] - model.scaler_means[i]) / model.scaler_scales[i];
        }
        
        return scaled;
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
    
    /**
     * Batch prediction for multiple inputs
     * 
     * @param inputs Vector of input arrays
     * @return Vector of output arrays
     */
    std::vector<std::array<double, 7>> predictBatch(const std::vector<std::array<double, 7>>& inputs) const {
        std::vector<std::array<double, 7>> results;
        results.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            results.push_back(predict(input));
        }
        
        return results;
    }
    
    /**
     * Print model information
     */
    void printModelInfo() const {
        if (!is_loaded) {
            std::cout << "Model not loaded." << std::endl;
            return;
        }
        
        std::cout << "\nModel Information:" << std::endl;
        std::cout << "==================" << std::endl;
        
        for (int joint_idx = 0; joint_idx < 7; joint_idx++) {
            const auto& model = joint_models[joint_idx];
            std::cout << "Joint " << joint_idx 
                      << ": Degree " << model.degree 
                      << ", Features " << model.n_features << std::endl;
        }
    }
    
    /**
     * Check if model is loaded
     */
    bool isLoaded() const {
        return is_loaded;
    }
};

// Example usage and testing
int main() {
    try {
        // Create predictor instance
        PolynomialJointPredictor predictor;
        
        // Load model from JSON file
        std::string json_file = "polynomial_models.json";  // Change this to your JSON file path
        
        if (!predictor.loadFromJSON(json_file)) {
            std::cerr << "Failed to load model. Make sure " << json_file << " exists." << std::endl;
            return 1;
        }
        
        // Example 1: Single prediction
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "EXAMPLE 1: Single Prediction" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::array<double, 7> set_targets = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7};
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Input (set_targets):" << std::endl;
        for (int i = 0; i < 7; i++) {
            std::cout << "  set_target_" << i << ": " << set_targets[i] << std::endl;
        }
        
        auto joint_positions = predictor.predict(set_targets);
        
        std::cout << "\nPredicted joint positions:" << std::endl;
        for (int i = 0; i < 7; i++) {
            std::cout << "  joint_pos_" << i << ": " << joint_positions[i] << std::endl;
        }
        
        // Example 2: Individual joint prediction
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "EXAMPLE 2: Individual Joint Prediction" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        int target_joint = 0;
        double single_joint_pred = predictor.predictSingleJoint(set_targets, target_joint);
        std::cout << "Joint " << target_joint << " prediction: " << single_joint_pred << std::endl;
        
        // Example 3: Batch prediction
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "EXAMPLE 3: Batch Prediction" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::vector<std::array<double, 7>> batch_inputs = {
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
            {-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1}
        };
        
        auto batch_results = predictor.predictBatch(batch_inputs);
        
        for (size_t sample = 0; sample < batch_inputs.size(); sample++) {
            std::cout << "\nSample " << sample + 1 << ":" << std::endl;
            std::cout << "  Input: [";
            for (int i = 0; i < 7; i++) {
                std::cout << batch_inputs[sample][i];
                if (i < 6) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            std::cout << "  Output: [";
            for (int i = 0; i < 7; i++) {
                std::cout << batch_results[sample][i];
                if (i < 6) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Example 4: Interactive input
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "EXAMPLE 4: Interactive Input" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::array<double, 7> user_input;
        std::cout << "Enter 7 set_target values (space-separated): ";
        
        bool valid_input = true;
        for (int i = 0; i < 7; i++) {
            if (!(std::cin >> user_input[i])) {
                std::cout << "Using default values instead." << std::endl;
                user_input = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                std::cin.clear();
                std::cin.ignore(10000, '\n');
                valid_input = false;
                break;
            }
        }
        
        auto user_result = predictor.predict(user_input);
        
        std::cout << "\nYour input: [";
        for (int i = 0; i < 7; i++) {
            std::cout << user_input[i];
            if (i < 6) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Predicted joints: [";
        for (int i = 0; i < 7; i++) {
            std::cout << user_result[i];
            if (i < 6) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "\n🎉 All examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}