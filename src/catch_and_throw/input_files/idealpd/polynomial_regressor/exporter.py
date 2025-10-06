import joblib
import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import os
from datetime import datetime

class PolynomialModelJSONExporter:
    """
    Export trained polynomial joint mapping models to JSON format.
    Creates a comprehensive JSON file with all model parameters needed for C++ implementation.
    Fixed to handle 14 input dimensions (7 joint_pos + 7 set_targets).
    """
    
    def __init__(self, model_path='polynomial_joint_models.pkl'):
        """
        Load the saved polynomial model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved .pkl model file
        """
        try:
            self.model_data = joblib.load(model_path)
            self.models = self.model_data['models']
            self.params = self.model_data['params']
            self.config = self.model_data['config']
            self.results = self.model_data.get('results', {})
            print(f"✓ Model loaded successfully from '{model_path}'")
            self._validate_model()
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _validate_model(self):
        """Validate that the model has the expected structure."""
        if len(self.models) != 7:
            raise ValueError(f"Expected 7 joint models, found {len(self.models)}")
        
        for joint_idx in range(7):
            if joint_idx not in self.models:
                raise ValueError(f"Missing model for joint {joint_idx}")
        
        print(f"✓ Model validation passed: {len(self.models)} joint models found")
    
    def extract_joint_parameters(self, joint_idx):
        """
        Extract all parameters for a specific joint model.
        
        Parameters:
        -----------
        joint_idx : int
            Joint index (0-6)
            
        Returns:
        --------
        dict: Complete parameter set for the joint
        """
        model = self.models[joint_idx]
        
        # Extract pipeline components
        scaler = model.named_steps['scaler']
        poly_features = model.named_steps['poly']
        regressor = model.named_steps['regressor']
        
        # Scaling parameters - should be 14 values (7 joint_pos + 7 set_targets)
        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()
        
        # Verify dimensions
        if len(scaler_mean) != 14:
            print(f"⚠️  Warning: Expected 14 scaling parameters, found {len(scaler_mean)}")
        
        # Polynomial parameters
        degree = self.params[joint_idx]['degree']
        powers = poly_features.powers_.tolist()
        
        # Generate feature names for documentation
        # First 7 are joint positions, next 7 are set targets
        input_names = [f'joint_pos_{i}' for i in range(7)] + [f'set_target_{i}' for i in range(7)]
        feature_names = poly_features.get_feature_names_out(input_names)
        
        # Regression parameters
        coefficients = regressor.coef_.tolist()
        intercept = float(regressor.intercept_)
        
        # Additional metadata
        alpha = self.params[joint_idx].get('alpha', None)
        n_features = len(coefficients)
        
        # Get test performance if available
        test_r2 = None
        test_mse = None
        if joint_idx in self.results:
            test_r2 = self.results[joint_idx].get('test_r2')
            test_mse = self.results[joint_idx].get('test_mse')
        
        return {
            'joint_index': joint_idx,
            'degree': degree,
            'n_features': n_features,
            'n_input_dims': 14,  # 7 joint_pos + 7 set_targets
            'performance': {
                'test_r2': test_r2,
                'test_mse': test_mse
            },
            'scaling': {
                'means': scaler_mean,
                'scales': scaler_scale,
                'method': 'StandardScaler'
            },
            'polynomial': {
                'powers': powers,
                'feature_names': feature_names.tolist(),
                'include_bias': False,
                'interaction_only': not self.config.get('include_interaction', True)
            },
            'regression': {
                'coefficients': coefficients,
                'intercept': intercept,
                'regularization': self.config.get('regularization', 'none'),
                'alpha': alpha
            }
        }
    
    def create_model_summary(self):
        """Create a summary of all joint models."""
        summary = {
            'total_joints': 7,
            'input_features': 14,  # 7 joint_pos + 7 set_targets
            'input_structure': {
                'joint_positions': list(range(0, 7)),
                'set_targets': list(range(7, 14))
            },
            'joint_details': []
        }
        
        for joint_idx in range(7):
            params = self.extract_joint_parameters(joint_idx)
            summary['joint_details'].append({
                'joint_index': joint_idx,
                'degree': params['degree'],
                'n_features': params['n_features'],
                'regularization': params['regression']['regularization'],
                'alpha': params['regression']['alpha'],
                'test_r2': params['performance']['test_r2'],
                'test_mse': params['performance']['test_mse']
            })
        
        return summary
    
    def export_to_json(self, output_file='polynomial_models.json', include_metadata=True, pretty_print=True):
        """
        Export all model parameters to JSON format.
        
        Parameters:
        -----------
        output_file : str
            Output JSON filename
        include_metadata : bool
            Whether to include metadata and documentation
        pretty_print : bool
            Whether to format JSON with indentation
            
        Returns:
        --------
        dict: The exported data structure
        """
        print("Extracting model parameters...")
        
        # Create the main export structure
        export_data = {
            'model_info': {
                'type': 'polynomial_joint_mapper_sequential',
                'description': 'Polynomial regression models for sequential joint position prediction',
                'input_features': 14,
                'input_description': 'joint_pos[t] (7) + set_target[t] (7) -> joint_pos[t+1] (7)',
                'output_joints': 7,
                'sklearn_version': 'scikit-learn pipeline',
                'export_timestamp': datetime.now().isoformat()
            },
            'configuration': self.config,
            'model_summary': self.create_model_summary(),
            'joints': {}
        }
        
        # Add metadata if requested
        if include_metadata:
            export_data['metadata'] = {
                'usage': {
                    'description': 'Each joint model follows: scaled_input -> polynomial_features -> linear_regression',
                    'scaling_formula': 'scaled_value = (input - mean) / scale',
                    'polynomial_formula': 'feature = product(input[i]^power[i] for i in range(14))',
                    'prediction_formula': 'output = intercept + sum(coeff[i] * feature[i])'
                },
                'input_structure': {
                    'indices_0_6': 'joint_pos at time t',
                    'indices_7_13': 'set_target at time t'
                },
                'input_names': [f'joint_pos_{i}' for i in range(7)] + [f'set_target_{i}' for i in range(7)],
                'output_names': [f'joint_pos_{i}' for i in range(7)],
                'feature_scaling': 'StandardScaler (zero mean, unit variance)',
                'polynomial_expansion': 'Full polynomial with interaction terms (if enabled)'
            }
        
        # Extract parameters for each joint
        print("Processing joints:")
        for joint_idx in range(7):
            print(f"  Joint {joint_idx}...", end='')
            export_data['joints'][f'joint_{joint_idx}'] = self.extract_joint_parameters(joint_idx)
            print(" ✓")
        
        # Write to file
        indent = 2 if pretty_print else None
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=indent)
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n✓ Export complete!")
        print(f"  File: {output_file}")
        print(f"  Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
        print(f"  Joints: {len(export_data['joints'])}")
        print(f"  Input dimensions: 14 (7 joint_pos + 7 set_targets)")
        
        return export_data
    
    def export_compact_json(self, output_file='polynomial_models_compact.json'):
        """
        Export a compact version with only essential parameters for C++ implementation.
        
        Parameters:
        -----------
        output_file : str
            Output JSON filename
        """
        print("Creating compact export...")
        
        compact_data = {
            'n_joints': 7,
            'n_inputs': 14,  # 7 joint_pos + 7 set_targets
            'joints': []
        }
        
        for joint_idx in range(7):
            params = self.extract_joint_parameters(joint_idx)
            
            compact_joint = {
                'idx': joint_idx,
                'degree': params['degree'],
                'means': params['scaling']['means'],
                'scales': params['scaling']['scales'],
                'powers': params['polynomial']['powers'],
                'coeffs': params['regression']['coefficients'],
                'intercept': params['regression']['intercept']
            }
            
            compact_data['joints'].append(compact_joint)
        
        with open(output_file, 'w') as f:
            json.dump(compact_data, f, separators=(',', ':'))  # No spaces for minimal size
        
        file_size = os.path.getsize(output_file)
        print(f"✓ Compact export complete: {output_file} ({file_size:,} bytes)")
        
        return compact_data
    
    def export_human_readable(self, output_file='polynomial_models_readable.json'):
        """
        Export a human-readable version with detailed explanations.
        
        Parameters:
        -----------
        output_file : str
            Output JSON filename
        """
        print("Creating human-readable export...")
        
        readable_data = {
            'README': {
                'description': 'Polynomial Joint Mapper - Human Readable Export',
                'purpose': 'This file contains all parameters needed to implement the polynomial regression models in any programming language',
                'input_format': 'Input vector: [joint_pos_0, ..., joint_pos_6, set_target_0, ..., set_target_6]',
                'output_format': 'Output vector: [joint_pos_0_next, ..., joint_pos_6_next]',
                'structure': {
                    'scaling': 'StandardScaler parameters for input normalization',
                    'polynomial': 'Powers matrix defining which polynomial terms to compute',
                    'regression': 'Linear regression coefficients and intercept'
                },
                'implementation_steps': [
                    '1. Concatenate current joint positions and set targets into 14-dimensional input',
                    '2. Scale inputs: scaled = (input - mean) / scale',
                    '3. Compute polynomial features using powers matrix',
                    '4. Apply linear regression: output = intercept + sum(coeff * feature)',
                    '5. Output is the predicted next joint positions'
                ]
            },
            'model_overview': self.create_model_summary(),
            'joints': {}
        }
        
        for joint_idx in range(7):
            params = self.extract_joint_parameters(joint_idx)
            
            # Create readable names for scaling parameters
            scaling_means = {}
            scaling_scales = {}
            for i in range(7):
                scaling_means[f'joint_pos_{i}'] = params['scaling']['means'][i]
                scaling_scales[f'joint_pos_{i}'] = params['scaling']['scales'][i]
            for i in range(7):
                scaling_means[f'set_target_{i}'] = params['scaling']['means'][i+7]
                scaling_scales[f'set_target_{i}'] = params['scaling']['scales'][i+7]
            
            readable_joint = {
                'joint_info': {
                    'index': joint_idx,
                    'polynomial_degree': params['degree'],
                    'total_features': params['n_features'],
                    'regularization': params['regression']['regularization'],
                    'test_r2': params['performance']['test_r2'],
                    'test_mse': params['performance']['test_mse']
                },
                'scaling_parameters': {
                    'description': 'StandardScaler parameters for input normalization',
                    'means': scaling_means,
                    'scales': scaling_scales
                },
                'polynomial_features': {
                    'description': 'Powers matrix - each row defines one polynomial term',
                    'example': 'powers=[2,0,0,0,0,0,0,1,0,0,0,0,0,0] means joint_pos_0^2 * set_target_0^1',
                    'powers_matrix': params['polynomial']['powers'],
                    'n_polynomial_features': len(params['polynomial']['powers']),
                    'feature_names_sample': params['polynomial']['feature_names'][:10] if len(params['polynomial']['feature_names']) > 10 else params['polynomial']['feature_names']
                },
                'regression_parameters': {
                    'description': 'Linear regression coefficients and intercept',
                    'intercept': params['regression']['intercept'],
                    'n_coefficients': len(params['regression']['coefficients']),
                    'coefficients_sample': params['regression']['coefficients'][:10] if len(params['regression']['coefficients']) > 10 else params['regression']['coefficients'],
                    'equation': f'joint_pos_{joint_idx}_next = {params["regression"]["intercept"]:.6f} + sum(coeff[i] * polynomial_feature[i])'
                }
            }
            
            readable_data['joints'][f'joint_{joint_idx}'] = readable_joint
        
        with open(output_file, 'w') as f:
            json.dump(readable_data, f, indent=4)
        
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"✓ Human-readable export complete: {output_file} ({file_size_mb:.2f} MB)")
        
        return readable_data
    
    def validate_export(self, json_file, n_test_samples=5):
        """
        Validate the exported JSON by comparing predictions with original model.
        
        Parameters:
        -----------
        json_file : str
            Path to exported JSON file
        n_test_samples : int
            Number of random test samples to validate
        """
        print(f"\nValidating export: {json_file}")
        
        # Load exported data
        with open(json_file, 'r') as f:
            exported_data = json.load(f)
        
        # Generate random test inputs (14 dimensions)
        np.random.seed(42)  # For reproducible results
        test_inputs = np.random.uniform(-1, 1, (n_test_samples, 14))
        
        print(f"Testing with {n_test_samples} random samples (14-dimensional inputs)...")
        
        max_diff = 0.0
        for i, test_input in enumerate(test_inputs):
            # Get prediction from original model
            original_pred = []
            for joint_idx in range(7):
                pred = self.models[joint_idx].predict([test_input])[0]
                original_pred.append(pred)
            
            # Simulate prediction using exported data
            exported_pred = []
            for joint_idx in range(7):
                if 'joints' in exported_data and f'joint_{joint_idx}' in exported_data['joints']:
                    joint_data = exported_data['joints'][f'joint_{joint_idx}']
                elif 'joints' in exported_data and isinstance(exported_data['joints'], list):
                    joint_data = exported_data['joints'][joint_idx]
                else:
                    raise ValueError(f"Cannot find joint {joint_idx} data in exported file")
                
                # Scale inputs
                scaled_input = []
                for j in range(14):
                    if 'scaling' in joint_data:
                        means = joint_data['scaling']['means']
                        scales = joint_data['scaling']['scales']
                    else:
                        means = joint_data['means']
                        scales = joint_data['scales']
                    
                    scaled = (test_input[j] - means[j]) / scales[j]
                    scaled_input.append(scaled)
                
                # Compute polynomial features
                features = []
                powers_data = joint_data['polynomial']['powers'] if 'polynomial' in joint_data else joint_data['powers']
                for powers in powers_data:
                    feature = 1.0
                    for j, power in enumerate(powers):
                        feature *= (scaled_input[j] ** power)
                    features.append(feature)
                
                # Apply regression
                if 'regression' in joint_data:
                    intercept = joint_data['regression']['intercept']
                    coefficients = joint_data['regression']['coefficients']
                else:
                    intercept = joint_data['intercept']
                    coefficients = joint_data['coeffs']
                
                prediction = intercept
                for coeff, feature in zip(coefficients, features):
                    prediction += coeff * feature
                
                exported_pred.append(prediction)
            
            # Compare predictions
            diff = np.abs(np.array(original_pred) - np.array(exported_pred))
            max_sample_diff = np.max(diff)
            max_diff = max(max_diff, max_sample_diff)
            
            print(f"  Sample {i+1}: max difference = {max_sample_diff:.2e}")
        
        print(f"\n✓ Validation complete!")
        print(f"  Maximum difference: {max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("  ✓ PERFECT MATCH - Export is accurate!")
        elif max_diff < 1e-6:
            print("  ✓ EXCELLENT - Export is highly accurate!")
        elif max_diff < 1e-3:
            print("  ⚠ GOOD - Small numerical differences detected")
        else:
            print("  ❌ WARNING - Significant differences detected!")
        
        return max_diff
    
    def export_all_formats(self, base_name='polynomial_models'):
        """
        Export in all available JSON formats.
        
        Parameters:
        -----------
        base_name : str
            Base filename (without extension)
        """
        print("=" * 60)
        print("POLYNOMIAL MODEL JSON EXPORTER")
        print("Sequential Model: joint_pos[t] + set_target[t] -> joint_pos[t+1]")
        print("=" * 60)
        
        # Export standard JSON
        standard_file = f"{base_name}.json"
        self.export_to_json(standard_file)
        
        # Export compact JSON
        compact_file = f"{base_name}_compact.json"
        self.export_compact_json(compact_file)
        
        # Export human-readable JSON
        readable_file = f"{base_name}_readable.json"
        self.export_human_readable(readable_file)
        
        # Validate exports
        print("\nValidating exports...")
        self.validate_export(standard_file)
        
        print("\n" + "=" * 60)
        print("EXPORT SUMMARY")
        print("=" * 60)
        print(f"Files created:")
        print(f"  - {standard_file} (standard format)")
        print(f"  - {compact_file} (minimal size)")
        print(f"  - {readable_file} (documentation)")
        
        print(f"\nModel summary:")
        summary = self.create_model_summary()
        print(f"  Input: {summary['input_features']} features (7 joint_pos + 7 set_targets)")
        print(f"  Output: {summary['total_joints']} joint positions")
        for joint in summary['joint_details']:
            r2_str = f", R²={joint['test_r2']:.3f}" if joint['test_r2'] else ""
            print(f"  Joint {joint['joint_index']}: degree {joint['degree']}, {joint['n_features']} features{r2_str}")
        
        return {
            'standard': standard_file,
            'compact': compact_file,
            'readable': readable_file
        }
    
    def export_cpp_header(self, output_file='polynomial_models.h'):
        """
        Export a C++ header file with model parameters as constants.
        
        Parameters:
        -----------
        output_file : str
            Output header filename
        """
        print(f"Creating C++ header file: {output_file}")
        
        header_content = """#ifndef POLYNOMIAL_MODELS_H
#define POLYNOMIAL_MODELS_H

#include <vector>
#include <array>

namespace PolynomialModels {

constexpr int N_JOINTS = 7;
constexpr int N_INPUTS = 14;  // 7 joint_pos + 7 set_targets

"""
        
        for joint_idx in range(7):
            params = self.extract_joint_parameters(joint_idx)
            
            header_content += f"// Joint {joint_idx} parameters\n"
            header_content += f"namespace Joint{joint_idx} {{\n"
            header_content += f"    constexpr int DEGREE = {params['degree']};\n"
            header_content += f"    constexpr int N_FEATURES = {params['n_features']};\n"
            header_content += f"    constexpr double INTERCEPT = {params['regression']['intercept']};\n"
            
            # Add arrays
            header_content += f"    const std::array<double, {len(params['scaling']['means'])}> MEANS = {{{', '.join(map(str, params['scaling']['means']))}}};\n"
            header_content += f"    const std::array<double, {len(params['scaling']['scales'])}> SCALES = {{{', '.join(map(str, params['scaling']['scales']))}}};\n"
            
            # Powers matrix (simplified for header)
            header_content += f"    // Powers matrix dimensions: [{len(params['polynomial']['powers'])}][{len(params['polynomial']['powers'][0])}]\n"
            header_content += f"    // See JSON file for full powers matrix\n"
            
            # Coefficients (first few for reference)
            n_show = min(5, len(params['regression']['coefficients']))
            header_content += f"    // First {n_show} coefficients (see JSON for all {len(params['regression']['coefficients'])})\n"
            header_content += f"    // {params['regression']['coefficients'][:n_show]}\n"
            
            header_content += f"}}\n\n"
        
        header_content += """} // namespace PolynomialModels

#endif // POLYNOMIAL_MODELS_H
"""
        
        with open(output_file, 'w') as f:
            f.write(header_content)
        
        print(f"✓ C++ header exported: {output_file}")
        
        return output_file


def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize exporter
        exporter = PolynomialModelJSONExporter('polynomial_joint_models.pkl')
        
        # Export all formats
        files = exporter.export_all_formats('polynomial_models')
        
        # Also export C++ header
        exporter.export_cpp_header('polynomial_models.h')
        
        print(f"\n🎉 All exports completed successfully!")
        print(f"   Use these JSON files to implement the models in C++ or other languages.")
        print(f"   The models expect 14-dimensional input: [joint_pos(7), set_target(7)]")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure 'polynomial_joint_models.pkl' exists in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()