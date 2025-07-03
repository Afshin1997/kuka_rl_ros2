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
        
        # Scaling parameters
        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()
        
        # Polynomial parameters
        degree = self.params[joint_idx]['degree']
        powers = poly_features.powers_.tolist()
        
        # Generate feature names for documentation
        input_names = [f'set_target_{i}' for i in range(7)]
        feature_names = poly_features.get_feature_names_out(input_names)
        
        # Regression parameters
        coefficients = regressor.coef_.tolist()
        intercept = float(regressor.intercept_)
        
        # Additional metadata
        alpha = self.params[joint_idx].get('alpha', None)
        n_features = len(coefficients)
        
        return {
            'joint_index': joint_idx,
            'degree': degree,
            'n_features': n_features,
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
            'input_features': 7,
            'joint_details': []
        }
        
        for joint_idx in range(7):
            params = self.extract_joint_parameters(joint_idx)
            summary['joint_details'].append({
                'joint_index': joint_idx,
                'degree': params['degree'],
                'n_features': params['n_features'],
                'regularization': params['regression']['regularization'],
                'alpha': params['regression']['alpha']
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
                'type': 'polynomial_joint_mapper',
                'description': 'Polynomial regression models for joint position prediction',
                'input_features': 7,
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
                    'polynomial_formula': 'feature = product(input[i]^power[i] for i in range(7))',
                    'prediction_formula': 'output = intercept + sum(coeff[i] * feature[i])'
                },
                'input_names': [f'set_target_{i}' for i in range(7)],
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
            'n_inputs': 7,
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
                'structure': {
                    'scaling': 'StandardScaler parameters for input normalization',
                    'polynomial': 'Powers matrix defining which polynomial terms to compute',
                    'regression': 'Linear regression coefficients and intercept'
                },
                'implementation_steps': [
                    '1. Scale inputs: scaled = (input - mean) / scale',
                    '2. Compute polynomial features using powers matrix',
                    '3. Apply linear regression: output = intercept + sum(coeff * feature)'
                ]
            },
            'model_overview': self.create_model_summary(),
            'joints': {}
        }
        
        for joint_idx in range(7):
            params = self.extract_joint_parameters(joint_idx)
            
            readable_joint = {
                'joint_info': {
                    'index': joint_idx,
                    'polynomial_degree': params['degree'],
                    'total_features': params['n_features'],
                    'regularization': params['regression']['regularization']
                },
                'scaling_parameters': {
                    'description': 'StandardScaler parameters for input normalization',
                    'means': {f'set_target_{i}': params['scaling']['means'][i] for i in range(7)},
                    'scales': {f'set_target_{i}': params['scaling']['scales'][i] for i in range(7)}
                },
                'polynomial_features': {
                    'description': 'Powers matrix - each row defines one polynomial term',
                    'example': 'powers=[2,1,0,0,0,0,0] means set_target_0^2 * set_target_1^1',
                    'powers_matrix': params['polynomial']['powers'],
                    'feature_names': params['polynomial']['feature_names']
                },
                'regression_parameters': {
                    'description': 'Linear regression coefficients and intercept',
                    'intercept': params['regression']['intercept'],
                    'coefficients': params['regression']['coefficients'],
                    'equation': f'joint_pos_{joint_idx} = {params["regression"]["intercept"]:.6f} + sum(coeff[i] * feature[i])'
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
        
        # Generate random test inputs
        np.random.seed(42)  # For reproducible results
        test_inputs = np.random.uniform(-1, 1, (n_test_samples, 7))
        
        print(f"Testing with {n_test_samples} random samples...")
        
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
                joint_data = exported_data['joints'][f'joint_{joint_idx}']
                
                # Scale inputs
                scaled_input = []
                for j in range(7):
                    scaled = (test_input[j] - joint_data['scaling']['means'][j]) / joint_data['scaling']['scales'][j]
                    scaled_input.append(scaled)
                
                # Compute polynomial features
                features = []
                for powers in joint_data['polynomial']['powers']:
                    feature = 1.0
                    for j, power in enumerate(powers):
                        feature *= (scaled_input[j] ** power)
                    features.append(feature)
                
                # Apply regression
                prediction = joint_data['regression']['intercept']
                for coeff, feature in zip(joint_data['regression']['coefficients'], features):
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
        for joint in summary['joint_details']:
            print(f"  Joint {joint['joint_index']}: degree {joint['degree']}, {joint['n_features']} features")
        
        return {
            'standard': standard_file,
            'compact': compact_file,
            'readable': readable_file
        }


def main():
    """Main function to demonstrate usage."""
    try:
        # Initialize exporter
        exporter = PolynomialModelJSONExporter('polynomial_joint_models.pkl')
        
        # Export all formats
        files = exporter.export_all_formats('polynomial_models')
        
        print(f"\n🎉 All exports completed successfully!")
        print(f"   Use these JSON files to implement the models in C++ or other languages.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure 'polynomial_joint_models.pkl' exists in the current directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()