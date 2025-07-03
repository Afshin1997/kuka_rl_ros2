import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

class ModelEvaluator:
    """
    A class to evaluate the performance of saved polynomial joint mapping models.
    """
    
    def __init__(self, model_path='polynomial_joint_models.pkl'):
        """
        Load the saved model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        """
        try:
            model_data = joblib.load(model_path)
            self.models = model_data['models']
            self.params = model_data['params'] 
            self.results = model_data['results']
            self.config = model_data['config']
            print(f"Model loaded successfully from '{model_path}'")
            print(f"Model configuration: {self.config}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please run the training script first.")
    
    def load_single_csv(self, csv_path):
        """
        Load and prepare data from a single CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
            
        Returns:
        --------
        X : numpy.ndarray
            Input features (set_targets)
        y : numpy.ndarray  
            True outputs (joint_positions)
        df : pandas.DataFrame
            Original dataframe
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded data from '{csv_path}' with {len(df)} samples")
            
            # Extract features and targets
            set_target_cols = [f'set_target_{i}' for i in range(7)]
            joint_pos_cols = [f'joint_pos_{i}' for i in range(7)]
            
            # Check if required columns exist
            missing_cols = []
            for col in set_target_cols + joint_pos_cols:
                if col not in df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                print(f"Available columns: {list(df.columns)}")
            
            X = df[set_target_cols].values
            y = df[joint_pos_cols].values
            
            return X, y, df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def predict(self, X):
        """
        Predict joint positions for given set targets.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features (set_targets)
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted joint positions
        """
        predictions = np.zeros((X.shape[0], 7))
        
        for joint_idx in range(7):
            predictions[:, joint_idx] = self.models[joint_idx].predict(X)
        
        return predictions
    
    def evaluate_performance(self, csv_path, show_details=True, max_samples=None):
        """
        Evaluate model performance on a single CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        show_details : bool
            Whether to show detailed sample-by-sample results
        max_samples : int or None
            Maximum number of samples to display in detail
            
        Returns:
        --------
        results_df : pandas.DataFrame
            DataFrame containing inputs, predictions, targets, and errors
        """
        # Load data
        X, y_true, original_df = self.load_single_csv(csv_path)
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate errors
        absolute_errors = np.abs(y_pred - y_true)
        squared_errors = (y_pred - y_true) ** 2
        
        # Create results dataframe
        results_data = {}
        
        # Add input features
        for i in range(7):
            results_data[f'set_target_{i}'] = X[:, i]
        
        # Add predictions
        for i in range(7):
            results_data[f'predicted_{i}'] = y_pred[:, i]
        
        # Add true values
        for i in range(7):
            results_data[f'actual_{i}'] = y_true[:, i]
        
        # Add absolute errors
        for i in range(7):
            results_data[f'abs_error_{i}'] = absolute_errors[:, i]
        
        # Add squared errors
        for i in range(7):
            results_data[f'sq_error_{i}'] = squared_errors[:, i]
        
        results_df = pd.DataFrame(results_data)
        
        # Calculate overall metrics for each joint
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY FOR: {csv_path}")
        print(f"{'='*80}")
        print(f"Number of samples: {len(X)}")
        print()
        
        print(f"{'Joint':<6} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<8} {'Max Error':<12}")
        print("-" * 72)
        
        for joint_idx in range(7):
            mse = mean_squared_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            r2 = r2_score(y_true[:, joint_idx], y_pred[:, joint_idx])
            max_error = np.max(absolute_errors[:, joint_idx])
            
            print(f"{joint_idx:<6} {mse:<12.6f} {rmse:<12.6f} {mae:<12.6f} {r2:<8.4f} {max_error:<12.6f}")
        
        # Overall statistics
        overall_mae = np.mean(absolute_errors)
        overall_rmse = np.sqrt(np.mean(squared_errors))
        overall_r2 = np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(7)])
        
        print("-" * 72)
        print(f"{'AVG':<6} {'-':<12} {overall_rmse:<12.6f} {overall_mae:<12.6f} {overall_r2:<8.4f} {'-':<12}")
        
        # Show detailed sample results
        if show_details:
            print(f"\n{'='*80}")
            print("DETAILED SAMPLE-BY-SAMPLE RESULTS:")
            print(f"{'='*80}")
            
            n_show = len(X) if max_samples is None else min(max_samples, len(X))
            
            for sample_idx in range(n_show):
                print(f"\nSample {sample_idx + 1}:")
                print("-" * 40)
                
                # Show inputs
                print("Set Targets:")
                for i in range(7):
                    print(f"  set_target_{i}: {X[sample_idx, i]:8.4f}")
                
                print("\nJoint Positions:")
                print(f"{'Joint':<6} {'Predicted':<12} {'Actual':<12} {'Abs Error':<12}")
                print("-" * 48)
                
                for joint_idx in range(7):
                    pred_val = y_pred[sample_idx, joint_idx]
                    actual_val = y_true[sample_idx, joint_idx] 
                    error_val = absolute_errors[sample_idx, joint_idx]
                    print(f"{joint_idx:<6} {pred_val:<12.6f} {actual_val:<12.6f} {error_val:<12.6f}")
                
                sample_mae = np.mean(absolute_errors[sample_idx, :])
                print(f"\nSample MAE: {sample_mae:.6f}")
                
                if sample_idx >= 4 and max_samples is None:  # Show first 5 by default
                    print(f"\n... (showing first 5 samples, {len(X)-5} more available)")
                    break
        
        return results_df
    
    def plot_performance(self, csv_path, save_plots=True):
        """
        Create comprehensive performance visualization plots.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        save_plots : bool
            Whether to save plots to files
        """
        # Load data and make predictions
        X, y_true, original_df = self.load_single_csv(csv_path)
        y_pred = self.predict(X)
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        # Plot 1-7: Actual vs Predicted for each joint
        for joint_idx in range(7):
            ax = axes[joint_idx]
            
            ax.plot(X[:, joint_idx], 'r--', lw=2, label='Target')
            ax.plot(y_true[:, joint_idx], 'b--', lw=2, label='Joint_Position')
            ax.plot(y_pred[:, joint_idx], 'g--', lw=2, label='Predicted')
            
            # ax.set_xlabel(f'Actual Joint {joint_idx}')
            # ax.set_ylabel(f'Predicted Joint {joint_idx}')
            ax.set_title(f'Joint {joint_idx}')
            ax.grid(True, alpha=0.3)
            ax.legend()
       
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"performance_evaluation_{csv_path.split('/')[-1].replace('.csv', '')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\nPlots saved to '{filename}'")
        
        plt.show()
    
    def compare_with_training_performance(self):
        """
        Compare current evaluation with training performance.
        """
        if hasattr(self, 'results'):
            print(f"\n{'='*80}")
            print("COMPARISON WITH TRAINING PERFORMANCE:")
            print(f"{'='*80}")
            print(f"{'Joint':<6} {'Train R²':<12} {'Train MAE':<12} {'Model Degree':<12}")
            print("-" * 48)
            
            for joint_idx in range(7):
                train_r2 = self.results[joint_idx]['test_r2']
                train_mae = self.results[joint_idx]['test_mae'] 
                degree = self.params[joint_idx]['degree']
                print(f"{joint_idx:<6} {train_r2:<12.4f} {train_mae:<12.6f} {degree:<12}")
        else:
            print("Training results not available in loaded model.")


def main():
    """
    Main function to demonstrate usage.
    """
    # Initialize evaluator
    evaluator = ModelEvaluator('polynomial_joint_models.pkl')
    
    # Evaluate on a single CSV file
    csv_file = "optimizer_files/ft_idealpd_1.csv"  # Default file path
    
    if not csv_file:
        csv_file = "test_sample.csv"  # Default file
        print(f"Using default file: {csv_file}")
    
    try:
        # Perform detailed evaluation
        results_df = evaluator.evaluate_performance(
            csv_file, 
            show_details=True, 
            max_samples=10  # Show first 10 samples in detail
        )
        
        # Create performance plots
        evaluator.plot_performance(csv_file, save_plots=True)
        
        # Compare with training performance
        evaluator.compare_with_training_performance()
        
        # Save detailed results to CSV
        output_file = f"detailed_results_{csv_file.split('/')[-1]}"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to '{output_file}'")
        
        # Show some statistics about the results
        print(f"\n{'='*50}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*50}")
        
        # Overall error statistics
        abs_error_cols = [col for col in results_df.columns if col.startswith('abs_error_')]
        overall_errors = results_df[abs_error_cols].values
        
        print(f"Overall MAE: {np.mean(overall_errors):.6f}")
        print(f"Overall RMSE: {np.sqrt(np.mean(results_df[[col for col in results_df.columns if col.startswith('sq_error_')]].values)):.6f}")
        print(f"Max error across all joints: {np.max(overall_errors):.6f}")
        print(f"Min error across all joints: {np.min(overall_errors):.6f}")
        print(f"Std of errors: {np.std(overall_errors):.6f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")


if __name__ == "__main__":
    main()