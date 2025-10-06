import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from pathlib import Path

class SequentialModelEvaluator:
    """
    A class to evaluate the performance of saved polynomial joint mapping models
    that use sequential data (joint_pos[t] + set_target[t] -> joint_pos[t+1]).
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
            print(f"✓ Model loaded successfully from '{model_path}'")
            print(f"  Configuration: {self.config}")
            print(f"  Input: joint_pos[t] (7) + set_target[t] (7) = 14 features")
            print(f"  Output: joint_pos[t+1] (7)")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please run the training script first.")
    
    def load_and_prepare_csv(self, csv_path):
        """
        Load and prepare sequential data from a single CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
            
        Returns:
        --------
        X : numpy.ndarray
            Sequential input features [joint_pos[t], set_target[t]]
        y : numpy.ndarray  
            True outputs (joint_pos[t+1])
        sequences : list
            List of (X, y, timestep) tuples for detailed analysis
        df : pandas.DataFrame
            Original dataframe
        """
        try:
            # Read CSV, keeping header and skipping rows 2 and 3
            df = pd.read_csv(csv_path, skiprows=[1, 2])
            print(f"✓ Loaded data from '{csv_path}'")
            
            # Define column names
            set_target_cols = [f'target_joint_{i}' for i in range(7)]
            joint_pos_cols = [f'pos_joint_{i}' for i in range(7)]
            
            # Check if required columns exist
            missing_cols = []
            for col in set_target_cols + joint_pos_cols:
                if col not in df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract data
            joint_positions = df[joint_pos_cols].values
            set_targets = df[set_target_cols].values
            
            # Create sequential data
            X_list = []
            y_list = []
            sequences = []
            
            for i in range(len(df) - 1):
                # Input: [joint_pos[i], set_target[i]]
                X_row = np.concatenate([joint_positions[i], set_targets[i]])
                # Output: joint_pos[i+1]
                y_row = joint_positions[i + 1]
                
                X_list.append(X_row)
                y_list.append(y_row)
                sequences.append((X_row, y_row, i))
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            print(f"  Created {len(X)} sequential samples")
            print(f"  Input shape: {X.shape}")
            print(f"  Output shape: {y.shape}")
            
            return X, y, sequences, df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def predict(self, X):
        """
        Predict next joint positions for given current state.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features [joint_pos[t], set_target[t]] shape (n_samples, 14)
            
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted joint positions for t+1, shape (n_samples, 7)
        """
        if X.shape[1] != 14:
            raise ValueError(f"Expected 14 input features, got {X.shape[1]}")
        
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
        X, y_true, sequences, original_df = self.load_and_prepare_csv(csv_path)
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate errors
        absolute_errors = np.abs(y_pred - y_true)
        squared_errors = (y_pred - y_true) ** 2
        
        # Create results dataframe
        results_data = {}
        
        # Add current joint positions (first 7 features)
        for i in range(7):
            results_data[f'current_joint_pos_{i}'] = X[:, i]
        
        # Add set targets (next 7 features)
        for i in range(7):
            results_data[f'set_target_{i}'] = X[:, i+7]
        
        # Add predictions (next joint positions)
        for i in range(7):
            results_data[f'predicted_next_joint_pos_{i}'] = y_pred[:, i]
        
        # Add true next joint positions
        for i in range(7):
            results_data[f'actual_next_joint_pos_{i}'] = y_true[:, i]
        
        # Add absolute errors
        for i in range(7):
            results_data[f'abs_error_{i}'] = absolute_errors[:, i]
        
        # Add timestep information
        results_data['timestep'] = [seq[2] for seq in sequences]
        
        results_df = pd.DataFrame(results_data)
        
        # Calculate overall metrics for each joint
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY FOR: {csv_path}")
        print(f"{'='*80}")
        print(f"Number of sequential samples: {len(X)}")
        print(f"Sequential prediction: joint_pos[t] + set_target[t] -> joint_pos[t+1]")
        print()
        
        print(f"{'Joint':<6} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<8} {'Max Error':<12}")
        print("-" * 72)
        
        joint_metrics = []
        for joint_idx in range(7):
            mse = mean_squared_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            r2 = r2_score(y_true[:, joint_idx], y_pred[:, joint_idx])
            max_error = np.max(absolute_errors[:, joint_idx])
            
            joint_metrics.append({
                'joint': joint_idx,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'max_error': max_error
            })
            
            print(f"{joint_idx:<6} {mse:<12.6f} {rmse:<12.6f} {mae:<12.6f} {r2:<8.4f} {max_error:<12.6f}")
        
        # Overall statistics
        overall_mae = np.mean(absolute_errors)
        overall_rmse = np.sqrt(np.mean(squared_errors))
        overall_r2 = np.mean([m['r2'] for m in joint_metrics])
        
        print("-" * 72)
        print(f"{'AVG':<6} {'-':<12} {overall_rmse:<12.6f} {overall_mae:<12.6f} {overall_r2:<8.4f} {'-':<12}")
        
        # Show detailed sample results
        if show_details:
            print(f"\n{'='*80}")
            print("DETAILED SEQUENTIAL PREDICTION RESULTS:")
            print(f"{'='*80}")
            
            n_show = len(X) if max_samples is None else min(max_samples, len(X))
            
            for i in range(n_show):
                print(f"\nSequence {i + 1} (timestep {sequences[i][2]} -> {sequences[i][2] + 1}):")
                print("-" * 60)
                
                # Show current state
                print("Current State (t):")
                print("  Joint Positions:")
                for j in range(7):
                    print(f"    joint_pos_{j}: {X[i, j]:8.4f}")
                
                print("  Set Targets:")
                for j in range(7):
                    print(f"    set_target_{j}: {X[i, j+7]:8.4f}")
                
                # Show predictions vs actual
                print("\nNext Joint Positions (t+1):")
                print(f"  {'Joint':<6} {'Predicted':<12} {'Actual':<12} {'Abs Error':<12}")
                print("  " + "-" * 48)
                
                for joint_idx in range(7):
                    pred_val = y_pred[i, joint_idx]
                    actual_val = y_true[i, joint_idx] 
                    error_val = absolute_errors[i, joint_idx]
                    print(f"  {joint_idx:<6} {pred_val:<12.6f} {actual_val:<12.6f} {error_val:<12.6f}")
                
                sequence_mae = np.mean(absolute_errors[i, :])
                print(f"\n  Sequence MAE: {sequence_mae:.6f}")
                
                if i >= 4 and max_samples is None:  # Show first 5 by default
                    print(f"\n... (showing first 5 sequences, {len(X)-5} more available)")
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
        X, y_true, sequences, original_df = self.load_and_prepare_csv(csv_path)
        y_pred = self.predict(X)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Main plots: Trajectory comparison for each joint
        for joint_idx in range(7):
            ax = plt.subplot(4, 2, joint_idx + 1)
            
            # Extract data for this joint
            current_pos = X[:, joint_idx]
            set_targets = X[:, joint_idx + 7]
            actual_next = y_true[:, joint_idx]
            predicted_next = y_pred[:, joint_idx]
            
            # Plot trajectories
            timesteps = np.arange(len(current_pos))
            
            # Current positions (shifted by 1 to align with predictions)
            ax.plot(timesteps, current_pos, 'b-', label='Current Position', alpha=0.7, linewidth=2)
            ax.plot(timesteps, set_targets, 'g--', label='Set Target', alpha=0.7, linewidth=1.5)
            # ax.plot(timesteps + 1, actual_next, 'r-', label='Actual Next', alpha=0.7, linewidth=2)
            ax.plot(timesteps + 1, predicted_next, 'red', linestyle=':', label='Predicted Next', linewidth=2.5)
            
            ax.set_xlabel('Sequence Index')
            ax.set_ylabel('Joint Position')
            ax.set_title(f'Joint {joint_idx} - Sequential Prediction')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        # Error distribution plot
        ax_error = plt.subplot(4, 2, 8)
        errors = y_pred - y_true
        
        # Create violin plot for error distribution
        error_data = []
        labels = []
        for i in range(7):
            error_data.append(errors[:, i])
            labels.append(f'J{i}')
        
        parts = ax_error.violinplot(error_data, positions=range(7), showmeans=True, showmedians=True)
        ax_error.set_xticks(range(7))
        ax_error.set_xticklabels(labels)
        ax_error.set_xlabel('Joint')
        ax_error.set_ylabel('Prediction Error')
        ax_error.set_title('Error Distribution by Joint')
        ax_error.grid(True, alpha=0.3, axis='y')
        ax_error.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"sequential_performance_{Path(csv_path).stem}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Performance plots saved to '{filename}'")
        
        plt.show()
        
        # Create actual vs predicted scatter plots
        fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for joint_idx in range(7):
            ax = axes[joint_idx]
            
            # Scatter plot
            ax.scatter(y_true[:, joint_idx], y_pred[:, joint_idx], 
                      alpha=0.5, s=20, c='blue', edgecolors='none')
            
            # Perfect prediction line
            min_val = min(y_true[:, joint_idx].min(), y_pred[:, joint_idx].min())
            max_val = max(y_true[:, joint_idx].max(), y_pred[:, joint_idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7)
            
            # Add metrics
            r2 = r2_score(y_true[:, joint_idx], y_pred[:, joint_idx])
            mae = mean_absolute_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Actual Next Position')
            ax.set_ylabel('Predicted Next Position')
            ax.set_title(f'Joint {joint_idx}')
            ax.grid(True, alpha=0.3)
        
        # Overall performance summary in the last subplot
        ax_summary = axes[7]
        ax_summary.axis('off')
        
        # Calculate overall metrics
        overall_r2 = np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(7)])
        overall_mae = np.mean(np.abs(y_pred - y_true))
        overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        summary_text = f"""Overall Performance Summary
        
Sequential Model: joint_pos[t] + set_target[t] → joint_pos[t+1]

Average R²: {overall_r2:.4f}
Average MAE: {overall_mae:.6f}
Average RMSE: {overall_rmse:.6f}

Total Sequences: {len(X)}
Input Dimensions: 14 (7 + 7)
Output Dimensions: 7"""
        
        ax_summary.text(0.1, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            filename2 = f"sequential_scatter_{Path(csv_path).stem}.png"
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            print(f"✓ Scatter plots saved to '{filename2}'")
        
        plt.show()
    
    def compare_with_training_performance(self):
        """
        Compare current evaluation with training performance.
        """
        if hasattr(self, 'results'):
            print(f"\n{'='*80}")
            print("COMPARISON WITH TRAINING PERFORMANCE:")
            print(f"{'='*80}")
            print(f"{'Joint':<6} {'Train R²':<12} {'Train MSE':<12} {'Train MAE':<12} {'Degree':<8} {'Features':<10}")
            print("-" * 68)
            
            for joint_idx in range(7):
                train_r2 = self.results[joint_idx].get('test_r2', 'N/A')
                train_mse = self.results[joint_idx].get('test_mse', 'N/A')
                train_mae = self.results[joint_idx].get('test_mae', 'N/A')
                degree = self.params[joint_idx]['degree']
                n_features = self.params[joint_idx].get('n_features', 'N/A')
                
                if isinstance(train_r2, float):
                    print(f"{joint_idx:<6} {train_r2:<12.4f} {train_mse:<12.6f} {train_mae:<12.6f} {degree:<8} {n_features:<10}")
                else:
                    print(f"{joint_idx:<6} {'N/A':<12} {'N/A':<12} {'N/A':<12} {degree:<8} {n_features:<10}")
        else:
            print("Training results not available in loaded model.")
    
    def analyze_temporal_performance(self, csv_path):
        """
        Analyze how prediction error changes over time/sequence.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file
        """
        # Load data and make predictions
        X, y_true, sequences, original_df = self.load_and_prepare_csv(csv_path)
        y_pred = self.predict(X)
        
        # Calculate errors for each timestep
        errors = np.abs(y_pred - y_true)
        
        # Create temporal analysis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Error over time for each joint
        for joint_idx in range(7):
            ax1.plot(errors[:, joint_idx], label=f'Joint {joint_idx}', alpha=0.7)
        
        ax1.set_xlabel('Sequence Index')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Prediction Error Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average error over time with moving average
        avg_errors = np.mean(errors, axis=1)
        ax2.plot(avg_errors, 'b-', alpha=0.5, label='Raw')
        
        # Add moving average
        window_size = min(20, len(avg_errors) // 10)
        if window_size > 1 and len(avg_errors) > window_size:
            moving_avg = np.convolve(avg_errors, np.ones(window_size)/window_size, mode='valid')
            # Calculate correct x-axis values for moving average
            x_values = np.arange(len(moving_avg)) + (window_size - 1) // 2
            ax2.plot(x_values, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
        
        ax2.set_xlabel('Sequence Index')
        ax2.set_ylabel('Average Absolute Error')
        ax2.set_title('Average Prediction Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'temporal_analysis_{Path(csv_path).stem}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Temporal analysis saved to 'temporal_analysis_{Path(csv_path).stem}.png'")


def main():
    """
    Main function to demonstrate usage.
    """
    # Initialize evaluator
    evaluator = SequentialModelEvaluator('polynomial_joint_models.pkl')
    
    # Evaluate on a single CSV file
    csv_file = "optimizer_files/recorded_data.csv"  # Update this path
    
    try:
        # Perform detailed evaluation
        results_df = evaluator.evaluate_performance(
            csv_file, 
            show_details=True, 
            max_samples=5  # Show first 5 sequences in detail
        )
        
        # Create performance plots
        evaluator.plot_performance(csv_file, save_plots=True)
        
        # Compare with training performance
        evaluator.compare_with_training_performance()
        
        # Analyze temporal performance
        # evaluator.analyze_temporal_performance(csv_file)
        
        # Save detailed results to CSV
        output_file = f"sequential_results_{Path(csv_file).stem}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Detailed results saved to '{output_file}'")
        
        # Show summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*60}")
        
        # Overall error statistics
        abs_error_cols = [col for col in results_df.columns if col.startswith('abs_error_')]
        overall_errors = results_df[abs_error_cols].values
        
        print(f"Overall MAE: {np.mean(overall_errors):.6f}")
        print(f"Overall RMSE: {np.sqrt(np.mean(overall_errors**2)):.6f}")
        print(f"Max error across all joints: {np.max(overall_errors):.6f}")
        print(f"Min error across all joints: {np.min(overall_errors):.6f}")
        print(f"Std of errors: {np.std(overall_errors):.6f}")
        
        # Per-joint summary
        print(f"\nPer-joint error ranges:")
        for i in range(7):
            joint_errors = results_df[f'abs_error_{i}'].values
            print(f"  Joint {i}: mean={np.mean(joint_errors):.6f}, "
                  f"min={np.min(joint_errors):.6f}, max={np.max(joint_errors):.6f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()