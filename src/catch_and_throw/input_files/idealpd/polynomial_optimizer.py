import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class PolynomialJointMapper:

    def __init__(self, max_degree=5, include_interaction=True, regularization='none'):
        """
        Initialize the polynomial mapper.
        
        Parameters:
        -----------
        max_degree : int
            Maximum polynomial degree to test (will test from 1 to max_degree)
        include_interaction : bool
            Whether to include interaction terms (e.g., x1*x2)
        regularization : str
            Type of regularization: 'none', 'ridge', or 'lasso'
        """
        self.max_degree = max_degree
        self.include_interaction = include_interaction
        self.regularization = regularization
        self.best_models = {}
        self.best_params = {}
        self.scalers = {}
        
    def load_data(self, folder_path):
        """Load all CSV files from the specified folder."""
        all_data = []
        csv_files = Path(folder_path).glob("*.csv")
        
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            all_data.append(df)
        
        if not all_data:
            raise ValueError(f"No CSV files found in {folder_path}")
            
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} samples from {len(all_data)} files")
        
        # Extract features and targets
        set_target_cols = [f'set_target_{i}' for i in range(7)]
        joint_pos_cols = [f'joint_pos_{i}' for i in range(7)]
        
        self.X = combined_data[set_target_cols].values
        self.y = combined_data[joint_pos_cols].values
        
        return self.X, self.y
    
    def create_regression_model(self, alpha=1.0):
        """Create the appropriate regression model based on regularization type."""
        if self.regularization == 'ridge':
            return Ridge(alpha=alpha, fit_intercept=True)
        elif self.regularization == 'lasso':
            return Lasso(alpha=alpha, fit_intercept=True, max_iter=2000)
        else:
            return LinearRegression(fit_intercept=True)
    
    def find_best_polynomial_order(self, X_train, y_train, X_val, y_val, joint_idx):
        """
        Find the best polynomial order for a specific joint.
        
        Returns:
        --------
        best_degree : int
        best_score : float
        degree_scores : dict
        """
        degree_scores = {}
        
        for degree in range(1, self.max_degree + 1):
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=degree, 
                include_bias=False,
                interaction_only=not self.include_interaction
            )
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', poly),
                ('regressor', self.create_regression_model())
            ])
            
            # Fit and evaluate
            pipeline.fit(X_train, y_train[:, joint_idx])
            y_pred = pipeline.predict(X_val)
            
            mse = mean_squared_error(y_val[:, joint_idx], y_pred)
            r2 = r2_score(y_val[:, joint_idx], y_pred)
            mae = mean_absolute_error(y_val[:, joint_idx], y_pred)
            
            # Calculate number of features
            n_features = poly.fit_transform(X_train[:1]).shape[1]
            
            # Adjusted R² to penalize complexity
            n_samples = len(y_train)
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            
            degree_scores[degree] = {
                'mse': mse,
                'r2': r2,
                'adj_r2': adj_r2,
                'mae': mae,
                'n_features': n_features,
                'pipeline': pipeline
            }
        
        # Select best degree based on adjusted R²
        best_degree = max(degree_scores.keys(), 
                         key=lambda d: degree_scores[d]['adj_r2'])
        
        return best_degree, degree_scores[best_degree], degree_scores
    
    def optimize_regularization(self, X_train, y_train, X_val, y_val, joint_idx, degree):
        """
        Optimize regularization parameter if using Ridge or Lasso.
        """
        if self.regularization == 'none':
            return None
        
        # Define alpha range to test
        alphas = np.logspace(-6, 3, 50)
        best_alpha = None
        best_score = float('-inf')
        
        for alpha in alphas:
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=not self.include_interaction
            )
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', poly),
                ('regressor', self.create_regression_model(alpha=alpha))
            ])
            
            pipeline.fit(X_train, y_train[:, joint_idx])
            y_pred = pipeline.predict(X_val)
            r2 = r2_score(y_val[:, joint_idx], y_pred)
            
            if r2 > best_score:
                best_score = r2
                best_alpha = alpha
        
        return best_alpha
    
    def fit(self, X=None, y=None, test_size=0.2, cv_folds=5):
        """
        Find the best polynomial model for each joint.
        """
        if X is None:
            X, y = self.X, self.y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Further split training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        results = {}
        
        print("\nOptimizing polynomial order for each joint...")
        print("=" * 80)
        
        for joint_idx in range(7):
            print(f"\nJoint {joint_idx}:")
            print("-" * 40)
            
            # Find best polynomial order
            best_degree, best_info, all_scores = self.find_best_polynomial_order(
                X_tr, y_tr, X_val, y_val, joint_idx
            )
            
            # Optimize regularization if needed
            best_alpha = self.optimize_regularization(
                X_tr, y_tr, X_val, y_val, joint_idx, best_degree
            )
            
            # Create final model with best parameters
            poly = PolynomialFeatures(
                degree=best_degree,
                include_bias=False,
                interaction_only=not self.include_interaction
            )
            
            if best_alpha is not None:
                regressor = self.create_regression_model(alpha=best_alpha)
            else:
                regressor = self.create_regression_model()
            
            final_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', poly),
                ('regressor', regressor)
            ])
            
            # Train on full training set
            final_pipeline.fit(X_train, y_train[:, joint_idx])
            
            # Evaluate on test set
            y_pred_test = final_pipeline.predict(X_test)
            test_mse = mean_squared_error(y_test[:, joint_idx], y_pred_test)
            test_r2 = r2_score(y_test[:, joint_idx], y_pred_test)
            test_mae = mean_absolute_error(y_test[:, joint_idx], y_pred_test)
            
            # Store results
            self.best_models[joint_idx] = final_pipeline
            self.best_params[joint_idx] = {
                'degree': best_degree,
                'alpha': best_alpha,
                'n_features': best_info['n_features']
            }
            
            results[joint_idx] = {
                'best_degree': best_degree,
                'best_alpha': best_alpha,
                'validation_scores': all_scores,
                'test_mse': test_mse,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'n_features': best_info['n_features']
            }
            
            # Print results
            print(f"Best degree: {best_degree}")
            print(f"Number of features: {best_info['n_features']}")
            if best_alpha:
                print(f"Best alpha: {best_alpha:.6f}")
            print(f"Test MSE: {test_mse:.6f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test MAE: {test_mae:.6f}")
            
            # Print degree comparison
            print("\nDegree comparison (validation set):")
            for deg in sorted(all_scores.keys()):
                scores = all_scores[deg]
                print(f"  Degree {deg}: R²={scores['r2']:.4f}, "
                      f"Adj-R²={scores['adj_r2']:.4f}, "
                      f"MSE={scores['mse']:.6f}, "
                      f"Features={scores['n_features']}")
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def predict(self, X):
        """Predict joint positions for given set targets."""
        if not self.best_models:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        predictions = np.zeros((X.shape[0], 7))
        
        for joint_idx in range(7):
            predictions[:, joint_idx] = self.best_models[joint_idx].predict(X)
        
        return predictions
    
    def plot_results(self):
        """Visualize the optimization results."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1-7: Degree comparison for each joint
        for joint_idx in range(7):
            ax = axes[joint_idx]
            degrees = []
            r2_scores = []
            adj_r2_scores = []
            
            for deg in sorted(self.results[joint_idx]['validation_scores'].keys()):
                degrees.append(deg)
                r2_scores.append(self.results[joint_idx]['validation_scores'][deg]['r2'])
                adj_r2_scores.append(self.results[joint_idx]['validation_scores'][deg]['adj_r2'])
            
            ax.plot(degrees, r2_scores, 'b-o', label='R²')
            ax.plot(degrees, adj_r2_scores, 'r--s', label='Adjusted R²')
            ax.axvline(x=self.results[joint_idx]['best_degree'], 
                      color='green', linestyle=':', label='Selected')
            ax.set_xlabel('Polynomial Degree')
            ax.set_ylabel('Score')
            ax.set_title(f'Joint {joint_idx} - Degree Selection')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 8: Overall performance summary
        ax = axes[7]
        joint_indices = list(range(7))
        test_r2_scores = [self.results[i]['test_r2'] for i in joint_indices]
        best_degrees = [self.results[i]['best_degree'] for i in joint_indices]
        
        ax2 = ax.twinx()
        bars = ax.bar(joint_indices, test_r2_scores, alpha=0.7, label='Test R²')
        line = ax2.plot(joint_indices, best_degrees, 'ro-', label='Best Degree', linewidth=2)
        
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Test R² Score', color='b')
        ax2.set_ylabel('Best Polynomial Degree', color='r')
        ax.set_title('Overall Performance Summary')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, r2) in enumerate(zip(bars, test_r2_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{r2:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 9: Feature count comparison
        ax = axes[8]
        n_features = [self.results[i]['n_features'] for i in joint_indices]
        ax.bar(joint_indices, n_features, color='orange', alpha=0.7)
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Number of Polynomial Features')
        ax.set_title('Model Complexity (Feature Count)')
        ax.grid(True, alpha=0.3)
        
        for i, nf in enumerate(n_features):
            ax.text(i, nf + 0.5, str(nf), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('polynomial_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, n_samples=100):
        """Plot actual vs predicted values for test samples."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Get predictions for test set
        y_pred = self.predict(self.X_test)
        
        # Limit samples for visibility
        n_plot = min(n_samples, len(self.X_test))
        indices = np.random.choice(len(self.X_test), n_plot, replace=False)
        
        for joint_idx in range(7):
            ax = axes[joint_idx]
            ax.scatter(self.y_test[indices, joint_idx], 
                      y_pred[indices, joint_idx], 
                      alpha=0.6, s=20)
            
            # Add perfect prediction line
            min_val = self.y_test[indices, joint_idx].min()
            max_val = self.y_test[indices, joint_idx].max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # Add R² score
            r2 = self.results[joint_idx]['test_r2']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nDegree = {self.best_params[joint_idx]["degree"]}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'Actual joint_pos_{joint_idx}')
            ax.set_ylabel(f'Predicted joint_pos_{joint_idx}')
            ax.set_title(f'Joint {joint_idx}')
            ax.grid(True, alpha=0.3)
        
        # Error distribution plot
        ax = axes[7]
        errors = y_pred - self.y_test
        ax.boxplot([errors[:, i] for i in range(7)], labels=[f'J{i}' for i in range(7)])
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Distribution by Joint')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('polynomial_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, filename='polynomial_joint_models.pkl'):
        """Save the trained models and parameters."""
        model_data = {
            'models': self.best_models,
            'params': self.best_params,
            'results': self.results,
            'config': {
                'max_degree': self.max_degree,
                'include_interaction': self.include_interaction,
                'regularization': self.regularization
            }
        }
        joblib.dump(model_data, filename)
        print(f"\nModels saved to '{filename}'")
    
    def print_polynomial_equations(self, joint_idx=None, max_terms=10):
        """
        Print the polynomial equations for specified joints.
        
        Parameters:
        -----------
        joint_idx : int or None
            If None, print equations for all joints
        max_terms : int
            Maximum number of terms to display
        """
        if joint_idx is not None:
            joint_indices = [joint_idx]
        else:
            joint_indices = range(7)
        
        print("\nPolynomial Equations:")
        print("=" * 80)
        
        for idx in joint_indices:
            model = self.best_models[idx]
            poly_features = model.named_steps['poly']
            regressor = model.named_steps['regressor']
            
            # Get feature names
            feature_names = poly_features.get_feature_names_out(
                [f'x{i}' for i in range(7)]
            )
            
            # Get coefficients
            coeffs = regressor.coef_
            intercept = regressor.intercept_
            
            # Sort by absolute coefficient value
            coeff_indices = np.argsort(np.abs(coeffs))[::-1]
            
            print(f"\nJoint {idx} (Degree {self.best_params[idx]['degree']}):")
            print(f"y{idx} = {intercept:.6f}")
            
            # Print top terms
            n_terms = min(max_terms, len(coeffs))
            for i in range(n_terms):
                coeff_idx = coeff_indices[i]
                coeff = coeffs[coeff_idx]
                if abs(coeff) > 1e-6:  # Skip very small coefficients
                    feature = feature_names[coeff_idx]
                    # Replace x0, x1, etc. with set_target names
                    for j in range(7):
                        feature = feature.replace(f'x{j}', f'st{j}')
                    
                    sign = '+' if coeff > 0 else ''
                    print(f"      {sign} {coeff:.6f} * {feature}")
            
            if len(coeffs) > max_terms:
                print(f"      ... and {len(coeffs) - max_terms} more terms")


# Main execution function
def main():
    # Configuration
    folder_path = "optimizer_files"  # UPDATE THIS PATH
    
    # Test different configurations
    configs = [
        {'regularization': 'none', 'include_interaction': True, 'max_degree':5},
        {'regularization': 'ridge', 'include_interaction': True, 'max_degree': 4},
        {'regularization': 'lasso', 'include_interaction': False, 'max_degree': 3},
    ]
    
    best_config = None
    best_overall_r2 = -float('inf')
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing configuration: {config}")
        print(f"{'='*80}")
        
        # Create and train model
        mapper = PolynomialJointMapper(**config)
        X, y = mapper.load_data(folder_path)
        results = mapper.fit(X, y)
        
        # Calculate overall performance
        overall_r2 = np.mean([results[i]['test_r2'] for i in range(7)])
        print(f"\nOverall average R²: {overall_r2:.4f}")
        
        if overall_r2 > best_overall_r2:
            best_overall_r2 = overall_r2
            best_config = config
            best_mapper = mapper
    
    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATION: {best_config}")
    print(f"Best overall R²: {best_overall_r2:.4f}")
    print(f"{'='*80}")
    
    # Use the best configuration for final analysis
    best_mapper.plot_results()
    best_mapper.plot_predictions()
    best_mapper.save_models()
    best_mapper.print_polynomial_equations(max_terms=15)
    
    # Example usage
    print("\n" + "="*80)
    print("EXAMPLE USAGE:")
    print("="*80)
    
    # Take first test sample
    example_input = best_mapper.X_test[0:1]
    predicted = best_mapper.predict(example_input)
    actual = best_mapper.y_test[0]
    
    print(f"\nInput (set_targets): {example_input[0]}")
    print(f"Predicted (joint_pos): {predicted[0]}")
    print(f"Actual (joint_pos): {actual}")
    print(f"Absolute errors: {np.abs(predicted[0] - actual)}")
    print(f"Mean absolute error: {np.mean(np.abs(predicted[0] - actual)):.6f}")


if __name__ == "__main__":
    main()