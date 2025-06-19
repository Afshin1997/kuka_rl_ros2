import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Load the raw action data
df = pd.read_csv('tm_received_joint_target_np.csv')
print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
print(f"Column names: {df.columns.tolist()}")

# Class for real-time Savitzky-Golay filtering
class RealTimeSavitzkyGolay:
    def __init__(self, window_length=11, polyorder=3, deriv=0, delta=1.0, initial_values=None):
        """Initialize the real-time Savitzky-Golay filter
        
        Args:
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial used for fitting
            deriv: Order of derivative to compute (0 means smoothed function)
            delta: Sampling period (used for derivatives)
            initial_values: Initial values to fill the buffer (if None, zeros are used)
        """
        from scipy.signal import savgol_coeffs
        
        if window_length % 2 == 0:
            window_length += 1  # Ensure window length is odd
            
        if polyorder >= window_length:
            polyorder = window_length - 1
            print(f"Warning: polyorder too large, reduced to {polyorder}")
            
        self.window_length = window_length
        self.polyorder = polyorder
        
        # Precompute Savitzky-Golay coefficients
        self.coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
        
        # Initialize circular buffer
        if initial_values is not None:
            if len(initial_values) >= window_length:
                # Use the most recent values if provided more than needed
                self.buffer = np.array(initial_values[-window_length:], dtype=float)
            else:
                # Pad with the first value if not enough values provided
                pad_values = np.full(window_length - len(initial_values), 
                                    initial_values[0] if len(initial_values) > 0 else 0.0)
                self.buffer = np.concatenate([pad_values, np.array(initial_values, dtype=float)])
        else:
            self.buffer = np.zeros(window_length, dtype=float)
            
        # Index to track position in circular buffer
        self.current_idx = 0
        
    def __call__(self, x):
        """Process a new data point and return the filtered value"""
        # Update buffer with new value (circular buffer approach)
        self.buffer[self.current_idx] = x
        self.current_idx = (self.current_idx + 1) % self.window_length
        
        # Rearrange buffer to get correct temporal order for filtering
        ordered_buffer = np.concatenate([
            self.buffer[self.current_idx:],
            self.buffer[:self.current_idx]
        ])
        
        # Apply precomputed coefficients to get filtered value
        filtered_value = np.sum(ordered_buffer * self.coeffs)
        
        return filtered_value

# Parameters for the Savitzky-Golay filter
window_length = 9  # Must be odd and smaller than your data length (31 rows)
polyorder = 3      # Must be less than window_length

# Create output dataframe to store filtered values
filtered_df = pd.DataFrame(columns=df.columns)

# Apply both real-time and standard SG filters
print("Applying filters to each joint...")

# Create an array to store filtered data for each approach
num_rows, num_cols = df.shape
standard_sg_filtered = np.zeros((num_rows, num_cols))
realtime_sg_filtered = np.zeros((num_rows, num_cols))

# Apply standard Savitzky-Golay (batch processing)
for col_idx, col in enumerate(df.columns):
    standard_sg_filtered[:, col_idx] = savgol_filter(
        df[col].values, window_length=window_length, polyorder=polyorder
    )

# Apply real-time Savitzky-Golay
for col_idx, col in enumerate(df.columns):
    # Initialize with first few values
    rt_filter = RealTimeSavitzkyGolay(
        window_length=window_length, 
        polyorder=polyorder,
        initial_values=df[col].values[:3].repeat(3)  # Repeat first values to initialize
    )
    
    # Process each value one by one
    for row_idx, value in enumerate(df[col].values):
        realtime_sg_filtered[row_idx, col_idx] = rt_filter(value)

# Create DataFrames for the filtered data
standard_sg_df = pd.DataFrame(standard_sg_filtered, columns=df.columns)
realtime_sg_df = pd.DataFrame(realtime_sg_filtered, columns=df.columns)

# Save filtered data to CSV files
standard_sg_df.to_csv('tm_actions_standard_sg_filtered.csv', index=False)
realtime_sg_df.to_csv('tm_actions_realtime_sg_filtered.csv', index=False)

print("Saved filtered data to 'tm_actions_standard_sg_filtered.csv' and 'tm_actions_realtime_sg_filtered.csv'")

# Create comparison plots for visual inspection
plt.figure(figsize=(18, 15))

for i, col in enumerate(df.columns):
    plt.subplot(4, 2, i+1)
    
    # Plot raw data
    plt.plot(df[col].values, 'k-', label='Raw Action', linewidth=1.5)
    
    # Plot filtered data
    plt.plot(standard_sg_df[col].values, 'r-', label=f'Standard S-G (w={window_length}, p={polyorder})', linewidth=1.5)
    plt.plot(realtime_sg_df[col].values, 'g-', label=f'Real-time S-G (w={window_length}, p={polyorder})', linewidth=1.5)
    
    plt.title(f'Joint {i}: {col}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('sg_filter_comparison.png', dpi=300)
plt.show()

# Create a combined dataset with raw and filtered values for complete analysis
combined_df = pd.DataFrame()
for col in df.columns:
    combined_df[f"{col}_raw"] = df[col]
    combined_df[f"{col}_standard_sg"] = standard_sg_df[col]
    combined_df[f"{col}_realtime_sg"] = realtime_sg_df[col]

combined_df.to_csv('tm_actions_all_filters_comparison.csv', index=False)
print("Saved combined comparison data to 'tm_actions_all_filters_comparison.csv'")

# Demonstrate effect of different window lengths
plt.figure(figsize=(18, 15))
window_sizes = [5, 7, 11]  # Different window sizes to compare

for col_idx, col in enumerate(df.columns[:4]):  # Show only first 4 joints to keep plot readable
    plt.subplot(2, 2, col_idx+1)
    
    # Plot raw data
    plt.plot(df[col].values, 'k-', label='Raw', linewidth=1.5)
    
    # Plot different window sizes
    for window in window_sizes:
        if window < len(df) and window > polyorder:
            filtered = savgol_filter(df[col].values, window_length=window, polyorder=polyorder)
            plt.plot(filtered, label=f'S-G (w={window}, p={polyorder})')
    
    plt.title(f'Joint {col_idx}: Effect of Window Size')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('sg_filter_window_comparison.png', dpi=300)
plt.show()

print("Analysis complete with visualizations saved to 'sg_filter_comparison.png' and 'sg_filter_window_comparison.png'")
