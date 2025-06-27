import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, butter, filtfilt


def exponential_smoothing(data, alpha):
    """Apply exponential smoothing to 1D data.
    Args:
        data (list): Input data points.
        alpha (float): Smoothing factor (0 < alpha < 1).
    Returns:
        list: Smoothed data points.
    """
    smoothed = [data[0]]  # Initialize with first raw value
    for i in range(1, len(data)):
        smoothed_val = (1 - alpha) * smoothed[i-1] + alpha * data[i]
        smoothed.append(smoothed_val)
    return smoothed

# Load the data
file_path = "/home/user/ros2_ws/src/ros2_optitrack_listener/recorded_pos/data_10.csv"
df = pd.read_csv(file_path)

# Create timestamp column if not present
if 'timestamp' not in df.columns:
    # Generate timestamps based on 250Hz sampling
    df['timestamp'] = np.arange(len(df)) / 250

# Extract position data
x, y, z = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

# Apply exponential smoothing
window_size = 35  # Must be odd
poly_order = 3

smoothed_x = savgol_filter(x, window_size, poly_order)
smoothed_y = savgol_filter(y, window_size, poly_order)
smoothed_z = savgol_filter(z, window_size, poly_order)

# Create DataFrame with smoothed data and include timestamp
smoothed_df = pd.DataFrame({
    'x': smoothed_x,
    'y': smoothed_y,
    'z': smoothed_z,
    'timestamp': df['timestamp']  # Add timestamp column from original df
})

# Process coordinates for both original and smoothed data
processed_df = pd.DataFrame({
    'X': -df["z"],
    'Y': df["x"] - 0.5,
    'Z': df["y"],
    'timestamp': df['timestamp']
})

smoothed_processed_df = pd.DataFrame({
    'X': -smoothed_df["z"],
    'Y': smoothed_df["x"] - 0.5,
    'Z': smoothed_df["y"],
    'timestamp': smoothed_df['timestamp']
})

# Get original timestamps
t_original = processed_df['timestamp'].values
data_original = processed_df[['X', 'Y', 'Z']].values
smoothed_data_original = smoothed_processed_df[['X', 'Y', 'Z']].values

# Calculate the duration of the recording
duration = t_original[-1] - t_original[0]

# Create 1000Hz timestamps (starting from the first original timestamp)
t_1000hz = np.arange(t_original[0], t_original[-1], 1/1000)

# Create interpolation function for both original and smoothed data
interp_func_original = interp1d(t_original, data_original, kind='cubic', axis=0, fill_value='extrapolate')
interp_func_smoothed = interp1d(t_original, smoothed_data_original, kind='cubic', axis=0, fill_value='extrapolate')

# Get interpolated data at 1000Hz
data_1000hz_original = interp_func_original(t_1000hz)
data_1000hz_smoothed = interp_func_smoothed(t_1000hz)

# Select 200Hz data points
t_200hz = np.arange(t_original[0], t_original[-1], 1/200)
# Find the closest indices in 1000Hz data to 200Hz timestamps
indices = np.searchsorted(t_1000hz, t_200hz)
data_200hz_original = data_1000hz_original[indices]
data_200hz_smoothed = data_1000hz_smoothed[indices]

# Create DataFrame for 200Hz data
df_200hz_original = pd.DataFrame(data_200hz_original, columns=['X', 'Y', 'Z'])
df_200hz_original['timestamp'] = t_200hz

df_200hz_smoothed = pd.DataFrame(data_200hz_smoothed, columns=['X', 'Y', 'Z'])
df_200hz_smoothed['timestamp'] = t_200hz

# Calculate velocities for both original and smoothed data
# df_200hz_original['vx'] = np.gradient(df_200hz_original['X'].values, df_200hz_original['timestamp'].values)
# df_200hz_original['vy'] = np.gradient(df_200hz_original['Y'].values, df_200hz_original['timestamp'].values)
# df_200hz_original['vz'] = np.gradient(df_200hz_original['Z'].values, df_200hz_original['timestamp'].values)
# df_200hz_original['speed'] = np.sqrt(df_200hz_original['vx']**2 + df_200hz_original['vy']**2 + df_200hz_original['vz']**2)

# df_200hz_smoothed['vx'] = np.gradient(df_200hz_smoothed['X'].values, df_200hz_smoothed['timestamp'].values)
# df_200hz_smoothed['vy'] = np.gradient(df_200hz_smoothed['Y'].values, df_200hz_smoothed['timestamp'].values)
# df_200hz_smoothed['vz'] = np.gradient(df_200hz_smoothed['Z'].values, df_200hz_smoothed['timestamp'].values)
# df_200hz_smoothed['speed'] = np.sqrt(df_200hz_smoothed['vx']**2 + df_200hz_smoothed['vy']**2 + df_200hz_smoothed['vz']**2)

def calculate_smooth_velocity(pos, t, fc=5, fs=200):
    """Calculate smoothed velocity using digital filtering.
    
    Args:
        pos: Position array
        t: Timestamp array
        fc: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
    """
    # Compute raw velocity
    dt = np.gradient(t)
    print(dt)
    vel = np.gradient(pos) / dt
    
    # Design low-pass filter
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = butter(2, normal_cutoff, btype='low')
    
    # Apply zero-phase filtering
    smooth_vel = filtfilt(b, a, vel)
    
    return smooth_vel

# Apply to both datasets
for df in [df_200hz_original, df_200hz_smoothed]:
    t = df['timestamp'].values
    df['vx'] = calculate_smooth_velocity(df['X'].values, t)
    df['vy'] = calculate_smooth_velocity(df['Y'].values, t)
    df['vz'] = calculate_smooth_velocity(df['Z'].values, t)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

# Save processed data
df_200hz_original.to_csv("/home/user/ros2_ws/src/ros2_optitrack_listener/recorded_pos/processed_trajectory_200hz_original.csv", 
                index=False)
df_200hz_smoothed.to_csv("/home/user/ros2_ws/src/ros2_optitrack_listener/recorded_pos/processed_trajectory_200hz_smoothed.csv", 
                index=False)

# Convert to numpy arrays for plotting
x_original = df_200hz_original["X"].values
y_original = df_200hz_original["Y"].values
z_original = df_200hz_original["Z"].values
vx_original = df_200hz_original["vx"].values
vy_original = df_200hz_original["vy"].values
vz_original = df_200hz_original["vz"].values
speed_original = df_200hz_original["speed"].values

x_smoothed = df_200hz_smoothed["X"].values
y_smoothed = df_200hz_smoothed["Y"].values
z_smoothed = df_200hz_smoothed["Z"].values
vx_smoothed = df_200hz_smoothed["vx"].values
vy_smoothed = df_200hz_smoothed["vy"].values
vz_smoothed = df_200hz_smoothed["vz"].values
speed_smoothed = df_200hz_smoothed["speed"].values

timestamps = df_200hz_original["timestamp"].values - df_200hz_original["timestamp"].values[0]

# Create 3D plot
fig = plt.figure(figsize=(15, 10))

# Original Position plot
ax1 = fig.add_subplot(221, projection="3d")
ax1.plot(x_original, y_original, z_original, label="Original Trajectory")
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")
ax1.set_zlabel("Z Position")
ax1.set_title("Original 3D Trajectory")
ax1.legend()

# Smoothed Position plot
ax2 = fig.add_subplot(222, projection="3d")
ax2.plot(x_smoothed, y_smoothed, z_smoothed, label="Smoothed Trajectory", color='orange')
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")
ax2.set_zlabel("Z Position")
ax2.set_title("Smoothed 3D Trajectory")
ax2.legend()

# Original Velocity plot
ax3 = fig.add_subplot(223)
ax3.plot(timestamps, vz_original, label="Original Speed")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Speed (m/s)")
ax3.set_title("Original Speed Profile")
ax3.legend()

# Smoothed Velocity plot
ax4 = fig.add_subplot(224)
ax4.plot(timestamps, vz_smoothed, label="Smoothed Speed", color='orange')
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Speed (m/s)")
ax4.set_title("Smoothed Speed Profile")
ax4.legend()

plt.tight_layout()
plt.show()