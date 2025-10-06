import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('recorded_data.csv')

# Extract ball position columns
ball_pos_x = df['ball_pos_x'].values
ball_pos_y = df['ball_pos_y'].values
ball_pos_z = df['ball_pos_z'].values

# Extract recorded ball velocities
recorded_raw_vx = df['ball_raw_vx'].values
recorded_raw_vy = df['ball_raw_vy'].values
recorded_raw_vz = df['ball_raw_vz'].values

# Extract recorded ball velocities
recorded_ema_vx = df['ball_ema_vx'].values
recorded_ema_vy = df['ball_ema_vy'].values
recorded_ema_vz = df['ball_ema_vz'].values

# Calculate time steps (assuming constant time step)
# Since we don't have time data, we'll assume a constant time step of 1 unit
# You may need to adjust this based on your actual sampling rate
dt = 0.01  # Time step in seconds (adjust if needed)

# Calculate velocities manually using finite differences
calculated_vx = np.zeros_like(ball_pos_x)
calculated_vy = np.zeros_like(ball_pos_y)
calculated_vz = np.zeros_like(ball_pos_z)

# Forward difference for the first point
calculated_vx[0] = (ball_pos_x[1] - ball_pos_x[0]) / dt
calculated_vy[0] = (ball_pos_y[1] - ball_pos_y[0]) / dt
calculated_vz[0] = (ball_pos_z[1] - ball_pos_z[0]) / dt

# Central difference for interior points
for i in range(1, len(ball_pos_x) - 1):
    calculated_vx[i] = (ball_pos_x[i+1] - ball_pos_x[i-1]) / (2 * dt)
    calculated_vy[i] = (ball_pos_y[i+1] - ball_pos_y[i-1]) / (2 * dt)
    calculated_vz[i] = (ball_pos_z[i+1] - ball_pos_z[i-1]) / (2 * dt)

# Backward difference for the last point
calculated_vx[-1] = (ball_pos_x[-1] - ball_pos_x[-2]) / dt
calculated_vy[-1] = (ball_pos_y[-1] - ball_pos_y[-2]) / dt
calculated_vz[-1] = (ball_pos_z[-1] - ball_pos_z[-2]) / dt

# Create time array for plotting
time = np.arange(len(ball_pos_x)) * dt

# Create subplots for each velocity component
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot X velocity component
axes[0].plot(time, recorded_raw_vx, 'b-', label='Recorded raw Vx', linewidth=2)
axes[0].plot(time, calculated_vx, 'r--', label='Calculated Vx', linewidth=2)
axes[0].plot(time, recorded_ema_vx, 'g-', label='Recorded ema Vx', linewidth=2)
axes[0].set_ylabel('Velocity X (m/s)')
axes[0].set_title('Ball Velocity Comparison - X Component')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Y velocity component
axes[1].plot(time, recorded_raw_vy, 'b-', label='Recorded raw Vy', linewidth=2)
axes[1].plot(time, calculated_vy, 'r--', label='Calculated Vy', linewidth=2)
axes[1].plot(time, recorded_ema_vy, 'g-', label='Recorded ema Vy', linewidth=2)

axes[1].set_ylabel('Velocity Y (m/s)')
axes[1].set_title('Ball Velocity Comparison - Y Component')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot Z velocity component
axes[2].plot(time, recorded_raw_vz, 'b-', label='Recorded raw Vz', linewidth=2)
axes[2].plot(time, calculated_vz, 'r--', label='Calculated Vz', linewidth=2)
axes[2].plot(time, recorded_ema_vz, 'g-', label='Recorded ema Vz', linewidth=2)

axes[2].set_ylabel('Velocity Z (m/s)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Ball Velocity Comparison - Z Component')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate and print some statistics
vx_diff = recorded_raw_vx - calculated_vx
vy_diff = recorded_raw_vy - calculated_vy
vz_diff = recorded_raw_vz - calculated_vz

print("Velocity Comparison Statistics:")
print(f"X Component - Mean Absolute Error: {np.mean(np.abs(vx_diff)):.4f} m/s")
print(f"Y Component - Mean Absolute Error: {np.mean(np.abs(vy_diff)):.4f} m/s")
print(f"Z Component - Mean Absolute Error: {np.mean(np.abs(vz_diff)):.4f} m/s")
print(f"X Component - Max Absolute Error: {np.max(np.abs(vx_diff)):.4f} m/s")
print(f"Y Component - Max Absolute Error: {np.max(np.abs(vy_diff)):.4f} m/s")
print(f"Z Component - Max Absolute Error: {np.max(np.abs(vz_diff)):.4f} m/s")

