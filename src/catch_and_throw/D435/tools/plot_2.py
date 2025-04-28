import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../tracking/recorded_data/ball_tracking_data_2_tracking.csv'
data = pd.read_csv(file_path)
# data = data.iloc[:-4]
file_path_predictions = '../tracking/recorded_data/ball_tracking_data_2_predictions.csv'
predictions_data = pd.read_csv(file_path_predictions)

# Extract the relevant columns for plotting
x_world = predictions_data['x_world'].to_numpy()
y_world = predictions_data['y_world'].to_numpy()
z_world = predictions_data['z_world'].to_numpy()

# Extract the time (`dt`) and calculate cumulative time for plotting
time = predictions_data['dt'].cumsum().to_numpy()

e_x = predictions_data['e_x'].to_numpy()

e_y = predictions_data['e_y'].to_numpy()

e_z = predictions_data['e_z'].to_numpy()

e_vx = predictions_data['e_vx'].to_numpy()

e_vy = predictions_data['e_vy'].to_numpy()

e_vz = predictions_data['e_vz'].to_numpy()

time_predictions = predictions_data['dt'].cumsum().to_numpy()



# Adjust the tracking data's time array to start from zero
adjusted_time = time - time[0]

# interpolated_x = np.interp(time_predictions, adjusted_time, x_world)
# print(interpolated_x.shape)

# interpolated_y = np.interp(time_predictions, adjusted_time, y_world)

# interpolated_z = np.interp(time_predictions, adjusted_time, z_world)



# Plot x_world, e_x, and interpolated data vs adjusted time

plt.figure(figsize=(8, 6))

plt.plot(adjusted_time, x_world, label='x_world', linewidth=2)

plt.plot(time_predictions, e_x, label='e_x (Error)', linestyle='--', linewidth=2)

# plt.plot(time_predictions, interpolated_x, label='Interpolated x', linestyle='-.', linewidth=2)

plt.xlabel('Time (s)')

plt.ylabel('X_world / e_x')

plt.title('X_world, e_x, and Interpolated x vs Time')

plt.grid(True)

plt.legend()

plt.show()



# Plot y_world, e_y, and interpolated data vs adjusted time

plt.figure(figsize=(8, 6))

plt.plot(adjusted_time, y_world, label='y_world', linewidth=2)

plt.plot(time_predictions, e_y, label='e_y (Error)', linestyle='--', linewidth=2)

# plt.plot(time_predictions, interpolated_y, label='Interpolated y', linestyle='-.', linewidth=2)

plt.xlabel('Time (s)')

plt.ylabel('Y_world / e_y')

plt.title('Y_world, e_y, and Interpolated y vs Time')

plt.grid(True)

plt.legend()

plt.show()



# Plot z_world, e_z, and interpolated data vs adjusted time

plt.figure(figsize=(8, 6))

plt.plot(adjusted_time, z_world, label='z_world', linewidth=2)

plt.plot(time_predictions, e_z, label='e_z (Error)', linestyle='--', linewidth=2)

# plt.plot(time_predictions, interpolated_z, label='Interpolated z', linestyle='-.', linewidth=2)

plt.xlabel('Time (s)')

plt.ylabel('Z_world / e_z')

plt.title('Z_world, e_z, and Interpolated z vs Time')

plt.grid(True)

plt.legend()

plt.show()

######################

plt.figure(figsize=(8, 6))
plt.plot(time_predictions, e_vx, label='e_vx (Error)', linestyle='--', linewidth=2)
plt.plot(time_predictions, e_vy, label='e_vy (Error)', linestyle='--', linewidth=2)
plt.plot(time_predictions, e_vz, label='e_vz (Error)', linestyle='--', linewidth=2)

# plt.plot(time_predictions, interpolated_z, label='Interpolated z', linestyle='-.', linewidth=2)

plt.xlabel('Time (s)')

plt.ylabel('e_vx / e_vy / e_vz')

plt.title('Z_world, e_z, and Interpolated z vs Time')

plt.grid(True)

plt.legend()

plt.show()

