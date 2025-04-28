import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory containing the prediction files
prediction_files_dir = "../tracking/recorded_data/"
prediction_files_pattern = os.path.join(prediction_files_dir, "*_predictions.csv")

# Get all prediction files matching the pattern
prediction_files = glob.glob(prediction_files_pattern)

# Load the simulation trajectory data
# simulation_data_path = "../tracking/recorded_data/ft_idealpd_2.csv"
# simulation_data = pd.read_csv(simulation_data_path)

# Ensure all data is converted to NumPy arrays before plotting
# sim_e_x = simulation_data['tennisball_pos_0'].to_numpy() * 1000
# sim_e_y = simulation_data['tennisball_pos_1'].to_numpy() * 1000
# sim_e_z = simulation_data['tennisball_pos_2'].to_numpy() * 1000

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot simulation trajectory
# ax.plot(sim_e_x, sim_e_y, sim_e_z, label='Simulation Trajectory', linewidth=2, linestyle='--')

# Loop through all prediction files and plot their trajectories
for file in prediction_files:
    # Load prediction data
    tracking_data = pd.read_csv(file)
    
    # Extract and convert prediction columns to NumPy arrays
    tracking_x = tracking_data['x_world'].to_numpy()
    tracking_y = tracking_data['y_world'].to_numpy()
    tracking_y[:] = -0.3 * 1000
    tracking_z = tracking_data['z_world'].to_numpy()

    tracking_e_x = tracking_data['e_x'].to_numpy()
    tracking_e_y = tracking_data['e_y'].to_numpy()
    tracking_e_y[:] = -0.3 * 1000
    tracking_e_z = tracking_data['e_z'].to_numpy()
    
    # Plot the predicted trajectory
    ax.plot(tracking_e_x, tracking_e_y, tracking_e_z, label=f'Predicted Trajectory ({os.path.basename(file)})', linestyle='--', linewidth=1)
    ax.plot(tracking_x, tracking_y, tracking_z, label=f'Trajectory ({os.path.basename(file)})', linestyle='-', linewidth=1)

# Setting titles and labels
ax.set_title("3D Comparison of Predicted and Simulated Ball Trajectories")
ax.set_xlabel("e_x (mm)")
ax.set_ylabel("e_y (mm)")
ax.set_zlabel("e_z (mm)")

# Add a legend
# ax.legend()

# Show plot
plt.show()



