import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_function(dir1, dir2, col1, col2):
    plt.figure(figsize=(18, 12))
    
    # Read the data
    df_camera = pd.read_csv(dir1, usecols=col1)
    df_ekf = pd.read_csv(dir2, usecols=col2)
    
    # Get the number of samples
    len_camera = len(df_camera)
    len_ekf = len(df_ekf)
    
    # Generate time arrays assuming equal duration from 0 to total_duration
    total_duration = 1  # Replace with actual duration if known
    t_camera = np.linspace(0, total_duration, len_camera)
    t_ekf = np.linspace(0, total_duration, len_ekf)
    
    for idx, (col_cam, col_ekf) in enumerate(zip(col1, col2)):
        plt.subplot(3, 1, idx+1)
        plt.plot(t_camera, np.array(df_camera[col_cam]), label=f'{col_cam}_camera')
        plt.plot(t_ekf, np.array(df_ekf[col_ekf]), label=f"{col_ekf}_EKF")
        plt.legend()
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel(col_cam)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    dir1 = '/home/afshin/kuka_repo_ros/src/catch_and_throw/D435/tracking/recorded_data/ball_tracking_data_3_tracking.csv'
    dir2 = '/home/afshin/kuka_repo_ros/src/catch_and_throw/D435/tracking/recorded_data/ball_tracking_data_3_predictions.csv'
    col_1_pos = ['x', 'y', 'z']
    col_2_ekf_pos = ['e_x', 'e_y', 'e_z']
    col_1_vel = ['vx', 'vy', 'vz']
    col_2_ekf_vel = ['e_vx', 'e_vy', 'e_vz']

    plot_function(dir1, dir2, col_1_pos, col_2_ekf_pos)
    plot_function(dir1, dir2, col_1_vel, col_2_ekf_vel)
