import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):

    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def exponential_moving_average(data, alpha):

    ema = np.zeros_like(data)
    ema[0] = data[0]  # Initialize with first value
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema

def plot_joint_velocities_with_smoothing(csv_file_path, sampling_frequency=100, 
                                       ma_window=3, ema_alpha=0.3):

    df = pd.read_csv(csv_file_path)
    
    # Calculate time step
    dt = 1.0 / sampling_frequency
    print(f"Time step (dt): {dt} seconds")
    
    # Create time arrays
    time_velocities = np.arange(1, len(df)) * dt
    
    # Calculate velocities using numerical differentiation
    joint_columns = [col for col in df.columns if col.startswith('joint_')]
    velocities = {}
    velocities_ma = {}
    velocities_ema = {}
    
    for joint in joint_columns:
        # Calculate raw velocity
        velocity_raw = np.diff(df[joint]) / dt
        velocities[joint] = velocity_raw
        
        # Apply moving average
        velocity_ma = moving_average(velocity_raw, ma_window)
        velocities_ma[joint] = velocity_ma
        
        # Apply exponential moving average
        velocity_ema = exponential_moving_average(velocity_raw, ema_alpha)
        velocities_ema[joint] = velocity_ema
    
    
    # Set up colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(joint_columns)))
    
    # Plot 1: Comparison of raw vs smoothed velocities for all joints
    fig1, axes1 = plt.subplots(len(joint_columns), 1, figsize=(14, 2.5 * len(joint_columns)))
    if len(joint_columns) == 1:
        axes1 = [axes1]
    
    for i, joint in enumerate(joint_columns):
        ax = axes1[i]
        
        # Plot raw, MA, and EMA velocities
        ax.plot(time_velocities, velocities[joint], 
                color='lightgray', linewidth=1, alpha=0.7, label='Raw')
        ax.plot(time_velocities, velocities_ma[joint], 
                color=colors[i], linewidth=2, label=f'Moving Avg (w={ma_window})')
        ax.plot(time_velocities, velocities_ema[joint], 
                color='red', linewidth=2, linestyle='--', label=f'Exp Moving Avg (α={ema_alpha})')
        
        ax.set_title(f'{joint.replace("_", " ").title()} - Velocity Comparison', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Velocity (rad/s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Create DataFrames for easy access
    velocity_df_raw = pd.DataFrame(velocities)
    velocity_df_raw['time'] = time_velocities
    
    velocity_df_ma = pd.DataFrame(velocities_ma)
    velocity_df_ma['time'] = time_velocities
    
    velocity_df_ema = pd.DataFrame(velocities_ema)
    velocity_df_ema['time'] = time_velocities
    
    return velocity_df_raw, velocity_df_ma, velocity_df_ema

def main():
    """
    Main function to run the velocity plotting script with smoothing.
    """
    # Configuration
    CSV_FILE_PATH = 'idealpd/tm_received_joint_pos_np.csv'
    SAMPLING_FREQUENCY = 200  # Hz
    MA_WINDOW = 5  # Moving average window size
    EMA_ALPHA = 0.5  # Exponential moving average alpha
    
    
    # Plot velocities with smoothing
    vel_raw, vel_ma, vel_ema = plot_joint_velocities_with_smoothing(
        CSV_FILE_PATH, SAMPLING_FREQUENCY, MA_WINDOW, EMA_ALPHA)

if __name__ == "__main__":
    main()