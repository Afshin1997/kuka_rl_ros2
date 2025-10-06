import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def random_throw(
    init_state,
    final_state,
    g=9.81,
    x_range=0.3,
    y_range=0.25,
    z_range=0.2,
    t_min=0.8,
    t_max=1.0,
    device=None
):
    if device is None:
        device = init_state.device

    N = init_state.shape[0]
    
    # Generate random offsets for initial positions
    rand_x_init = (torch.rand(N, device=device) * 2 - 1) * x_range
    rand_y_init = (torch.rand(N, device=device) * 2 - 1) * y_range
    rand_z_init = (torch.rand(N, device=device) * 2 - 1) * z_range

    # Perturbed initial positions
    x_init = init_state[:, 0] + rand_x_init
    y_init = init_state[:, 1] + rand_y_init
    z_init = init_state[:, 2] + rand_z_init

    # Random flight times
    t = t_min + (t_max - t_min) * torch.rand(N, device=device)
    
    # For real-world scenarios: target velocities
    # vx: [-3, -2] m/s
    # vz: [1.3, 2.0] m/s
    target_vx = -3.0 + torch.rand(N, device=device) * 1.0  # [-3, -2]
    target_vz = 1.3 + torch.rand(N, device=device) * 0.7   # [1.3, 2.0]
    
    # Calculate final positions based on target velocities
    # x_final = x_init + vx * t
    x_final = x_init + target_vx * t
    
    # For y, keep similar to original but adjust for real-world
    y_final = final_state[:, 1] - torch.rand(N, device=device) * 0.4 + 0.1
    
    # z_final = z_init + vz * t - 0.5 * g * t^2
    z_final = z_init + target_vz * t - 0.5 * g * t * t

    # Compute velocities
    vx = target_vx  # Already set
    vy = (y_final - y_init) / t
    vz = target_vz  # Already set
    
    out = torch.stack([x_init, y_init, z_init, vx, vy, vz, t], dim=1)
    return out

def compute_trajectory(x0, y0, z0, vx, vy, vz, t_max, g=9.81, num_points=50):
    """Compute trajectory points given initial conditions"""
    t = np.linspace(0, t_max, num_points)
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t - 0.5 * g * t**2
    return x, y, z

def plot_ranges(init_pos, final_pos, x_range, y_range, z_range):
    """Plot the range values for positions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Initial position ranges
    ax1.bar(['X', 'Y', 'Z'], 
            [2*x_range, 2*y_range, 2*z_range],
            color=['red', 'green', 'blue'], alpha=0.7)
    ax1.set_title('Initial Position Ranges (meters)', fontsize=14)
    ax1.set_ylabel('Range (m)', fontsize=12)
    ax1.set_ylim(0, 0.7)
    for i, (label, value) in enumerate(zip(['X', 'Y', 'Z'], 
                                          [2*x_range, 2*y_range, 2*z_range])):
        ax1.text(i, value + 0.02, f'±{value/2:.2f}m', ha='center', fontsize=10)
    
    # Final position ranges
    final_ranges = [
        0.6,  # -0.6 to 0 for x
        0.4,  # -0.2 to +0.1 for y  
        0.4   # 0 to 0.4 for z
    ]
    ax2.bar(['X', 'Y', 'Z'], final_ranges,
            color=['red', 'green', 'blue'], alpha=0.7)
    ax2.set_title('Final Position Ranges (meters)', fontsize=14)
    ax2.set_ylabel('Range (m)', fontsize=12)
    ax2.set_ylim(0, 0.7)
    
    # Add specific range annotations
    ax2.text(0, final_ranges[0] + 0.02, '-0.6 to 0m', ha='center', fontsize=10)
    ax2.text(1, final_ranges[1] + 0.02, '-0.2 to +0.1m', ha='center', fontsize=10)
    ax2.text(2, final_ranges[2] + 0.02, '0 to 0.4m', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_velocity_distributions(states):
    """Plot velocity distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vx = states[:, 3].numpy()
    vy = states[:, 4].numpy()
    vz = states[:, 5].numpy()
    
    axes[0].hist(vx, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[0].set_title('X Velocity Distribution', fontsize=14)
    axes[0].set_xlabel('Velocity (m/s)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].axvline(vx.mean(), color='darkred', linestyle='--', 
                    label=f'Mean: {vx.mean():.2f} m/s')
    axes[0].legend()
    
    axes[1].hist(vy, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title('Y Velocity Distribution', fontsize=14)
    axes[1].set_xlabel('Velocity (m/s)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].axvline(vy.mean(), color='darkgreen', linestyle='--',
                    label=f'Mean: {vy.mean():.2f} m/s')
    axes[1].legend()
    
    axes[2].hist(vz, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[2].set_title('Z Velocity Distribution', fontsize=14)
    axes[2].set_xlabel('Velocity (m/s)', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].axvline(vz.mean(), color='darkblue', linestyle='--',
                    label=f'Mean: {vz.mean():.2f} m/s')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def plot_3d_trajectories(states, num_trajectories=1000, sample_size=50):
    """Plot 3D trajectories"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample trajectories to plot
    if num_trajectories > states.shape[0]:
        num_trajectories = states.shape[0]
    
    indices = np.random.choice(states.shape[0], num_trajectories, replace=False)
    
    # Create colormap based on initial velocity magnitude
    velocities = torch.sqrt(states[indices, 3]**2 + 
                           states[indices, 4]**2 + 
                           states[indices, 5]**2).numpy()
    colors = plt.cm.viridis(velocities / velocities.max())
    
    # Plot each trajectory
    for i, idx in enumerate(indices):
        x0, y0, z0 = states[idx, :3].numpy()
        vx, vy, vz = states[idx, 3:6].numpy()
        t_flight = states[idx, 6].item()
        
        x, y, z = compute_trajectory(x0, y0, z0, vx, vy, vz, t_flight)
        
        # Only plot if trajectory stays above ground
        # if np.all(z >= 0):
        ax.plot(x, y, z, color=colors[i], alpha=0.3, linewidth=0.5)
    
    # Plot initial and final regions
    # Initial region
    init_x = [3.0 - 0.3, 3.0 + 0.3]
    init_y = [-0.45 - 0.25, -0.45 + 0.25]
    init_z = [0.5 - 0.2, 0.5 + 0.2]
    
    # Draw initial box
    from itertools import product
    vertices = list(product(init_x, init_y, init_z))
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # x-direction
        (0, 2), (1, 3), (4, 6), (5, 7),  # y-direction
        (0, 4), (1, 5), (2, 6), (3, 7)   # z-direction
    ]
    for edge in edges:
        points = [vertices[edge[0]], vertices[edge[1]]]
        ax.plot3D(*zip(*points), 'r-', linewidth=2, alpha=0.5)
    
    # Final region (approximate based on the ranges)
    final_x = [0.51 - 0.6, 0.51]
    final_y = [-0.45 - 0.2, -0.45 + 0.1]
    final_z = [0.65, 0.65 + 0.4]
    
    vertices_final = list(product(final_x, final_y, final_z))
    for edge in edges:
        points = [vertices_final[edge[0]], vertices_final[edge[1]]]
        ax.plot3D(*zip(*points), 'b-', linewidth=2, alpha=0.5)
    
    # Add start and end points
    ax.scatter(3.0, -0.45, 0.5, color='red', s=100, marker='o', label='Initial Position')
    ax.scatter(0.51, -0.45, 0.65, color='blue', s=100, marker='*', label='Target Position')
    
    # Labels and formatting
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'{num_trajectories} Tennis Ball Trajectories', fontsize=16)
    ax.legend()
    
    # Set reasonable axis limits
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 0.5)
    ax.set_zlim(0, 2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                               norm=plt.Normalize(vmin=velocities.min(), 
                                                vmax=velocities.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Initial Velocity Magnitude (m/s)', fontsize=12)
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initial configuration from your code
    N = 1000  # Number of trajectories
    init_pos = torch.tensor([3.0, -0.45, 0.5], device=device).repeat(N, 1)
    final_pos = torch.tensor([0.51, -0.45, 0.65], device=device).repeat(N, 1)
    
    # Generate random throws
    states = random_throw(init_pos, final_pos, device=device)
    
    # Move to CPU for plotting
    states = states.cpu()
    
    # Plot range values
    print("Plotting position ranges...")
    plot_ranges(init_pos[0].cpu(), final_pos[0].cpu(), 
                x_range=0.3, y_range=0.25, z_range=0.2)
    
    # Plot velocity distributions
    print("Plotting velocity distributions...")
    plot_velocity_distributions(states)
    
    # Plot 3D trajectories
    print("Plotting 3D trajectories...")
    plot_3d_trajectories(states, num_trajectories=1000)
    
    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"Initial position range: X=[{states[:, 0].min():.3f}, {states[:, 0].max():.3f}], "
          f"Y=[{states[:, 1].min():.3f}, {states[:, 1].max():.3f}], "
          f"Z=[{states[:, 2].min():.3f}, {states[:, 2].max():.3f}]")
    print(f"Velocity range: Vx=[{states[:, 3].min():.3f}, {states[:, 3].max():.3f}], "
          f"Vy=[{states[:, 4].min():.3f}, {states[:, 4].max():.3f}], "
          f"Vz=[{states[:, 5].min():.3f}, {states[:, 5].max():.3f}]")
    print(f"Flight time range: [{states[:, 6].min():.3f}, {states[:, 6].max():.3f}] seconds")