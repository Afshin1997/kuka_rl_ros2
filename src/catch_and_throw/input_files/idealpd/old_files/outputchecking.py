import torch
import pandas as pd
import numpy as np

# Load observation data from CSV
df = pd.read_csv("ft_idealpd.csv")
obs_data = torch.tensor(df.values, dtype=torch.float32)

# Load exported JIT model
policy = torch.jit.load("policy.pt")
policy.eval()

# Extract observations (all columns except actions)
obs_data = torch.tensor(df.values, dtype=torch.float32)  # shape [N, 67]

with torch.no_grad():
    simulated_actions = []
    for i in range(len(obs_data)):
        obs = obs_data[i].unsqueeze(0)  # Keep all 67 features
        action = policy(obs)  # Policy handles splitting internally
        simulated_actions.append(action.squeeze().numpy())

# Get recorded actions (offset by 1 timestep)
recorded_actions = df[["pure_actions_0", "pure_actions_1", "pure_actions_2", 
                      "pure_actions_3", "pure_actions_4", "pure_actions_5",
                      "pure_actions_6"]].values[1:]  # skip first row

# Compare
for i, (simulated, recorded) in enumerate(zip(simulated_actions, recorded_actions)):
    print(f"Step {i}:")
    print("Simulated:", simulated.round(4))
    print("Recorded: ", recorded.round(4))
    print("Match?:   ", np.allclose(simulated, recorded, atol=3e-3))  # tolerance for floating point
    print()