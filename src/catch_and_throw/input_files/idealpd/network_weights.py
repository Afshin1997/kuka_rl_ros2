import torch
import numpy as np

import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from torch.distributions import Normal

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {act_name}")

class ActorCriticAfshin(nn.Module):
    def __init__(
        self,
        num_actor_obs=67,       
        num_critic_obs=116,    
        num_actions=7,
        actor_hidden_dims=[256, 128, 128, 32],  # Exactly these sizes
        critic_hidden_dims=[256, 128, 128, 64], # Exactly these sizes
        num_categories=7,
        embedding_dim=5
    ):
        super().__init__()
        self.num_categories = num_categories
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Actor Network (EXACT original structure)
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs - num_categories + embedding_dim, 256), # layer 0
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 7),
            nn.Softsign()
        )
        
        # Critic Network (EXACT original structure)
        self.critic = nn.Sequential(
            nn.Linear(114, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
        self.std = nn.Parameter(torch.ones(num_actions))

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        

    def _process_observations(self, observations: torch.Tensor):
        """
        Process observations by replacing the last num_categories features (one-hot)
        with an embedding vector.
        """
        batch_size = observations.shape[0]
        categorical_obs = torch.argmax(observations[:, -self.num_categories:], dim=1)  # [batch]
        continuous_obs = observations[:, :-self.num_categories]  # [batch, original_dim - num_categories]

        embedded = self.embedding(categorical_obs)  # [batch, embedding_dim]

        combined = torch.cat([continuous_obs, embedded], dim=-1)  # [batch, adjusted_dim]
        print("combined", combined)
        return combined



    def act_inference(self, observations):
        combined = self._process_observations(observations)
        actions_mean = self.actor(combined)
        return actions_mean


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""
    def __init__(self, shape=67, eps=1e-8, task_dim=7):
        super().__init__()
        self.eps = eps
        self.task_dim = task_dim
        self.main_dim = shape - task_dim
        
        # Register buffers properly
        self.register_buffer("_mean", torch.zeros(self.main_dim).unsqueeze(0))
        self.register_buffer("_var", torch.ones(self.main_dim).unsqueeze(0))
        self.register_buffer("_std", torch.ones(self.main_dim).unsqueeze(0))

    def forward(self, x):
        x_main = x[:, :-self.task_dim]
        x_task = x[:, -self.task_dim:]
        
        # Use registered buffers
        x_main_norm = (x_main - self._mean) / (self._std + self.eps)
        return torch.cat([x_main_norm, x_task], dim=1)

def verify_with_csv(checkpoint_path, csv_path, num_categories=7, embedding_dim=5):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")
    print("Total columns:", len(df.columns))
    # Extract observations and actions
    # Assuming the first 67 columns are observations (adjust if needed)
    obs_columns = [col for col in df.columns]
    pure_action_columns = [col for col in df.columns if col.startswith('pure_actions_')]
    
    # Verify we have the right number of observations
    if len(obs_columns) != 67:
        print(f"Warning: Expected 67 observation columns, found {len(obs_columns)}")
    

    # Initialize model and normalizer
    model = ActorCriticAfshin(
        num_actor_obs=67,
        num_critic_obs=116,
        num_actions=7,
        num_categories=num_categories,
        embedding_dim=embedding_dim
    )
    print("Checkpoint keys:", checkpoint['model_state_dict'].keys())  
    print("Embedding weights BEFORE loading:", model.embedding.weight[:,:])  # First 3 elements
    print("std before loading model", model.std[0])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    print("Embedding weights AFTER loading:", model.embedding.weight[:,:])  # Should change!
    print("std afster loading model", model.std)
    obs_normalizer = EmpiricalNormalization(
        shape=67,
        task_dim=7  # Matches num_categories used in training
    )
    print("init mean buffer:", obs_normalizer._mean[0,:5])

    norm_state = checkpoint['obs_norm_state_dict']
    renamed_norm_state = {
        '_mean': norm_state['_mean_cont'],
        '_var': norm_state['_var_cont'],
        '_std': norm_state['_std_cont']
    }
    obs_normalizer.load_state_dict(renamed_norm_state)
    obs_normalizer.eval()
    # print("loaded mean buffer:", obs_normalizer._mean[0,:])
    # print("CSV feature means:", df.iloc[:, :].mean().values)
    
    # Process each row and compare outputs
    num_samples = min(5, len(df))  # Compare first 100 samples for efficiency
    mse_errors = []
    
    print("\nComparing network outputs with recorded actions:")
    print("="*70)
    print("Index  |  Max Abs Diff  |  MSE       |  Match?")
    print("-"*70)
    
    for i in range(num_samples):
        # Get observation and ground truth action
        obs = df[obs_columns].iloc[i].values.astype(np.float32)
        true_action = df[pure_action_columns].iloc[i+1].values.astype(np.float32)
        
        # Process through model
        with torch.no_grad():
            # Normalize observation
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # print("obs_tensor",obs_tensor)
            normalized_obs = obs_normalizer(obs_tensor)
            # print("normalized_obs", normalized_obs)
            # Get model prediction
            pred_action = model.act_inference(normalized_obs).numpy()[0]
            # print(pred_action)
        
        # Calculate differences
        print("pred_action",pred_action)
        print("true_action", true_action)
        abs_diff = np.abs(pred_action - true_action)
        max_diff = np.max(abs_diff)
        mse = np.mean((pred_action - true_action)**2)
        mse_errors.append(mse)
        
        # Determine if they match (within some tolerance)
        tolerance = 1e-1
        matches = "✅" if max_diff < tolerance else "❌"
        
        print(f"{i:6}|  {max_diff:.6f}    |  {mse:.6f}  |  {matches}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average MSE across samples: {np.mean(mse_errors):.6f}")
    print(f"Max MSE across samples: {np.max(mse_errors):.6f}")
    print(f"Percentage matching (within tolerance): {100 * sum(1 for x in mse_errors if x < tolerance)/num_samples:.1f}%")

if __name__ == "__main__":
    checkpoint_path = "ft_idealpd.pt"  # Update with your actual path
    csv_path = "env_3_data.csv"     # Your CSV file
    
    verify_with_csv(checkpoint_path, csv_path)