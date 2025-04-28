# from __future__ import annotations
# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# import torch.nn.functional as F
# import math

# def get_activation(act_name):
#     if act_name == "elu":
#         return nn.ELU()
#     elif act_name == "selu":
#         return nn.SELU()
#     elif act_name == "relu":
#         return nn.ReLU()
#     elif act_name == "crelu":
#         return nn.CReLU()
#     elif act_name == "lrelu":
#         return nn.LeakyReLU()
#     elif act_name == "tanh":
#         return nn.Tanh()
#     elif act_name == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         raise ValueError(f"Invalid activation function: {act_name}")

# class ActorCriticAfshin(nn.Module):
#     is_recurrent = False

#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         actor_hidden_dims=[128, 64, 64],
#         critic_hidden_dims=[128, 64, 64],
#         activation="elu",
#         init_noise_std=1.0,
#         num_categories=7,
#         embedding_dim=5,
#         **kwargs,
#     ):
#         if kwargs:
#             print(
#                 "ActorCriticAfshin.__init__ got unexpected arguments, which will be ignored: "
#                 + str([key for key in kwargs.keys()])
#             )
#         super().__init__()

#         # Setup embedding for categorical inputs
#         self.num_categories = num_categories
#         self.embedding_dim = embedding_dim
#         self.embedding = nn.Embedding(num_categories, embedding_dim)
#         nn.init.uniform_(self.embedding.weight, -0.05, 0.05)  # Smaller range than original

#         # Adjust input dimensions to account for embedding
#         adjusted_num_actor_obs = num_actor_obs - num_categories + embedding_dim
#         # adjusted_num_critic_obs = num_critic_obs - num_categories + embedding_dim
#         adjusted_num_critic_obs = num_critic_obs - num_categories + embedding_dim


#         activation_fn = get_activation(activation)

#         # Actor network
#         actor_layers = []
#         actor_layers.append(nn.Linear(adjusted_num_actor_obs, actor_hidden_dims[0]))
#         # actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
#         actor_layers.append(activation_fn)
        
#         # actor_layers.append(nn.Dropout(p=0.1))

#         for i in range(len(actor_hidden_dims) - 1):
#             actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
#             actor_layers.append(nn.LayerNorm(actor_hidden_dims[i + 1]))
#             actor_layers.append(activation_fn)
            
#             # actor_layers.append(nn.Dropout(p=0.1))

#         actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
#         actor_layers.append(nn.LayerNorm(num_actions))  # Add LayerNorm before activation
#         # actor_layers.append(nn.Softsign())
#         self.actor = nn.Sequential(*actor_layers)

#         # Critic network
#         critic_layers = []
#         critic_layers.append(nn.Linear(adjusted_num_critic_obs, critic_hidden_dims[0]))
#         # critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
#         critic_layers.append(activation_fn)
        
#         # critic_layers.append(nn.Dropout(p=0.1))

#         for i in range(len(critic_hidden_dims) - 1):
#             critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
#             critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
#             critic_layers.append(activation_fn)
            
#             # critic_layers.append(nn.Dropout(p=0.1))

#         critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
#         self.critic = nn.Sequential(*critic_layers)

#         def _init_weights(m):
#             """Universal weight initialization handler"""
#             if isinstance(m, nn.Linear):
#                 if hasattr(self, 'actor') and len(list(self.actor.children())) >= 3 and m is list(self.actor.children())[-3]:
#                     nn.init.xavier_normal_(m.weight, gain=0.1)
#                 elif hasattr(self, 'critic') and m is list(self.critic.children())[-1]:
#                     nn.init.xavier_uniform_(m.weight, gain=0.01)
#                 else:
#                     nn.init.xavier_normal_(m.weight, gain=1.0)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.LayerNorm):
#                 if hasattr(self, 'actor') and len(list(self.actor.children())) >= 2 and m is list(self.actor.children())[-2]:
#                     nn.init.constant_(m.weight, 0.5)
#                 else:
#                     nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#         # Apply initialization after network creation
#         self.apply(_init_weights)
        
#         # Action noise
#         # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         self.log_std = nn.Parameter(
#             torch.full(
#                 (num_actions,), 
#                 math.log(init_noise_std),  # Initialize with log(init_noise_std)
#                 device='cuda'  # Adjust device as needed
#             )
#         )
#         self.std = torch.exp(self.log_std)

#         self.min_log_std = math.log(0.1)  # Minimum std of 0.1
#         self.max_log_std = math.log(1.0)  # Maximum std of 1.0
#         self.distribution = None


#         print(f"Actor MLP: {self.actor}")
#         print(f"Critic MLP: {self.critic}")

#         # disable args validation for speedup
#         Normal.set_default_validate_args = False

#     def reset(self, dones=None):
#         pass

#     def forward(self):
#         raise NotImplementedError

#     @property
#     def action_mean(self):
#         return self.distribution.mean

#     @property
#     def action_std(self):
#         return self.distribution.stddev
#         # return torch.exp(self.log_std)

#     @property
#     def entropy(self):
#         return self.distribution.entropy().sum(dim=-1)

#     def _process_observations(self, observations: torch.Tensor):
#         """
#         Process observations by replacing the last num_categories features (one-hot)
#         with an embedding vector.
#         """
#         batch_size = observations.shape[0]
#         # Split continuous and categorical parts
#         categorical_obs = torch.argmax(observations[:, -self.num_categories:], dim=1)  # [batch]
#         continuous_obs = observations[:, :-self.num_categories]  # [batch, original_dim - num_categories]

#         embedded = self.embedding(categorical_obs)  # [batch, embedding_dim]

#         # Concatenate continuous with embedded
#         combined = torch.cat([continuous_obs, embedded], dim=-1)  # [batch, adjusted_dim]
#         # print("combined", combined[0, -6:])
#         return combined

#     def update_distribution(self, observations):
#         # category = torch.argmax(observations[:, -self.num_categories:], dim=1)
#         combined = self._process_observations(observations)
#         mean = self.actor(combined)
#         # std = torch.exp(self.log_std)
#         # self.std = torch.exp(self.log_std)
#         self.log_std.data = torch.clamp(self.log_std.data, 
#                                min=self.min_log_std,
#                                max=self.max_log_std)
#         self.std = torch.exp(self.log_std)
#         self.distribution = Normal(mean, mean * 0.0 + self.std)

#     def act(self, observations, **kwargs):
#         self.update_distribution(observations)
#         return self.distribution.sample()

#     def get_actions_log_prob(self, actions):
#         return self.distribution.log_prob(actions).sum(dim=-1)

#     def act_inference(self, observations):
#         combined = self._process_observations(observations)
#         actions_mean = self.actor(combined)
#         return actions_mean

#     def evaluate(self, critic_observations, **kwargs):
#         # Clone to convert inference tensor to a regular tensor
#         # critic_observations = critic_observations.clone()
#         # print("critic_observations", critic_observations[0, -6:])
#         combined = self._process_observations(critic_observations)
#         value = self.critic(combined)
        
#         return value

from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import math

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
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        num_categories=7,
        embedding_dim=5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticAfshin.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Setup embedding for categorical inputs
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)

        # Adjust input dimensions to account for embedding
        adjusted_num_actor_obs = num_actor_obs - num_categories + embedding_dim
        # adjusted_num_critic_obs = num_critic_obs - num_categories + embedding_dim
        adjusted_num_critic_obs = num_critic_obs - num_categories + embedding_dim


        activation_fn = get_activation(activation)

        # Actor network
        actor_layers = []
        actor_layers.append(nn.Linear(adjusted_num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        # actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
        # actor_layers.append(nn.Dropout(p=0.1))

        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(nn.LayerNorm(actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
            # actor_layers.append(nn.LayerNorm(actor_hidden_dims[i + 1]))
            # actor_layers.append(nn.Dropout(p=0.1))

        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        actor_layers.append(nn.Softsign())
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        critic_layers.append(nn.Linear(adjusted_num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        # critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
        # critic_layers.append(nn.Dropout(p=0.1))

        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
            # critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
            # critic_layers.append(nn.Dropout(p=0.1))

        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # self.log_std = nn.Parameter(
        #     torch.full(
        #         (num_actions,), 
        #         math.log(init_noise_std),  # Initialize with log(init_noise_std)
        #         device='cuda'  # Adjust device as needed
        #     )
        # )
        # self.std = torch.exp(self.log_std)
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
        # return torch.exp(self.log_std)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _process_observations(self, observations: torch.Tensor):
        """
        Process observations by replacing the last num_categories features (one-hot)
        with an embedding vector.
        """
        batch_size = observations.shape[0]
        # Split continuous and categorical parts
        categorical_obs = torch.argmax(observations[:, -self.num_categories:], dim=1)  # [batch]
        continuous_obs = observations[:, :-self.num_categories]  # [batch, original_dim - num_categories]

        embedded = self.embedding(categorical_obs)  # [batch, embedding_dim]

        # Concatenate continuous with embedded
        combined = torch.cat([continuous_obs, embedded], dim=-1)  # [batch, adjusted_dim]
        # print("combined", combined[0, -6:])
        return combined

    def update_distribution(self, observations):
        # category = torch.argmax(observations[:, -self.num_categories:], dim=1)
        combined = self._process_observations(observations)
        mean = self.actor(combined)
        # std = torch.exp(self.log_std)
        # self.std = torch.exp(self.log_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        combined = self._process_observations(observations)
        actions_mean = self.actor(combined)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # Clone to convert inference tensor to a regular tensor
        # critic_observations = critic_observations.clone()
        # print("critic_observations", critic_observations[0, -6:])
        combined = self._process_observations(critic_observations)
        value = self.critic(combined)
        
        return value