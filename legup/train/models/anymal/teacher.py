from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch import nn
from typing import Tuple


class Teacher(nn.Module):
    """Teacher network for imitation learning. Gets perfect information and is trained using PPO."""

    def __init__(self, cfg: dict):
        """Initialize the teacher network.

        Args:
            cfg (dict): configuration dictionary for the teacher network
        """
        super(Teacher, self).__init__()
        self.cfg = cfg
        # number of actions, 3 deltas and 1 phase shift per leg
        self.latent_dim_pi = 16
        self.latent_dim_vf = 1

        # we take 52 height samples from around each foot, and then pass each foot's values through the height encoder
        self.height_encoder = nn.Sequential(
            nn.Linear(52, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 60),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(60, 24),
        )

        self.privilege_encoder = nn.Sequential(
            nn.Linear(50, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
        )
        # Length of the encoding outputs 96 + 24 + 133 = 253
        # takes in all of the encoded data and outputs desired action deltas
        self.actor = nn.Sequential(
            nn.Linear(253, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 160),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.latent_dim_pi)
        )

        # takes in all of the encoded data and outputs a value estimate
        # num params critic = 253 * 256 + 256*128 + 128 = 100352
        self.critic = nn.Sequential(
            nn.Linear(253, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    # Heights: tensor of size (4,52) that contains the height samples for each leg, one vector per leg, 51 examples per leg
    # priv: data for all of the priviliged info, should be a 1d tensor of size 50
    # proprioception: data for all of the proprioception info, should be a 1d tensor of size 133

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes in an observation vector and outputs actor and critic outputs.

        Args:
            obs (torch.Tensor): Observation vector of shape (num_envs, 391). 133 values for proprioception, 208 for height maps around each foot (52 per foot),
            and 50 for the privileged information. Proprioceptive data is not passed through an encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Outputs of the actor and critic networks of shape (num_envs, 16) and (num_envs, 1) respectively.
        """

        encoded_data = self.split_and_encode_data(obs)
        return self.forward_actor(encoded_data), self.forward_critic(encoded_data)

    def split_and_encode_data(self, obs: torch.Tensor) -> torch.Tensor:
        """Splits the observation vector into its 3 parts and encodes them.

        Args:
            obs (torch.Tensor): Observation vector of shape (num_envs, 391). 133 values for proprioception, 208 for height maps around each foot (52 per foot),
            and 50 for the privileged information.

        Returns:
            torch.Tensor: Encoded observation vector of shape (num_envs, 253).
        """
        proprio = obs[:, :133]
        extro = obs[:, 133:341]
        priv = obs[:, 341:]

        return self.forward_data(extro, priv, proprio)

    def forward_data(self, heights: torch.Tensor, priv: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        """Encodes the data.

        Args:
            heights (torch.Tensor): Heights of the terrain around each foot. Shape (num_envs, 208).
            priv (torch.Tensor): Privileged information of contact states and forces. Shape (num_envs, 50).
            proprioception (torch.Tensor): Information the robot is able to sense about itself. Shape (num_envs, 133).

        Returns:
            torch.Tensor: Concatenated encoded observations of shape (num_envs, 253)
        """
        # divide vector up into 4 legs and pass through individually instead of all at once
        heights = heights.reshape(-1, 4, 52)
        heights = self.height_encoder(heights).flatten(start_dim=1)

        all_priv = self.privilege_encoder(priv)

        return torch.cat((heights, all_priv, proprioception), dim=1)

    def forward_actor(self, encoded_data: torch.Tensor) -> torch.Tensor:
        """Pass encoded data through the actor network.

        Args:
            encoded_data (torch.Tensor): Encoded observation vector of shape (num_envs, 253).

        Returns:
            torch.Tensor: Output of the actor network of shape (num_envs, 16).
        """

        # handle case where data is not encoded due to function being called directly
        # we should figure out where this bug comes from
        if encoded_data.shape[1] == 391:
            encoded_data = self.split_and_encode_data(encoded_data)

        return self.actor(encoded_data)

    def forward_critic(self, encoded_data):
        """Pass encoded data through the critic network.

        Args:
            encoded_data (torch.Tensor): Encoded observation vector of shape (num_envs, 253).

        Returns:
            torch.Tensor: Output of the actor network of shape (num_envs, 1).
        """

        # handle case where data is not encoded due to function being called directly
        # we should figure out where this bug comes from
        if encoded_data.shape[1] == 391:
            encoded_data = self.split_and_encode_data(encoded_data)

        return self.critic(encoded_data)


class CustomTeacherActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, cfg, *args, **kwargs):
        """Custom policy that uses the Teacher network as the feature extractor."""
        super().__init__(cfg, *args, **kwargs)
        
        self.cfg = cfg

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Teacher(self.cfg)
