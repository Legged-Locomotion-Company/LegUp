import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class Teacher(nn.Module):
    """Teacher network for imitation learning. Gets perfect information and is trained using PPO.
    """

    def __init__(self, cfg: dict):
        """Initialize the teacher network.

        Args:
            cfg (dict): configuration dictionary
        """
        super(Teacher, self).__init__()
        self.cfg = cfg
        # number of actions, 3 deltas and 1 phase shift per leg
        self.latent_dim_pi = 16
        self.latent_dim_vf = 1

        self.height_encoder = nn.Sequential(
            nn.Linear(52, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 60),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(60, 24),
        )

        self.priviligde_encoder = nn.Sequential(
            nn.Linear(50, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
        )
        # 96 + 24 + 133 = 253
        # 16 out, 12 joint, 4 phase
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

        # 253 * 256 + 256*128 + 128 = 100352
        self.critic = nn.Sequential(
            nn.Linear(253, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    # Heights: tensor of size (4,51) that contains the height samples for each leg, one vector per leg, 51 examples per leg
    # priv: data for all of the priviliged info, should be a 1d tensor of size 50
    # proprioception: data for all of the proprioception info, should be a 1d tensor of size 133

    def forward(self, obs):
        # proprio, extro, priv = obs # 133, 208, 50

        proprio = obs[:, :133]
        extro = obs[:, 133:341]
        priv = obs[:, 341:]

        encoded_data = self.forward_data(extro, priv, proprio)
        return self.forward_actor(encoded_data), self.forward_critic(encoded_data)

    def forward_data(self, heights, priv, proprioception):
        # divide vector up into 4 legs and pass through individually instead of all at once
        heights = heights.reshape(-1, 4, 52)
        heights = self.height_encoder(heights).flatten(start_dim=1)

        all_priv = self.priviligde_encoder(priv)
        # 96/4

        return torch.cat((heights, all_priv, proprioception), dim=1)

    def forward_actor(self, encoded_data):
        if encoded_data.shape[1] == 391:
            proprio = encoded_data[:, :133]
            extro = encoded_data[:, 133:341]
            priv = encoded_data[:, 341:]
            encoded_data = self.forward_data(extro, priv, proprio)

        return self.actor(encoded_data)

    def forward_critic(self, encoded_data):
        if encoded_data.shape[1] == 391:
            proprio = encoded_data[:, :133]
            extro = encoded_data[:, 133:341]
            priv = encoded_data[:, 341:]
            encoded_data = self.forward_data(extro, priv, proprio)

        return self.critic(encoded_data)


# example usage
# TODO: update model inputs for test

# model = Teacher(None)
# heights = torch.rand(3, 208)
# priv = torch.rand(3, 50)
# proprioception = torch.rand(3, 133)
# obs = torch.cat((heights, priv, proprioception), dim=1)
# # print(obs.shape)

# out = model(obs)
# print(out[0].shape)
# # output: torch.Size([16])

# print(sum(p.numel() for p in model.parameters()))
# output: 243,485


class CustomTeacherActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomTeacherActorCriticPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Teacher(None)
