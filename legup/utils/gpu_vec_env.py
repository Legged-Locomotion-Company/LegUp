from legup.train.agents.anymal import AnymalAgent

import torch

# Wrapper for ConcurrentTrainingEnv to convert returned torch tensors to numpy and input numpy arrays to torch tensors
# TODO: generalize it to not just the `ConcurrentTrainingEnv` environment
class GPUVecEnv(AnymalAgent):
    def step(self, actions):
        actions = torch.from_numpy(actions).cuda()

        new_obs, reward, dones, infos = super().step(actions)

        new_obs = new_obs.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        dones = dones.cpu().detach().numpy()

        return new_obs, reward, dones, infos

    def reset(self):
        obs = super().reset()
        return obs.cpu().detach().numpy()