import torch

from typing import Union, List, Tuple

class HistoryBuffer:
    """Buffer to store most recently updated data that is updated every few seconds"""
    def __init__(self, num_envs: int, dt: float, update_freq: int, history_size: int, data_size: int, device: torch.device, fill: torch.Tensor = None):
        """
        Args:
            num_envs (int): number of environments to store data for
            dt (float): `dt` that the simulator is using, this essentially is the time (seconds) between each successive `step` call
            update_freq (int): how often (in seconds) to update the data in the buffer
            history_size (int): how much data to store in the buffer
            data_size (int): size of the data we are storing -- data must be 1-dimensional
            device (torch.device): device (cpu/cuda) that data should be on
            fill (torch.Tensor): values to fill the buffer with, defaults to None
        """
        self.device = device
        self.dt = dt
        self.update_freq = update_freq
        self.history_size = history_size
        self.num_envs = num_envs

        self.elapsed_time = torch.zeros(num_envs).to(self.device)
        self.data = torch.zeros(num_envs, data_size, history_size).to(self.device)

        if fill is not None:
            for i in range(history_size):
                self.data[:, :, i] = fill

    
    def step(self, new_data: torch.Tensor):
        """Updates the buffer if enough time has elapsed (specified by `dt`) from the previous call

        Args:
            new_data (torch.Tensor): Most recent data to be added if time has passed
        """
        self.elapsed_time += self.dt

        update_idx = self.elapsed_time >= self.update_freq
        self.elapsed_time[update_idx] = 0

        self.data[update_idx, :, 1:] = self.data[update_idx, :, :-1]
        self.data[update_idx, :, 0] = new_data[update_idx, :] # 0 is the newest
    
    def get(self, idx: Union[int, List[int]]) -> torch.Tensor:
        """Gets the data at the specified index

        Args:
            idx (Union[int, List[int]]): index of data, can be list or int

        Returns:
            torch.Tensor: data at that index, shape `(num_envs, data_size, :)`
        """
        return self.data[:, :, idx]
    
    def flatten(self) -> torch.Tensor:
        """Gets all the data in the buffer, flattened

        Returns:
            torch.Tensor: flattened data in buffer, shape `(num_envs, data_size * history_size)`
        """
        return self.data.view(self.num_envs, -1)

    def reset(self, update_idx: Union[int, List[int]]):
        """Zeros out all data in the buffer

        Args:
            update_idx (Union[int, List[int]]): indices of data to zero out
        """
        self.elapsed_time[update_idx] = 0
        self.data[update_idx, :, :] = 0