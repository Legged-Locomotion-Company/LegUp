from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

import numpy as np
import os


class CustomLocalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, env, training_id, root_path, verbose=0):
        """_summary_

        Args:
            env (_type_): _description_
            training_id (_type_): _description_
            root_path (_type_): _description_
            verbose (int, optional): _description_. Defaults to 0.
        """

        self.env_ = env
        self.training_id = training_id

        self.model_save_path = f'{root_path}/checkpoints/{self.training_id}'

        super().__init__(verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # print("BEFORE FIRST ROLLOUT TRAINING START)")

        # os.makedirs(self.tb_dir_path, exist_ok=True)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print("BEFORE ROLLOUT (ROLLOUT START)")
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(
                f"rewards/{name}", infos['terms'][idx].item())

        self.model.save(os.path.join(
            self.model_save_path, str(self.num_timesteps)))

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class CustomWandbCallback(WandbCallback):
    def __init__(self, env, training_id, root_path, verbose=1):

        # TODO: This is for debug, get rid of when it works
        self.debug_count = 0

        self.env_ = env
        self.training_id = training_id

        self.model_save_path = f'{root_path}/checkpoints/{self.training_id}'
        self.video_buffer = []

        WandbCallback.__init__(
            self, model_save_path=self.model_save_path, verbose=verbose)

        Monitor(env)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # print("BEFORE FIRST ROLLOUT TRAINING START)")

        # os.makedirs(self.tb_dir_path, exist_ok=True)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print("BEFORE ROLLOUT (ROLLOUT START)")
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        TODO add a way to save the video every n steps

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        self.video_buffer.append(self.env_.render())

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        self.debug_count += 1

        give_debug_exception = False

        if self.debug_count > 4:
            if give_debug_exception:
                raise Exception("test_exception")

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(
                f"rewards/{name}", infos['terms'][idx].item())

        # For some reason the video needs to be transposed to frames, channels, height, width

        numpy_video = np.array(self.video_buffer).transpose([0, 3, 1, 2])

        wandb.log(
            {"video": wandb.Video(numpy_video, fps=20, format="gif")})

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(f"rewards/{name}", infos['terms'][idx].item())

        self.model.save(os.path.join(
            self.model_save_path, str(self.num_timesteps)))

        # , base_path='checkpoints')
        wandb.save(os.path.join(self.model_save_path, '*'))

        self.video_buffer = []

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
