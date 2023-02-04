import cv2
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback


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

        cv2.imshow('training', self.env_.render())
        cv2.waitKey(1)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        infos = self.locals['infos'][0]
        for reward_name, reward_val in zip(infos['reward_names'], infos['reward_terms']):
            self.logger.record(
                f"rewards/{reward_name}", reward_val.item())

        for log_name, log_val in infos['logs'].items():
            self.logger.record(
                f"logs/{log_name}", log_val)

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

        self.rollout_count = 0
        self.video_save_freq = 10
        self.save_video_this_rollout = False

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

        self.save_video_this_rollout = False

        if self.rollout_count % self.video_save_freq == 0:
            self.save_video_this_rollout = True

        self.video_buffer = []
        self.rollout_count += 1

        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        TODO add a way to save the video every n steps

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if self.save_video_this_rollout:
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
        for reward_name, reward_val in zip(infos['reward_names'], infos['reward_terms']):
            self.logger.record(
                f"rewards/{reward_name}", reward_val.item())

        for log_name, log_val in infos['logs'].items():
            self.logger.record(
                f"logs/{log_name}", log_val)

        # For some reason the video needs to be transposed to frames, channels, height, width

        numpy_video = np.array(self.video_buffer).transpose([0, 3, 1, 2])

        if self.save_video_this_rollout:
            wandb.log(
                {"video": wandb.Video(numpy_video, fps=20, format="gif")})

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
