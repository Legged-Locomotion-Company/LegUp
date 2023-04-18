from dataclasses import dataclass
from typing import List, Optional, Any
from enum import Enum
from omegaconf import OmegaConf


@dataclass
class AnymalAgentConfig:
    env_name: str
    tensorboard_log_dir: str

    # command: List
    # turn_command: float
    # knee_threshold: List
    # max_torque: float
    # max_tilt: float
    # clip_bias: float
    # pos_delta_clip: float
    # phase_delta_clip: float
    # reward_scales: RewardScalesConfig
    # curriculum_exponent: float


@dataclass
class QuiltSimConfig:
    env_spacing: float
    num_agents_per_env: int  # env = patch
    num_envs_per_terrain_type: int

    num_terrain: int
    terrain_border: int
    vertical_terrain_scale: float
    horizontal_terrain_scale: float


@dataclass
class RL:
    parallel_envs: int
    n_steps: int
    n_epochs: int
    batch_size: int
    entropy_coef: float
    value_coef: float
    learning_rate: float
    gae_lambda: float
    discount: float
    clip_range: float
    verbose: int


def validate_single(klass: Any) -> None:
    OmegaConf.merge(klass, OmegaConf.structured(type(klass)))


def validate(*classes) -> None:
    [validate_single(klass) for klass in classes]
