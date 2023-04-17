from dataclasses import dataclass
from typing import List, Any, TypeVar, Union

import torch
from omegaconf import OmegaConf, ListConfig, DictConfig

@dataclass
class RewardScalesConfig:
    velocity: float
    body_motion: float
    foot_clearance: float
    shank_knee_col: float
    joint_velocities: float
    joint_constraints: float
    target_smoothness: float
    torque: float
    slip: float
    pos_delta_clip: float
    phase_clip: float

@dataclass
class AnymalAgentConfig:
    env_name: str
    tensorboard_log_dir: str
    command: List
    turn_command: float
    knee_threshold: List
    max_torque: float
    max_tilt: float
    clip_bias: float
    pos_delta_clip: float
    phase_delta_clip: float
    reward_scales: RewardScalesConfig
    curriculum_exponent: float   

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