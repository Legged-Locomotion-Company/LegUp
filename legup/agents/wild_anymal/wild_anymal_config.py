from dataclasses import dataclass

from legup.common.legup_config import AgentConfig

from typing import List, Dict


@dataclass
class WildAnymalRewardScales:
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
class WildAnymalConfig(AgentConfig):
    max_torque: float
    max_tilt: float
    pos_delta_clip: float
    phase_delta_clip: float
    reward_scales: WildAnymalRewardScales
    curriculum_exponent: float
