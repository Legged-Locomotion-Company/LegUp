from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

@dataclass
class AgentConfig:
    env_name: str
    tensorboard_log_dir: str

@dataclass
class AssetConfig:
    asset_path: str
    filename: str
    stiffness: float
    damping: float

@dataclass
class SimulationConfig:
    dt: float
    substeps: int
    num_threads: int
    num_position_iterations: int
    num_velocity_iterations: int
    use_gpu: bool

# class TerrainType(Enum):
#     FLAT = 0
#     ROUGH = 1
#     ANGLED = 2
#     STEPS = 3

# @dataclass 
# class TerrainConfig:
#     type: TerrainType

#     rough_sigma: float
#     angled_slope: float
#     step_width: float
#     step_height: float

@dataclass
class CameraConfig:
    capture_width: int
    capture_height: int
    render_target_env: int
    render_target_actor: int

    draw_collision_mesh: bool
    draw_position: bool
    draw_rotation: bool
    draw_command: bool

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


# @dataclass
# class LegupConfig:
#     eval: bool
#     headless: bool
#     use_gpu: bool

#     agent: AgentConfig
#     env: IsaacConfig
#     rl: RL
    

    