from dataclasses import dataclass

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

@dataclass 
class TerrainConfig:
    env_spacing: float
    terrain_border: int
    slope_threshold: float
    vertical_terrain_scale: float
    horizontal_terrain_scale: float

@dataclass
class CameraConfig:
    capture_width: int
    capture_height: int
    render_target_env: int
    render_target_actor: int

    headless: bool
    draw_collision_mesh: bool
    draw_position: bool
    draw_rotation: bool
    draw_command: bool

@dataclass
class IsaacConfig:    
    terrain_config: TerrainConfig
    asset_config: AssetConfig
    sim_config: SimulationConfig
    camera_config: CameraConfig