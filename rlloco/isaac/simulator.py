from isaacgym import gymapi

from typing import List

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
       
class SimulatorContext:
    """Handles the creation of the simulation environment and adding agents"""

    def __init__(self, num_environments: int, use_cuda: bool, asset_root: str, asset_path: str):
        """
        Args:
            num_environments (int): number of parallel environments to create in simulator
            use_cuda (bool): whether or not to use the GPU pipeline
            asset_root (str): root folder where the URDF to load is
            asset_path (str): path to URDF file from `asset_root`
        """
        # TODO: clean up parameters into objects, terrain config (flat/rough)
        
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.gym = gymapi.acquire_gym()
        self.sim = self.create_simulation_environment()

        self.asset_handle = self.create_asset(asset_root, asset_path)
        self.env_actor_handles = self.create_envs(self.asset_handle, num_environments)
    
        self.gym.prepare_sim(self.sim)

    def create_simulation_environment(self):
        """Creates a basic simulation environment through isaacgym. For now, it only adds the ground plane.

        Returns:
            Sim: IsaacGym "sim" object
        """
        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / 60.
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = self.use_cuda

        sim_params.physx.use_gpu = self.use_cuda
        sim_params.physx.num_threads = 8
        sim_params.physx.solver_type = 1  # more robust, slightly more expensive
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        # sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        # sim_params.physx.num_threads = 4
        # sim_params.physx.num_subscenes = 4
        # sim_params.physx.max_depenetration_velocity
        # sim_params.physx.friction_offset_threshold
        # sim_params.physx.friction_correlation_distance
        # sim_params.physx.default_buffer_size_multiplier
        # sim_params.physx.contact_offset
        # sim_params.physx.contact_collection
        # sim_params.physx.bounce_threshold_velocity
        # sim_params.physx.always_use_articulations
        sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        # plane_params.segmentation_id
        self.gym.add_ground(sim, plane_params)

        return sim

    def create_asset(self, asset_root: str, asset_path: str):
        """Loads an asset into the IsaacGym environment. It's only been tested with URDF files but it may work for other types

        Args:
            asset_root (str): root folder where the URDF to load is
            asset_path (str): path to URDF file from `asset_root`

        Returns:
            Handle: handle to asset in IsaacGym
        """
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        # TODO: more asset options

        return self.gym.load_asset(self.sim, asset_root, asset_path)

    def create_envs(self, asset_handle, num_environments: int, spacing: int = 1):
        """Adds actors into the environment created in the `create_simulation_environment` call. It will
        add one actor for each environment in a random position. Configurable positions and asset
        properties will be added in the future.
        
        Args:
            asset_handle (Handle): handle to asset in IsaacGym created by call to `create_asset`
            num_environments (int): number of environments to create
            spacing (int, optional): Spacing between each environment; the width of the environment is `2 * spacing`. Defaults to 1.

        Returns:
            List[Tuple[Handle, Handle]]: List containing handles to each created environment and actor in that environment
        """

        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(num_environments))

        pose = gymapi.Transform()
        identity_quat = R.identity().as_quat()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.35)  # TODO: figure out correct height
        pose.r = gymapi.Quat(*identity_quat)

        env_actor_handles = []
        for env_index in range(num_environments):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor = self.gym.create_actor(env, asset_handle, pose, f"Actor", env_index, 1)

            self.gym.enable_actor_dof_force_sensors(env, actor)

            asset_props = self.gym.get_asset_dof_properties(asset_handle)
            asset_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            asset_props["stiffness"].fill(17)
            asset_props["damping"].fill(0.4)

            self.gym.set_actor_dof_properties(env, actor, asset_props)

            env_actor_handles.append((env, actor))
            
        return env_actor_handles
    
    def release(self):
        """Frees any acquired resources and ends the simulation"""

        self.gym.destroy_sim(self.sim)
        self.gym.destroy_viewer(self.viewer)