from isaacgym import gymapi, terrain_utils
import torch

from typing import Optional, List

import numpy as np

from legup.environment.isaac.config import IsaacConfig, AssetConfig, CameraConfig, SimulationConfig, TerrainConfig
from legup.common.abstract_agent import AbstractAgent
from legup.common.abstract_terrain import AbstractTerrain

class IsaacGymFactory:

    @staticmethod
    def create_sim(gym, config: SimulationConfig):
        sim_params = gymapi.SimParams() # type: ignore
        sim_params.dt = 1. / config.dt
        sim_params.substeps = config.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z # type: ignore
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81) # type: ignore
        sim_params.use_gpu_pipeline = config.use_gpu

        sim_params.physx.use_gpu = config.use_gpu
        sim_params.physx.num_threads = config.num_threads
        sim_params.physx.solver_type = 1  # more robust, slightly more expensive
        sim_params.physx.num_position_iterations = config.num_position_iterations
        sim_params.physx.num_velocity_iterations = config.num_velocity_iterations
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
        return gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params) # type: ignore

    @staticmethod
    def create_terrain(sim, gym, config: TerrainConfig, terrains: List[AbstractTerrain]) -> torch.Tensor:
        '''
            - each environment has a width of config.env_spacing and height of config.env_spacing
            - should probably add some border to each environment too
            - this is scaled with the horizontal scaling to get the number of points it represents in the heightmap
            - this heightmap is populated using the terrain_utils functions
            - and then converted to a triangle mesh with terrain_utils
        '''

        num_envs = sum([terr.get_num_patches() for terr in terrains]) # total number of environments/patches in simulation # nopep8
        num_columns = int(np.sqrt(num_envs))  # since we have a square quilt, this is the side length # nopep8
        num_rows = int(np.ceil(num_envs / num_columns))  # since we have a square quilt, this is the side width # nopep8

        patch_width = int(2 * config.env_spacing / config.horizontal_terrain_scale)
        border_size = int(2 * config.terrain_border / config.horizontal_terrain_scale)
        terrain_len_columns = num_columns * patch_width + border_size
        terrain_len_rows = num_rows * patch_width + border_size

        heightfield = np.zeros((terrain_len_columns, terrain_len_rows), dtype=np.int16)

        for col_idx in range(num_columns):
            for row_idx in range(num_rows):
                col_ptr, row_ptr = col_idx * patch_width, row_idx * patch_width
                terrain_slice = heightfield[col_ptr:col_ptr+patch_width, row_ptr:row_ptr+patch_width]
                
                patch_terrain = terrains[col_idx * num_rows + row_idx]
                terrain_slice[...] = patch_terrain.create_heightfield(patch_width, config.horizontal_terrain_scale, config.vertical_terrain_scale)

        vertices, triangles = \
            terrain_utils.convert_heightfield_to_trimesh(heightfield,
                                                         horizontal_scale=config.horizontal_terrain_scale,
                                                         vertical_scale=config.vertical_terrain_scale,
                                                         slope_threshold=config.slope_threshold)

        tm_params = gymapi.TriangleMeshParams() # type: ignore
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -1.
        tm_params.transform.p.y = -1.
        gym.add_triangle_mesh(sim, vertices.flatten(),
                              triangles.flatten(), tm_params)

        heightfield = torch.from_numpy(heightfield).float()
        return heightfield.unfold(0, patch_width, patch_width).unfold(1, patch_width, patch_width).reshape(-1, patch_width, patch_width)

    @staticmethod
    def create_actors(sim, gym, agent: AbstractAgent, config: IsaacConfig, terrains: List[AbstractTerrain]):
        # create asset and initialize relevant properties
        asset_options = gymapi.AssetOptions() # type: ignore # TODO: more asset options
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_handle = gym.load_asset(
            sim, config.asset_config.asset_path, config.asset_config.filename)

        asset_props = gym.get_asset_dof_properties(asset_handle)
        asset_props["driveMode"].fill(gymapi.DOF_MODE_POS) # type: ignore
        asset_props["stiffness"].fill(config.asset_config.stiffness)
        asset_props["damping"].fill(config.asset_config.damping)

        # create environment structures
        spacing = config.terrain_config.env_spacing
        num_envs = sum([terr.get_num_patches() for terr in terrains])
        num_agents = sum([terr.get_num_robots() * terr.get_num_patches() for terr in terrains])
        num_per_row = int(np.sqrt(num_envs))

        lower_space = gymapi.Vec3(-spacing, -spacing, 0.0) # type: ignore
        upper_space = gymapi.Vec3(spacing, spacing, spacing) # type: ignore

        actor_positions = agent.sample_new_position(
            num_agents, tuple(lower_space), tuple(upper_space))
        actor_rotations = agent.sample_new_quaternion(num_agents)

        envs, actors = [], []

        # initialize environments and agents in simulation
        global_idx = 0
        for terrain in terrains:
            for patch_idx in range(terrain.get_num_patches()):
                env_handle = gym.create_env(sim, lower_space, upper_space, num_per_row)
                
                for actor_idx in range(terrain.get_num_robots()):
                    pose = gymapi.Transform() # type: ignore
                    pose.p = gymapi.Vec3(*actor_positions[global_idx]) # type: ignore
                    pose.r = gymapi.Quat(*actor_rotations[global_idx]) # type: ignore

                    actor_handle = gym.create_actor(
                        env_handle, asset_handle, pose, f"env-{patch_idx}_actor-{actor_idx}", patch_idx, 1)

                    gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
                    gym.set_actor_dof_properties(
                        env_handle, actor_handle, asset_props)

                    actors.append(actor_handle)
                    global_idx += 1

                envs.append(env_handle)

        return envs, actors, asset_handle

    @staticmethod
    def create_camera(sim, gym, config: CameraConfig):
        camera_props = gymapi.CameraProperties() # type: ignore
        camera_props.width = config.capture_width
        camera_props.height = config.capture_height
        camera_props.use_collision_geometry = config.draw_collision_mesh

        render_env = gym.get_env(config.render_target_env)
        render_actor = gym.get_actor_handle(
            render_env, config.render_target_actor)
        render_body_target = gym.get_actor_rigid_body_handle(
            render_env, render_actor, 0)  # 0 = body

        camera_handle = gym.create_camera_sensor(render_env, camera_props)

        camera_offset = gymapi.Vec3(-0.5, -0.5, 1) # type: ignore
        camera_rotation = gymapi.Quat().from_euler_zyx( # type: ignore
            np.radians(0), np.radians(45), np.radians(45))
        camera_transform = gymapi.Transform(camera_offset, camera_rotation) # type: ignore

        gym.attach_camera_to_body(
            camera_handle, render_env, render_body_target, camera_transform, gymapi.FOLLOW_POSITION) # type: ignore
        gym.set_light_parameters(sim, 0,
                                 gymapi.Vec3(0.8, 0.8, 0.8), # type: ignore
                                 gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0, 0, 0)) # type: ignore

        return camera_handle
