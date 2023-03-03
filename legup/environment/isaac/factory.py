import torch
import numpy as np
from isaacgym import gymapi, terrain_utils

from legup.common.legup_config import IsaacConfig
from legup.common.abstract_agent import AbstractAgent


class IsaacGymFactory:

    @staticmethod
    def create_sim(gym, config: IsaacConfig):
        sim_params = gymapi.SimParams()  # type: ignore
        sim_params.dt = 1. / config.sim_config.dt
        sim_params.substeps = config.sim_config.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z  # type: ignore
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # type: ignore
        sim_params.use_gpu_pipeline = config.sim_config.use_gpu

        sim_params.physx.use_gpu = config.sim_config.use_gpu
        sim_params.physx.num_threads = config.sim_config.num_threads
        sim_params.physx.solver_type = 1  # more robust, slightly more expensive
        sim_params.physx.num_position_iterations = config.sim_config.num_position_iterations
        sim_params.physx.num_velocity_iterations = config.sim_config.num_velocity_iterations
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
        return gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    @staticmethod
    def create_terrain(sim, gym, agent: AbstractAgent, config: IsaacConfig):
        '''
            - each environment has a width of config.env_spacing and height of config.env_spacing
            - should probably add some border to each environment too
            - this is scaled with the horizontal scaling to get the number of points it represents in the heightmap
            - this heightmap is populated using the terrain_utils functions
            - and then converted to a triangle mesh with terrain_utils
        '''
        num_envs = config.num_envs_per_terrain_type * \
            config.num_terrain  # total number of environments/patches in simulation
        # since we have a square quilt, this is the side length
        num_columns = int(np.sqrt(num_envs))
        # since we have a square quilt, this is the side width
        num_rows = int(np.ceil(num_envs / num_columns))

        patch_width = 2 * config.env_spacing / config.horizontal_terrain_scale
        border_size = 2 * config.terrain_border / config.horizontal_terrain_scale
        terrain_len_columns = num_columns * patch_width + border_size
        terrain_len_rows = num_rows * patch_width + border_size

        heightfield = np.zeros(
            (terrain_len_columns, terrain_len_rows), dtype=np.int16)  # type: ignore

        # idxs = np.unravel_index(np.arange(num_envs), (num_columns, num_rows))
        # for cx, cy in zip(*idxs):
        #     heightfield[cx, cy]

        # vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        # tm_params = gymapi.TriangleMeshParams()
        # tm_params.nb_vertices = vertices.shape[0]
        # tm_params.nb_triangles = triangles.shape[0]
        # tm_params.transform.p.x = -1.
        # tm_params.transform.p.y = -1.
        # gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

        return heightfield

    @staticmethod
    def create_actors(sim, gym, agent: AbstractAgent, config: IsaacConfig):
        # create asset and initialize relevant properties
        asset_options = gymapi.AssetOptions()  # TODO: more asset options
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_handle = gym.load_asset(
            sim, config.asset_config.asset_path, config.asset_config.filename)

        asset_props = gym.get_asset_dof_properties(asset_handle)
        asset_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        asset_props["stiffness"].fill(config.asset_config.stiffness)
        asset_props["damping"].fill(config.asset_config.damping)

        # create environment structures
        spacing = config.env_spacing
        num_envs = config.num_envs_per_terrain_type * config.num_terrain
        num_per_row = int(np.sqrt(num_envs))

        lower_space = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper_space = gymapi.Vec3(spacing, spacing, spacing)

        actor_positions = agent.sample_new_position(
            num_envs * config.num_agents_per_env, tuple(lower_space), tuple(upper_space))
        actor_rotations = agent.sample_new_quaternion(
            num_envs * config.num_agents_per_env)

        envs, actors = [], []

        # initialize environments and agents in simulation
        for env_idx in range(num_envs):
            env_handle = gym.create_env(
                sim, lower_space, upper_space, num_per_row)
            for actor_idx in range(config.num_agents_per_env):
                idx = env_idx * num_envs + actor_idx
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*actor_positions[idx])
                pose.r = gymapi.Quat(*actor_rotations[idx])

                actor_handle = gym.create_actor(
                    env_handle, asset_handle, pose, f"env-{env_idx}_actor-{actor_idx}", env_idx, 1)

                gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
                gym.set_actor_dof_properties(
                    env_handle, actor_handle, asset_props)

                actors.append(actor_handle)

            envs.append(env_handle)

        return envs, actors, asset_handle

    @staticmethod
    def create_camera(sim, gym, config: IsaacConfig):
        camera_props = gymapi.CameraProperties()
        camera_props.width = config.camera_config.capture_width
        camera_props.height = config.camera_config.capture_height
        camera_props.use_collision_geometry = config.camera_config.draw_collision_mesh

        render_env = gym.get_env(config.camera_config.render_target_env)
        render_actor = gym.get_actor_handle(
            render_env, config.camera_config.render_target_actor)
        render_body_target = gym.get_actor_rigid_body_handle(
            render_env, render_actor, 0)  # 0 = body

        camera_handle = gym.create_camera_sensor(render_env, camera_props)

        camera_offset = gymapi.Vec3(-0.5, -0.5, 1)
        camera_rotation = gymapi.Quat().from_euler_zyx(
            np.radians(0), np.radians(45), np.radians(45))
        camera_transform = gymapi.Transform(camera_offset, camera_rotation)

        gym.attach_camera_to_body(
            camera_handle, render_env, render_body_target, camera_transform, gymapi.FOLLOW_POSITION)
        gym.set_light_parameters(sim, 0, gymapi.Vec3(
            0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0, 0, 0))

        return camera_handle
