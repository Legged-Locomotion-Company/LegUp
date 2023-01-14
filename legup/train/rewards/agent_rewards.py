from rewards import *


class WildAnymalReward:
    """
    Reward function for the Wild Anymal robot.
    https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf (pages 18+19)
    """

    def __init__(self, env, robot_config, train_config, dt: float):
        """Initialize reward function.

        Args:
            env (_type_): Isaac Gym environment
            robot_config (_type_): Robot interface configuration dictionary
            train_config (_type_): Configuration dictionary for training
            dt (float): Time step
        """
        self.env = env
        self.robot_config = robot_config
        self.reward_scales = train_config['reward_scales']

        self.dt = dt
        self.knee_threshold = train_config['knee_threshold']

    def __call__(self, previous_joint_velocities, joint_target_t_1, joint_target_t_2, curriculum_factor: float = 1.0) -> torch.Tensor:
        """Compute reward.

        Args:
            curriculum_factor (float, optional): Curriculum factor. Defaults to 1.0.

        Returns:
            torch.Tensor: Reward of shape (num_envs,)
        """

        v_des = self.env.get_desired_velocity()
        v_act = self.env.get_actual_velocity()
        w_des = self.env.get_desired_angular_velocity()
        w_act = self.env.get_actual_angular_velocity()

        velocity_reawrds = lin_velocity(v_des, v_act) + ang_velocity(
            w_des[:, 2], w_act[:, 2]) + linear_ortho_velocity(v_des, v_act)

        reward = self.reward_scales['velocity'] * velocity_reawrds

        reward += self.reward_scales['body_motion'] * \
            body_motion(v_act[:, 2], w_act[:, 0], w_act[:, 1])

        # is the foot height measured from the ground or from the body?
        # currrent implimentation is that is is measured from the body, ie foot 0.2m above ground would produce a value of -0.2
        # we need a way get multiple positions around the foot
        # get foot heights
        h = self.env.get_rb_position(
        )[:, self.robot_config['foot_indices'], 2]
        reward += self.reward_scales['foot_clearance'] * foot_clearance(h)

        # get positions of the shank and knee from config
        rigid_bodies = self.robot_config['shank_indices'].extend(
            self.robot_config['knee_indices'])
        contact_states = self.env.get_contact_states()[:, rigid_bodies]
        reward += self.reward_scales['shank_knee_col'] * \
            shank_or_knee_col(contact_states, curriculum_factor)

        # set joint velocities. If no joint history exists, set to zero
        joint_velocities = self.env.get_joint_velocities()
        if previous_joint_velocities is None:
            previous_joint_velocities = torch.zeros_like(joint_velocities)

        reward += self.reward_scales['joint_velocites'] * joint_motion(
            joint_velocities, previous_joint_velocities, self.dt, curriculum_factor)

        # update previous joint velocities
        previous_joint_velocities = joint_velocities

        # We only set a threshold for the knee joints.
        joint_positions = self.env.get_joint_positions()
        knee_joint_positions = joint_positions[:,
                                               self.robot_config['knee_indices']]
        reward += self.reward_scales['joint_constraint'] * \
            joint_constraint(knee_joint_positions, self.knee_threshold)

        # If no joint history exists (first iteration), set to zero
        if joint_target_t_1 is None:
            joint_target_t_1 = torch.zeros_like(joint_positions)
            joint_target_t_2 = torch.zeros_like(joint_positions)

        reward += self.reward_scales['target_smoothness'] * target_smoothness(
            joint_positions, joint_target_t_1, joint_target_t_2, curriculum_factor)

        # update joint target history
        joint_target_t_2 = joint_target_t_1
        joint_target_t_1 = joint_positions

        # calculate torque reward
        torques = self.env.get_joint_torques()
        reward += self.reward_scales['torque'] * \
            torque_reward(torques, curriculum_factor)

        # get what feet are in contact with the ground
        feet_contact = self.env.get_contact_states(
        )[:, self.robot_config['foot_indices']]

        # get the velocity of each foot
        feet_velocity = self.env.get_rb_velocity(
        )[:, self.robot_config['foot_indices']]

        reward += self.reward_scales['slip'] * \
            slip(feet_contact, feet_velocity, curriculum_factor)
