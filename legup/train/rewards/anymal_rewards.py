from legup.robots.Robot import Robot
from legup.train.rewards.rewards import *

from omegaconf import DictConfig
import torch

class WildAnymalReward:
    """
    Reward function for the Wild Anymal robot.
    https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf (pages 18+19)
    """

    def __init__(self, env, robot_cfg: Robot, train_cfg: DictConfig, dt: float):
        """Initialize reward function.

        Args:
            env: Isaac Gym environment
            robot_config (Robot): Robot interface configuration class
            train_config (DictConfig): Configuration dictionary for training
            dt (float): Time step
        """
        self.env = env
        self.robot_config = robot_cfg
        self.reward_scales = train_cfg['reward_scales']

        self.dt = dt
        self.knee_threshold = train_cfg['knee_threshold']

        self.train_cfg = train_cfg

    def __call__(self, previous_joint_velocities: torch.Tensor, joint_target_t_1: torch.Tensor, joint_target_t_2: torch.Tensor, curriculum_factor: float = 1.0) -> torch.Tensor:
        """Compute reward.

        Args:
            curriculum_factor (float, optional): Curriculum factor. Defaults to 1.0.

        Returns:
            torch.Tensor: Reward of shape (num_envs,)
        """

        v_act = self.env.get_linear_velocity()
        v_des = torch.zeros_like(v_act)
        for i in range(3):
            v_des[:, i] = self.train_cfg.command[i]

        w_act = self.env.get_angular_velocity()
        w_des = torch.zeros_like(w_act)
        w_des[:] = self.train_cfg.turn_command

        velocity_rewards = lin_velocity(v_des, v_act) + ang_velocity(
            w_des[:, 2], w_act[:, 2]) + linear_ortho_velocity(v_des, v_act)

        reward = self.reward_scales.velocity * velocity_rewards

        reward += self.reward_scales.body_motion * \
            body_motion(v_act[:, 2], w_act[:, 0], w_act[:, 1])

        # is the foot height measured from the ground or from the body?
        # currrent implimentation is that is is measured from the body, ie foot 0.2m above ground would produce a value of -0.2
        # we need a way get multiple positions around the foot
        # get foot heights
        h = self.env.get_rb_position(
        )[:, self.robot_config.foot_indices, 2]
        reward += self.reward_scales.foot_clearance * foot_clearance(h)

        # get positions of the shank and knee from config
        rigid_bodies = self.robot_config.shank_indices + self.robot_config.knee_indices
        contact_states = self.env.get_contact_states()[:, rigid_bodies]
        reward += self.reward_scales.shank_knee_col * \
            shank_or_knee_col(contact_states, curriculum_factor)

        # set joint velocities. If no joint history exists, set to zero
        joint_velocities = self.env.get_joint_velocity()
        if previous_joint_velocities is None:
            previous_joint_velocities = torch.zeros_like(joint_velocities)

        reward += self.reward_scales.joint_velocities * joint_motion(
            joint_velocities, previous_joint_velocities, self.dt, curriculum_factor)

        # update previous joint velocities
        previous_joint_velocities = joint_velocities

        # We only set a threshold for the knee joints.
        joint_positions = self.env.get_joint_position()
        knee_joint_positions = joint_positions[:,
                                               self.robot_config.knee_indices]
        reward += self.reward_scales.joint_constraints * \
            joint_constraint(knee_joint_positions, self.knee_threshold)

        # If no joint history exists (first iteration), set to zero
        if joint_target_t_1 is None:
            joint_target_t_1 = torch.zeros_like(joint_positions)
            joint_target_t_2 = torch.zeros_like(joint_positions)

        reward += self.reward_scales.target_smoothness * target_smoothness(
            joint_positions, joint_target_t_1, joint_target_t_2, curriculum_factor)

        # update joint target history
        joint_target_t_2 = joint_target_t_1
        joint_target_t_1 = joint_positions

        # calculate torque reward
        torques = self.env.get_joint_torque()
        reward += self.reward_scales.torque * \
            torque_reward(torques, curriculum_factor)

        # get what feet are in contact with the ground
        feet_contact = self.env.get_contact_states(
        )[:, self.robot_config.foot_indices]

        # get the velocity of each foot
        feet_velocity = self.env.get_rb_linear_velocity(
        )[:, self.robot_config.foot_indices]

        reward += self.reward_scales.slip * \
            slip(feet_contact, feet_velocity, curriculum_factor)

        return reward
