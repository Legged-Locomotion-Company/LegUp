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

        reward_log = {}
        reward = torch.zeros(self.env.num_environments).to(self.env.device)

        v_act = self.env.get_linear_velocity()[:, :2]
        v_des = torch.tensor([self.train_cfg.command[0], self.train_cfg.command[1]]).to(self.env.device).expand(v_act.shape)

        w_act = self.env.get_angular_velocity()
        w_des = torch.zeros_like(w_act)
        w_des[:] = self.train_cfg.turn_command

        velocity_rewards = lin_velocity(v_des, v_act) + ang_velocity(
            w_des[:, 2], w_act[:, 2]) + linear_ortho_velocity(v_des, v_act) * self.reward_scales.velocity

        reward_log['lin_velocity_reward'] = velocity_rewards
        reward += velocity_rewards

        body_motion = self.reward_scales.body_motion * \
            body_motion(v_act[:, 2], w_act[:, 0], w_act[:, 1])

        reward_log['body_motion_reward'] = body_motion
        reward += body_motion

        # is the foot height measured from the ground or from the body?
        # currrent implimentation is that is is measured from the body, ie foot 0.2m above ground would produce a value of -0.2
        # we need a way get multiple positions around the foot
        # get foot heights
        h = self.env.get_rb_position(
        )[:, self.robot_config.foot_indices, 2]

        foot_clearance_reward = self.reward_scales.foot_clearance * foot_clearance(h)
        reward_log['foot_clearance_reward'] = foot_clearance_reward
        reward += foot_clearance_reward

        

        # set joint velocities. If no joint history exists, set to zero
        joint_velocities = self.env.get_joint_velocity()
        if previous_joint_velocities is None:
            previous_joint_velocities = torch.zeros_like(joint_velocities)

        joint_velocity_reward = self.reward_scales.joint_velocities * joint_motion(
            joint_velocities, previous_joint_velocities, self.dt, curriculum_factor)

        reward_log['joint_velocity_reward'] = joint_velocity_reward
        reward += joint_velocity_reward

        # update previous joint velocities
        previous_joint_velocities = joint_velocities

        # We only set a threshold for the knee joints.
        joint_positions = self.env.get_joint_position()
        knee_joint_positions = joint_positions[:,
                                               self.robot_config.knee_indices]
        joint_constraints_reward = self.reward_scales.joint_constraints * \
            joint_constraint(knee_joint_positions, self.knee_threshold)

        reward_log['joint_constraints_reward'] = joint_constraints_reward
        reward += joint_constraints_reward

        # If no joint history exists (first iteration), set to zero
        if joint_target_t_1 is None:
            joint_target_t_1 = torch.zeros_like(joint_positions)
            joint_target_t_2 = torch.zeros_like(joint_positions)

        target_smoothness_reward = self.reward_scales.target_smoothness * target_smoothness(
            joint_positions, joint_target_t_1, joint_target_t_2, curriculum_factor)

        reward_log['target_smoothness_reward'] = target_smoothness_reward
        reward += target_smoothness_reward

        # update joint target history
        joint_target_t_2 = joint_target_t_1
        joint_target_t_1 = joint_positions

        # calculate torque reward
        torques = self.env.get_joint_torque()
        torque_reward = self.reward_scales.torque * \
            torque_reward(torques, curriculum_factor)

        reward_log['torque_reward'] = torque_reward
        reward += torque_reward

        # get what feet are in contact with the ground
        feet_contact = self.env.get_contact_states(
        )[:, self.robot_config.foot_indices]

        # get the velocity of each foot
        feet_velocity = self.env.get_rb_linear_velocity(
        )[:, self.robot_config.foot_indices]

        slip_reward = self.reward_scales.slip * \
            slip(feet_contact, feet_velocity, curriculum_factor)
        
        reward_log['slip_reward'] = slip_reward
        reward += slip_reward

        return reward, reward_log.keys(), reward_log.values()
