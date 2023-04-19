from legup.common.rewards import reward, RewardArgs

import torch


class WildAnymalRewardArgs(RewardArgs):
    command: torch.Tensor
    curriculum_factor: float
    joint_position_hist: torch.Tensor
    joint_velocity_hist: torch.Tensor


@reward
def lin_velocity(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """If the norm of the desired velocity is 0, then the reward is exp(-norm(v_act) ^ 2)
    If the dot product of the desired velocity and the actual velocity is greater than the norm of the desired velocity, then the reward is 1.
    Otherwise, the reward is exp(-(dot(v_des, v_act) - norm(v_des)) ^ 2)
    """

    # we need the norm of v_des 3 times, so we calculate it once and store it
    # remove the last dimension of v_des_norm, since it is 1
    v_des = reward_args.command
    v_act = reward_args.dynamics.get_linear_velocity()

    v_act_norm_sq = torch.einsum('Bi,Bi->B', v_act, v_act)
    v_des_norm = torch.norm(v_des, dim=-1).squeeze()  # (num_envs,)

    act_dot_des = torch.einsum('Bi,Bi->B', v_des, v_act)

    result = (-(act_dot_des - v_des_norm).square()).exp()

    mask = v_des_norm == 0.0
    result[mask] = (-torch.einsum(v_act_norm_sq[mask])).exp()

    mask = act_dot_des > v_des_norm
    result[mask] = 1.0

    return result


@reward
def ang_velocity(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """If the desired angular velocity is 0, then the reward is exp(-(w_act_yaw) ^ 2).
    If the dot product of the desired angular velocity and the actual angular velocity is greater than the desired angular velocity, then the reward is 1.
    Otherwise, the reward is exp(-(dot(w_des_yaw, w_act_yaw) - w_des_yaw) ^ 2)
    """

    if (command := getattr(reward_args, 'command')) is None:
        command = torch.zeros((reward_args.dynamics.get_num_agents(), 3),
                              device=reward_args.device)

    w_des_yaw = command[:, :2]
    w_act_yaw = reward_args.dynamics.get_angular_velocity()[:, :2]

    dots = w_des_yaw * w_act_yaw

    result = torch.exp(-(dots - w_des_yaw)**2)

    mask = w_des_yaw == 0.0
    result[mask] = torch.exp(-w_act_yaw[mask]**2)

    mask = dots > w_des_yaw
    result[mask] = 1.0

    result = result.squeeze()

    return result


@reward
def linear_ortho_velocity(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """
    This term penalizes the velocity orthogonal to the target direction .
    Reward is exp(-3 * norm(v_0) ^ 2), where v_0 is v_act-(dot(v_des, v_act))*v_des.
    """

    v_des = reward_args.command
    v_act = reward_args.dynamics.get_linear_velocity()

    dot_v_des_v_act = torch.sum(
        v_des * v_act, dim=1).unsqueeze(1)  # (num_envs,1)
    v_0 = v_act - dot_v_des_v_act * v_des  # (num_envs,3)

    x = torch.exp(-3 * squared_norm(v_0))

    return x


@reward
def body_motion(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """This term penalizes the body velocity in directions not part of the command
    Reward is -1.25*v_z ^ 2 - 0.4 * abs(w_x) - 0.4 * abs(w_y)
    """
    v_z = reward_args.dynamics.get_linear_velocity()[:, 2]
    w_x = reward_args.dynamics.get_angular_velocity()[:, 0]
    w_y = reward_args.dynamics.get_angular_velocity()[:, 1]

    x = (-1.25*torch.pow(v_z, 2) - 0.4 *
         torch.abs(w_x) - 0.4 * torch.abs(w_y))

    return x


@reward
def foot_clearance(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """Penalizes the model if the foot is more than 0.2 meters above the ground, with a reward of - 1 per foot that is not in compliance."""

    foot_idxs = reward_args.robot.primary_contact_link_model_idxs

    h = reward_args.dynamics.get_rb_position()[:, foot_idxs, 2]

    x = torch.sum(-1.0 * (h > 0.2), dim=1)

    return x


@reward
def shank_or_knee_col(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """If any of the shanks or knees are in contact with the ground, the reward is -curriculum_factor"""

    shank_link_idxs = reward_args.robot.secondary_contact_link_model_idxs

    is_col = reward_args.dynamics.get_contact_states()[:, shank_link_idxs]

    x = -reward_args.curriculum_factor * torch.any(is_col, dim=1)

    return x


@reward
def joint_motion(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """This term penalizes the joint velocity and acceleration to avoid vibrations"""

    j_vel_hist = reward_args.joint_velocity_hist
    j_vel = j_vel_hist[0]
    j_vel_t_1 = j_vel_hist[1]

    accel = (j_vel - j_vel_t_1)/reward_args.dynamics.get_dt()
    x = -reward_args.curriculum_factor * \
        torch.sum(0.01*(j_vel)**2 + accel**2, dim=1)

    return x


@reward
def joint_constraint(reward_args: WildAnymalRewardArgs):
    """This term penalizes the joint position if it is outside of the joint limits."""

    joint_pos = reward_args.dynamics.get_joint_position()
    q = joint_pos[:, reward_args.robot.limited_joint_model_idxs]
    q_th = reward_args.robot.limited_joint_limits

    mask = q > q_th
    x = torch.sum(-torch.pow(q-q_th, 2) * mask, dim=1)

    return x


@reward
def target_smoothness(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """This term penalizes the smoothness of the target foot trajectories"""

    joint_target_hist = reward_args.dynamics.joint_pos
    joint_target_t = joint_target_hist[0]
    joint_target_tm1 = joint_target_hist[1]
    joint_target_tm2 = joint_target_hist[2]

    x = -reward_args.curriculum_factor * torch.sum((joint_target_t - joint_target_tm1)**2 + (
        joint_target_t - 2*joint_target_tm1 + joint_target_tm2)**2, dim=1)

    return x


@reward
def torque(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """This term penalizes the torque to reduce energy consumption"""

    tau = reward_args.dynamics.get_joint_torque()

    x = -reward_args.curriculum_factor * torch.sum(tau**2, dim=1)

    return x


@reward
def slip(reward_args: WildAnymalRewardArgs) -> torch.Tensor:
    """We penealize the foot velocity if the foot is in contact with the ground to reduce slippage"""

    foot_idxs = reward_args.robot.primary_contact_link_model_idxs
    foot_is_in_contact = reward_args.dynamics.get_contact_states()[
        :, foot_idxs]

    feet_vel = reward_args.dynamics.get_rb_linear_velocity()[
        :, foot_idxs]

    x = -reward_args.curriculum_factor * \
        torch.sum((foot_is_in_contact * squared_norm(feet_vel)), dim=1)

    return x


def squared_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """Calculates the squared norm of a tensor
    Args:
        x(torch.Tensor): Arbitrarily shaped tensor
        dim(int, optional): Dimension to calculate the norm over. Defaults to - 1.
    Returns:
        torch.Tensor: The squared norm of the tensor across the given dimension
    """
    return torch.sum(torch.pow(x, 2), dim=dim)
