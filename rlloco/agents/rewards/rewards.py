import torch
from rewards.reward_helpers import squared_norm

"""
This file contains the reward functions for the agents.
Currently implemented functions all come from the Wild Anymal paper https://leggedrobotics.github.io/rl-perceptiveloco/assets/pdf/wild_anymal.pdf (pages 18+19)
"""


def lin_velocity(v_des: torch.Tensor, v_act: torch.Tensor):
    """If the norm of the desired velocity is 0, then the reward is exp(-norm(v_act)^2)\n
    If the dot product of the desired velocity and the actual velocity is greater than the norm of the desired velocity, then the reward is 1\n
    Otherwise, the reward is exp(-(dot(v_des, v_act) - norm(v_des))^2)


    Args:
        v_des (torch.Tensor): Desired X,Y,Z velocity of shape (num_envs, 3)
        v_act (torch.Tensor): Actual  X,Y,Z velocity of shape (num_envs, 3)

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    # we need the norm of v_des 3 times, so we calculate it once and store it
    # remove the last dimension of v_des_norm, since it is 1
    v_des_norm = torch.norm(v_des, dim=1).squeeze()  # (num_envs,)

    # calculate the dot product of v_des and v_act across dim 1
    dot_v_des_v_act = torch.sum(v_des * v_act, dim=1)  # (num_envs,)

    # use masks to select the correct reward function for a given env
    x = torch.exp(squared_norm(v_act)) * (v_des_norm == 0)
    x = 1 * (dot_v_des_v_act > v_des_norm)
    x = torch.exp(-torch.pow(dot_v_des_v_act - v_des_norm, 2)) * \
        ~((v_des == 0) + torch.sum(v_des * v_act, dim=1) > v_des_norm)

    return x


def ang_velocity(w_des_yaw, w_act_yaw):
    """If the desired angular velocity is 0, then the reward is exp(-(w_act_yaw)^2)\n
    If the dot product of the desired angular velocity and the actual angular velocity is greater than the desired angular velocity, then the reward is 1\n
    Otherwise, the reward is exp(-(dot(w_des_yaw,w_act_yaw) - w_des_yaw)^2)

    Args:
        w_des_yaw (torch.Tensor): Desired yaw velocity of shape (num_envs, 1)
        w_act_yaw (torch.Tensor): Actual yaw velocity of shape (num_envs, 1)

    Returns:
        torch.Tensor: the reward for each env of shape (num_envs,)
    """
    w_act_yaw = w_act_yaw.squeeze()  # (num_envs,)
    w_des_yaw = w_des_yaw.squeeze()  # (num_envs,)

    # dot product = elementwise multiplication since w_des_yaw and w_act_yaw are 1D
    dot_w_des_w_act = w_des_yaw * w_act_yaw

    x = torch.exp(-torch.pow(w_act_yaw, 2)) * (w_des_yaw == 0)
    x = 1 * (dot_w_des_w_act > w_des_yaw)
    x = torch.exp(-torch.pow(dot_w_des_w_act - w_des_yaw, 2)) * \
        ~((w_des_yaw == 0) + (w_des_yaw * w_act_yaw > w_des_yaw))

    return x


'''
Calculates the linear orthogonal velocity given:
v_des: desired velocity 
v_act: actual velocity
'''


def linear_ortho_velocity(v_des, v_act):
    return torch.exp(-3 * torch.pow(torch.norm(v_act-(v_des*v_act)*v_des, dim=1), 2))


'''
Calculates the motion of the body velocity which is not part of the command
v_z: velocity in the z direction
w_x: omega in the x direction
w_y: omega in the y direction
'''


def body_motion(v_z, w_x, w_y):
    return -1.25*v_z**2 - 0.4 * torch.abs(w_x) - 0.4 * torch.abs(w_y)


'''
Calculates the foot clearance
h: height of requested foot clearance, (num_envs, 4, num_heights_per_foot)
'''


def foot_clearance(h):
    return (torch.max(h, dim=1)[0] < -0.2).to(torch.long).to(device)


def shank_or_knee_col(is_col):
    return -self.curriculum_factor * torch.any(is_col, dim=1)


def joint_motion(j_vel, j_vel_t_1):
    accel = (j_vel - j_vel_t_1)/self.dt
    self.prev_joint_vel = j_vel
    return -self.curriculum_factor * torch.sum(0.01*(j_vel)**2 + accel**2, dim=1)

# q = joint positions
# q_th = joint position thresholds


def joint_contraint(q, q_th):
    mask = q > q_th
    return torch.sum(-torch.pow(q-q_th, 2) * mask, dim=1)

# q_t_des  = joint position desired
# q_t_1_des = joint position desired at previous time step


def target_smoothness(q_t_des, q_t_1_des, q_t_2_des):
    # self.prev_q_des = torch.cat(
    #    (self.prev_q_des[:, 1].reshape(self.num_envs, 12, 1), q_t_des), dim=2)

    # self.prev_q_des[:, :, 1] = self.prev_q_des[:, :, 0]
    #self.prev_q_des[:, :, 0] = self.env.get_joint_position()
    return -self.curriculum_factor * torch.sum((q_t_des - q_t_1_des)**2 + (q_t_des - 2*q_t_1_des + q_t_2_des)**2, dim=1)


def torque_reward(tau):
    return -self.curriculum_factor * torch.sum(tau**2, dim=1)


'''
foot in contact = truthy array
feet_vel = feet velocity
'''


def slip(foot_is_in_contact, feet_vel):
    return -self.curriculum_factor * torch.sum((foot_is_in_contact * feet_vel)**2, dim=1)
