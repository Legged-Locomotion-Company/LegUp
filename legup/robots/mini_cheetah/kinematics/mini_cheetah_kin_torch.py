from legup.robots.common.kinematics.kin_utils import *
from legup.robots.common.kinematics.inv_kin_algs import dls_invkin, pinv_invkin
from legup.robots.Robot import Robot

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def get_home_position():
# return torch.tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).to(device)


def build_jacobian_and_fk(q_vec, leg):
    # create the multipliers depending on the leg
    front = 1 if (leg == 0 or leg == 1) else -1
    left = 1 if (leg == 0 or leg == 2) else -1

    NUM_ENVS, _ = q_vec.shape

    hip_angles = q_vec[:, 3*leg + 0]
    shoulder_angles = q_vec[:, 3*leg + 1]
    knee_angles = q_vec[:, 3*leg + 2]

    # Create the homogenous transforms for the joints in their zero position
    # [0.14775, 0.049, 0]
    hip_to_torso_zero_v = torch.tensor([-0.14775*front, -0.049*left, 0])
    hip_to_torso_zero_ht = translation_ht(hip_to_torso_zero_v)

    # [0.055, -0.019, 0]
    shoulder_to_hip_zero_v = torch.tensor([-0.055*front, -0.019*left, 0])
    shoulder_to_hip_zero_ht = translation_ht(shoulder_to_hip_zero_v)

    # [0, 0.049, -0.2085]
    knee_to_shoulder_zero_v = torch.tensor([0, -0.049*left, 0.2085])
    knee_to_shoulder_zero_ht = translation_ht(knee_to_shoulder_zero_v)

    # [0, 0, -0.194]
    foot_to_knee_zero_v = torch.tensor([0, 0, 0.194])
    foot_to_knee_zero_ht = translation_ht(foot_to_knee_zero_v)

    foot_to_shoulder_zero_ht = foot_to_knee_zero_ht @ knee_to_shoulder_zero_ht
    foot_to_hip_zero_ht = foot_to_shoulder_zero_ht @ shoulder_to_hip_zero_ht
    foot_to_torso_zero_ht = foot_to_hip_zero_ht @ hip_to_torso_zero_ht

    torso_to_foot_zero_ht = invert_ht(foot_to_torso_zero_ht)

    # create rotation axes
    hip_ax = torch.tensor([front * 1., 0., 0.], device=device)
    shoulder_ax = torch.tensor([0., left * 1., 0.], device=device)
    knee_ax = torch.tensor([0., left * 1., 0.], device=device)

    # create the screw axes
    hip_screw_b = screw_axis(hip_ax, foot_to_hip_zero_ht[:3, 3])
    shoulder_screw_b = screw_axis(shoulder_ax, foot_to_shoulder_zero_ht[:3, 3])
    knee_screw_b = screw_axis(knee_ax, foot_to_knee_zero_ht[:3, 3])

    hip_screwmat_b = screwvec_to_mat(hip_screw_b)
    shoulder_screwmat_b = screwvec_to_mat(shoulder_screw_b)
    knee_screwmat_b = screwvec_to_mat(knee_screw_b)

    hip_ht_b = screwmat_to_ht_batch(hip_screwmat_b, hip_angles)
    shoulder_ht_b = screwmat_to_ht_batch(shoulder_screwmat_b, shoulder_angles)
    knee_ht_b = screwmat_to_ht_batch(knee_screwmat_b, knee_angles)

    foot_pos = torso_to_foot_zero_ht @ hip_ht_b @ shoulder_ht_b @ knee_ht_b

    minus_hip_ht_b = screwvec_to_ht_batch(-hip_screw_b, hip_angles)
    minus_shoulder_ht_b = screwvec_to_ht_batch(
        -shoulder_screw_b, shoulder_angles)
    minus_knee_ht_b = screwvec_to_ht_batch(-knee_screw_b, knee_angles)
    minus_foot_ht_b = screwmat_to_ht_batch(-torch.eye(4, device=device),
                                           torch.zeros(NUM_ENVS, device=device))

    j_knee_b = knee_screw_b.expand(NUM_ENVS, 6)
    j_shoulder_b = ht_adj_batch(minus_knee_ht_b) @ shoulder_screw_b
    j_hip_b = ht_adj_batch(minus_knee_ht_b @ minus_shoulder_ht_b) @ hip_screw_b

    torso_to_foot_rot = foot_pos[:, :3, :3]
    torso_to_foot_rot_adj = torch.zeros((NUM_ENVS, 6, 6), device=device)
    torso_to_foot_rot_adj[:, :3, :3] = torso_to_foot_rot
    torso_to_foot_rot_adj[:, 3:, 3:] = torso_to_foot_rot

    j_knee = torch.einsum('Bij,Bj->Bi', torso_to_foot_rot_adj, j_knee_b)
    j_shoulder = torch.einsum(
        'Bij,Bj->Bi', torso_to_foot_rot_adj, j_shoulder_b)
    j_hip = torch.einsum('Bij,Bj->Bi', torso_to_foot_rot_adj, j_hip_b)

    j = torch.stack((j_hip, j_shoulder, j_knee), dim=-1)

    return j, foot_pos[:, :3, 3]


def build_jacobian_and_fk_all_feet(q_vec):
    """This function builds the jacobian and fk for all 4 of the feet

    Args:
        q_vec (torch.Tensor): a (NUM_ENVS, 12) tensor of joint angles for each of the robots

    Returns:
        torch.Tensor: a (NUM_ENVS x 4 x 3) tensor representing the positions of each of the 4 feet of each of the robots
    """
    FL_J, FL_r = build_jacobian_and_fk(q_vec, 0)
    FR_J, FR_r = build_jacobian_and_fk(q_vec, 1)
    BL_J, BL_r = build_jacobian_and_fk(q_vec, 2)
    BR_J, BR_r = build_jacobian_and_fk(q_vec, 3)

    all_jacobs = torch.stack((FL_J, FR_J, BL_J, BR_J), dim=1)
    all_positions = torch.stack((FL_r, FR_r, BL_r, BR_r), dim=1)

    return all_jacobs, all_positions


def use_ik(robot: Robot, q_vec, goals, ik_alg):
    """This function unpacks the stacked up vectors into the feet for the ik function, and restacks them at the end.

    Args:
        robot (Robot): the robot config for kinematics
        q_vec (torch.Tensor): a (NUM_ENVS x 12) vector which contains the joint positions of the robots joints.
        goals (torch.Tensor): a (NUM_ENVS x 12) vector which contains the goal (x,y,z) positions for the feet relative to their home position all stacked up in one vector.
        ik_alk ((torch.tensor, torch.tensor) -> torch.tensor)

    Returns:
        torch.Tensor: returns a (NUM_ENVS x 12) vector which contains the new targets for each of the robot's joints
    """

    NUM_ENVS = q_vec.shape[0]

    goals_per_foot = goals.reshape(-1, 4, 3)

    # Here we find the foot errors and the jacobian

    jacobian, absolute_foot_pos = build_jacobian_and_fk_all_feet(q_vec)
    _, home_pos = build_jacobian_and_fk_all_feet(
        robot.home_position.reshape((1, -1)))
    relative_foot_pos = absolute_foot_pos - home_pos
    error = goals_per_foot - relative_foot_pos

    per_foot_errors = ik_alg(
        jacobian[:, :, 3:].reshape(-1, 3, 3), error.reshape(-1, 3))

    # overwriting the nan values with 0
    torch.nan_to_num(per_foot_errors, out=per_foot_errors)

    per_env_errors = per_foot_errors.reshape((-1, 12))

    q_vec_mod = (q_vec+torch.pi).remainder(2*torch.pi) - torch.pi

    # Enforce joint polarity
    wrong_direction = (q_vec_mod * robot.joint_polarity) < 0.0
    corrections = (-2 * q_vec_mod[wrong_direction]).clamp(min=-0.25, max=0.25)

    targets = q_vec_mod
    targets[:] = q_vec.clone()

    targets[wrong_direction] += corrections
    targets[~wrong_direction] += per_env_errors[~wrong_direction]

    return targets


def mini_cheetah_dls_invkin(q_vec, goals):
    """Simply executes the dls_invkin algorithm on the mini_cheetah robot

    Args:
        q_vec (torch.Tensor): a (NUM_ENVS x 12) vector which contains the joint positions of the robots joints.
        goals (torch.Tensor): a (NUM_ENVS x 12) vector which contains the goal (x,y,z) positions for the feet relative to their home position all stacked up in one vector.

    Returns:
        torch.Tensor: returns a (NUM_ENVS x 12) vector which contains the new targets for each of the robot's joints
    """

    from legup.robots.mini_cheetah.mini_cheetah import MiniCheetah

    return use_ik(MiniCheetah, q_vec, goals, dls_invkin)


def mini_cheetah_pinv_invkin(q_vec, goals):
    return use_ik(q_vec, goals, pinv_invkin)
