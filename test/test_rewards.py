from legup.train.rewards.rewards import *

from colorama import Fore, Back, Style
import pytest

import torch

# https://realpython.com/pytest-python-testing/


@pytest.mark.parametrize("num_envs", [2, 10, 1000])
def test_confirm_reward_velocity_dimensions(num_envs):
    """
    Confirm that the reward dimensions are correct.
    """
    lin_v_shape = lin_velocity(torch.rand([num_envs, 2]), torch.rand(
        [num_envs, 2])).shape
    assert lin_v_shape == torch.Size(
        [num_envs, ]), "lin_velocity should be shape ({num_envs},), but is {lin_v_shape}".format(num_envs=num_envs, lin_v_shape=lin_v_shape)

    ang_v_shape = ang_velocity(torch.rand(
        [num_envs, 1]), torch.rand([num_envs, 1])).shape
    assert ang_v_shape == torch.Size(
        [num_envs, ]), "ang_velocity should be shape ({num_envs},), but is {ang_v_shape}".format(num_envs=num_envs, ang_v_shape=ang_v_shape)

    lin_ortho_vel_shape = linear_ortho_velocity(torch.rand([num_envs, 3]), torch.rand(
        [num_envs, 3])).shape
    assert lin_ortho_vel_shape == torch.Size(
        [num_envs, ]), "lin_orthogonal_velocity should be shape ({num_envs},), but is {lin_ortho_vel_shape}".format(num_envs=num_envs, lin_ortho_vel_shape=lin_ortho_vel_shape)


@pytest.mark.parametrize("num_envs", [1, 2, 10, 1000])
def test_confirm_reward_joint_dimensions(num_envs):
    """
    Confirm that the reward dimensions are correct.
    """
    joint_motion_shape = joint_motion(torch.rand([num_envs, 1]), torch.rand(
        [num_envs, 1]), 0.1, 0.1).shape
    assert joint_motion_shape == torch.Size([num_envs, ]), "joint_motion should be shape ({num_envs},), but is {joint_motion_shape}".format(
        num_envs=num_envs, joint_motion_shape=joint_motion_shape)

    joint_constraints_shape = joint_constraint(torch.rand([num_envs, 1]), torch.rand(
        [num_envs, 1])).shape
    assert joint_constraints_shape == torch.Size([num_envs, ]), "joint_constraints should be shape ({num_envs},), but is {joint_constraints_shape}".format(
        num_envs=num_envs, joint_constraints_shape=joint_constraints_shape)


@pytest.mark.parametrize("num_envs", [1, 2, 10, 1000])
def test_confirm_reward_foot_dimensions(num_envs):
    """
    Confirm that the reward dimensions are correct.
    """
    slip_shape = slip(torch.rand([num_envs, 1]), torch.rand(
        [num_envs, 1, 3]), 0.1).shape
    assert slip_shape == torch.Size([num_envs, ]), "slip_reward should be shape ({num_envs},), but is {slip_shape}".format(
        num_envs=num_envs, slip_shape=slip_shape)

    foot_clearance_shape = foot_clearance(
        torch.rand([num_envs, 4, 10])).shape
    assert foot_clearance_shape == torch.Size(
        [num_envs, ]), "foot_clearance should be shape ({num_envs},), but is {foot_clearance_shape}".format(num_envs=num_envs, foot_clearance_shape=foot_clearance_shape)


@pytest.mark.parametrize("num_envs", [1, 2, 10, 1000])
def test_confirm_reward_movement_dimensions(num_envs):
    """
    Confirm that the reward dimensions are correct.
    """

    body_motion_shape = body_motion(torch.rand([num_envs, 1]), torch.rand(
        [num_envs, 1]), torch.rand([num_envs, 1])).shape
    assert body_motion_shape == torch.Size([num_envs, ]), "body_motion should be shape ({num_envs},), but is {body_motion_shape}".format(
        num_envs=num_envs, body_motion_shape=body_motion_shape)

    target_smoothness_shape = target_smoothness(torch.rand([num_envs, 1]), torch.rand([num_envs, 1]), torch.rand(
        [num_envs, 1]), 0.1).shape
    assert target_smoothness_shape == torch.Size([num_envs, ]), "target_smoothness should be shape ({num_envs},), but is {target_smoothness_shape}".format(
        num_envs=num_envs, target_smoothness_shape=target_smoothness_shape)

    torque_reward_shape = torque_reward(
        torch.rand([num_envs, 1]), 0.1).shape
    assert torque_reward_shape == torch.Size(
        [num_envs, ]), "torque_reward should be shape ({num_envs},), but is {torque_reward_shape}".format(num_envs=num_envs, torque_reward_shape=torque_reward_shape)


@pytest.mark.parametrize("num_envs", [1, 2, 10, 1000])
def test_confirm_reward_collision_dimensions(num_envs):
    """
    Confirm that the reward dimensions are correct.
    """
    shank_or_knee_col_shape = shank_or_knee_col(
        torch.rand([num_envs, 2]), 0.1).shape
    assert shank_or_knee_col_shape == torch.Size(
        [num_envs, ]), "shank_or_knee_col should be shape ({num_envs},), but is {shank_or_knee_col_shape}".format(num_envs=num_envs, shank_or_knee_col_shape=shank_or_knee_col_shape)


@pytest.mark.parametrize("num_envs", [1, 2, 10, 1000])
def test_clip_reward_dimensions(num_envs):
    """
    Confirm that the reward dimensions are correct.
    """
    clip_shape = clip_reward(torch.rand([num_envs, 12]), 0.1).shape
    assert clip_shape == torch.Size(
        [num_envs, ]), "clip_reward should be shape ({num_envs},), but is {clip_shape}".format(num_envs=num_envs, clip_shape=clip_shape)
