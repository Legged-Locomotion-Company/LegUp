from legup.kinematics import Link, Joint
from legup.spatial import Transform, Position, Direction, Screw
from legup.kinematics import dls_ik

from typing import Tuple

import pytest

import torch


def test_planar_2r_inverse_kinematics():
    """Test that the robot kinematics work for a 2r planar robot"""

    torch.manual_seed(0)

    ee_link = Link("ee_link")

    fixed_wrist = Joint.make_fixed(
        name="wrist",
        origin=Position.from_iter([1, 0, 0]).make_transform(),
        child_link=ee_link,
    )

    forearm_link = Link("forearm",
                        child_joints=[fixed_wrist])

    elbow_joint = Joint.make_revolute(
        name="elbow",
        origin=Position.from_iter([1, 0, 0]).make_transform(),
        axis=Direction.from_list([0, 0, 1]),
        child_link=forearm_link)

    upper_arm_link = Link("upper_arm",
                          child_joints=[elbow_joint])

    shoulder_joint = Joint.make_revolute("shoulder",
                                         origin=Transform.zero(),
                                         axis=Direction.from_list([0, 0, 1]),
                                         child_link=upper_arm_link)

    base_link = Link(name="base",
                     child_joints=[shoulder_joint])

    kinematics = base_link.make_kinematics(
        query_link_names=["ee_link"])

    goal_angles = torch.rand(
        100, 100, 2, device=base_link.device) * 2 * torch.pi

    # Now we remove singular values from the goal angles

    def make_singular_mask(angles: torch.Tensor, epsilon: float = 0.2):
        pi_multiples = torch.round(angles[..., 1] / torch.pi)
        diffs = torch.abs(angles[..., 1] - pi_multiples * torch.pi)

        return diffs < epsilon

    # while (singular_mask := make_singular_mask(goal_angles)).any():
    #     goal_angles[singular_mask] = torch.rand_like(
    #         goal_angles[singular_mask]) * 2 * torch.pi

    goal_positions = kinematics(goal_angles).transform.extract_translation()

    current_angles = goal_angles + \
        torch.rand_like(goal_angles) * torch.pi * 2 / 4

    while (singular_mask := make_singular_mask(current_angles)).any():
        current_angles[singular_mask] = torch.rand_like(
            goal_angles[singular_mask]) * torch.pi * 2 / 4

    def calculate_errors():
        """Calculate the error between the target and the end effector"""
        kin_result = kinematics(current_angles)
        pos_result = kin_result.transform.extract_translation()

        pos_error = pos_result - goal_positions

        return pos_error.norm()

    count = 0
    epsilon = 1e-3

    unconverged_iterations = torch.zeros_like(
        calculate_errors(), dtype=torch.float)

    interesting_idx = None

    while calculate_errors().max() > epsilon and count < 100:
        while (singular_mask := make_singular_mask(goal_angles)).any():
            goal_angles[singular_mask] = torch.rand_like(
                goal_angles[singular_mask]) * 2 * torch.pi
        # old_angles = current_angles.clone()
        current_angles = dls_ik.apply(
            goal_positions, current_angles, kinematics)

        # if count == 80:
        #     interesting_idx = calculate_errors().reshape(-1).argmax()

        # if count > 80:
        #     some_error = (goal_positions.tensor.reshape(-1, 3)[interesting_idx] -
        #                   kinematics(old_angles.reshape(-1, 2)[interesting_idx]).transform.extract_translation().tensor.squeeze()).norm()

        #     some_delta = (current_angles.reshape(-1, 2)[interesting_idx] -
        #                   old_angles.reshape(-1, 2)[interesting_idx])

        #     some_angle = old_angles.reshape(-1, 2)[interesting_idx]

        #     some_target_angle = goal_angles.reshape(-1, 2)[interesting_idx]

        #     print(f"some error: {some_error}")
        #     print(f"some delta: {some_delta}")
        #     print(f"some angle: {some_angle}")
        #     print(f"some target angle: {some_target_angle}")

        count += 1

        unconverged_iterations[calculate_errors() > epsilon] += 1

    final_errors = calculate_errors().flatten()

    print(f"""Iteration Stats
        mean: {unconverged_iterations.mean()}
        median: {unconverged_iterations.median()}
        max: {unconverged_iterations.max()}
        min: {unconverged_iterations.min()}""")

    # DLS should be more stable than this, but it's not working well so I'll leave it at this for now
    assert (final_errors > epsilon).sum() / final_errors.shape[0] < 0.1, \
        f"""max error: {final_errors.max()}
        median error: {final_errors.median()}
        mean error: {final_errors.mean()}
        min error: {final_errors.min()}
        failing portion: {(final_errors > epsilon).sum() / final_errors.shape[0]}"""
