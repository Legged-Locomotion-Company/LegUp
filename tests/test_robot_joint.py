import pytest

import torch

# from legup.common.spatial import Transform, Direction
# from legup.common.robot import RobotJoint, RobotLink


# def test_revolute_joint_instantiates():
#     """Test the creation of a revolute robot joint."""

#     origin = Transform(torch.eye(4))
#     axis = Direction(torch.tensor([0, 0, 1]))
#     child_link = RobotLink(name="test_link")

#     joint = RobotJoint.make_revolute(name="test_joint",
#                                      origin=origin,
#                                      axis=axis,
#                                      child_link=child_link,
#                                      device=child_link.device)

#     assert joint is not None


# def test_revolute_joint_applies():
#     """Test the apply function of a revolute joint"""

#     origin = Transform(torch.eye(4))
#     axis = Direction(torch.tensor([0, 0, 1]))
#     child_link = RobotLink(name="test_link")

#     joint = RobotJoint.make_revolute(name="test_joint",
#                                      origin=origin,
#                                      axis=axis,
#                                      child_link=child_link,
#                                      device=child_link.device)

#     test_thetas = torch.rand(9, 8, 7, 6, 5, device=joint.device)

#     result = joint.apply(test_thetas)

#     expected_result = torch.eye(
#         4, device=joint.device).repeat(9, 8, 7, 6, 5, 1, 1)

#     expected_result[..., 0, 0] = torch.cos(test_thetas)
#     expected_result[..., 0, 1] = -torch.sin(test_thetas)
#     expected_result[..., 1, 0] = torch.sin(test_thetas)
#     expected_result[..., 1, 1] = torch.cos(test_thetas)

#     assert result.tensor.allclose(expected_result)
