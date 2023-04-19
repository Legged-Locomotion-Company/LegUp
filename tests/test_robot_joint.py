import pytest

import torch

from legup.common.spatial import Transform, Direction
from legup.common.kinematics import Link, Joint


def test_revolute_joint_instantiates():
    """Test the creation of a revolute robot joint."""

    origin = Transform(torch.eye(4))
    axis = Direction(torch.tensor([0, 0, 1]))
    child_link = Link(name="test_link")

    joint = Joint.make_revolute(name="test_joint",
                                     origin=origin,
                                     axis=axis,
                                     child_link=child_link,
                                     device=child_link.device)

    assert joint is not None
