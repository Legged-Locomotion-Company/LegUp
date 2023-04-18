from legup.common.spatial.spatial import Transform, Screw, Twist

import pytest

import torch
import numpy as np
from scipy.linalg import expm, logm


def test_transform_init_batched():
    """Test the creation of a transform."""
    transform_tensor = torch.empty(100, 20, 4, 9, 4, 4)
    transform_tensor[:] = torch.eye(4)
    transform_tensor_copy = transform_tensor.clone()

    transform = Transform(transform_tensor)

    assert transform is not None
    assert transform.tensor.allclose(transform_tensor_copy)


def test_transform_init_single():
    """Test the creation of a transform."""
    transform_tensor = torch.empty(4, 4)
    transform_tensor[:] = torch.eye(4)
    transform_tensor_copy = transform_tensor.clone()

    transform = Transform(transform_tensor)

    assert transform is not None
    assert transform.tensor.allclose(transform_tensor_copy)


def test_transform_init_invalid():
    """Test the creation of a transform."""
    transform_tensor = torch.rand(4, 4, 5)
    transform_tensor_copy = transform_tensor.clone()

    with pytest.raises(ValueError):
        transform = Transform(transform_tensor)


def test_twist_to_rotation_batched():
    """Test the conversion of a twist tensor to a rotation tensor"""

    # create a random set of twists
    twists = Twist.rand(200, 100, 5, 4)

    # get the skews of those twists
    skews = twists.skew()

    expected_result = torch.matrix_exp(skews.tensor)

    result = twists.exp_map()

    assert torch.allclose(expected_result, result.tensor, atol=1e-4)


def test_screw_to_transform_batched():
    """Test the creation of a transform."""

    screw = Screw.rand(9, 8, 7)

    transform = screw.exp_map()

    skew = screw.skew()

    correct_result = torch.matrix_exp(skew.tensor)

    assert torch.allclose(correct_result, transform.tensor, atol=1e-4)
