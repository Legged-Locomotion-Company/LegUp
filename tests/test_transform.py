from legup.common.robot import Transform

import pytest

import torch

@pytest.mark.transform_initialization
def test_transform_init_batched():
    """Test the creation of a transform."""
    transform_tensor = torch.empty(100, 20, 4, 9, 4, 4)
    transform_tensor[:] = torch.eye(4)
    transform_tensor_copy = transform_tensor.clone()

    transform = Transform(transform_tensor)

    assert transform is not None
    assert transform.tensor.allclose(transform_tensor_copy)

@pytest.mark.transform_initialization
def test_transform_init_single():
    """Test the creation of a transform."""
    transform_tensor = torch.empty(4, 4)
    transform_tensor[:] = torch.eye(4)
    transform_tensor_copy = transform_tensor.clone()

    transform = Transform(transform_tensor)

    assert transform is not None
    assert transform.tensor.allclose(transform_tensor_copy)
