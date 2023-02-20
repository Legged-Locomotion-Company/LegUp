from legup.common.tensor_wrapper import TensorWrapper

import torch


def test_tensor_wrapper_class_instantiation_shaped():
    """Test that the TensorWrapper class can be instantiated."""

    class WrappedTensorType(TensorWrapper):
        def __init__(self, tensor):
            super().__init__(tensor, [3, 3])

    tensor_to_wrap = torch.rand(9, 10, 3, 3)
    tensor_wrapper = WrappedTensorType(tensor_to_wrap)

    assert tensor_wrapper.pre_shape() == [9, 10]


def test_tensor_wrapper_class_instantiation_single():
    """Test that the TensorWrapper class can be instantiated."""

    class WrappedTensorType(TensorWrapper):
        def __init__(self, tensor):
            super().__init__(tensor, [3, 3])

    tensor_to_wrap = torch.rand(3, 3)
    tensor_wrapper = WrappedTensorType(tensor_to_wrap)

    assert tensor_wrapper.pre_shape() == []
