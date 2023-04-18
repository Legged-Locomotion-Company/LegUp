from legup.common.tensor_types import TensorWrapper

import torch


def test_tensor_wrapper_class_instantiation_shaped():
    """Test that the TensorWrapper class can be instantiated."""

    class WrappedTensorType(TensorWrapper):
        def __init__(self, tensor):
            self.initialize_base(tensor, [3, 3])

    tensor_to_wrap = torch.rand(9, 10, 3, 3)
    tensor_wrapper = WrappedTensorType(tensor_to_wrap)

    assert tensor_wrapper.pre_shape() == [9, 10]


def test_tensor_wrapper_class_instantiation_single():
    """Test that the TensorWrapper class can be instantiated."""

    class WrappedTensorType(TensorWrapper):
        def __init__(self, tensor):
            self.initialize_base(tensor, [3, 3])

    tensor_to_wrap = torch.rand(3, 3)
    tensor_wrapper = WrappedTensorType(tensor_to_wrap)

    assert tensor_wrapper.pre_shape() == []


def test_tensor_wrapper_class_broadcast():
    """Test that the TensorWrapper class can be instantiated."""

    class WrappedTensorType(TensorWrapper):
        def __init__(self, tensor):
            self.initialize_base(tensor, [3, 3])

    a = torch.rand(1, 5, 3, 3)
    a_wrapper = WrappedTensorType(a)

    b = torch.rand(1, 3, 3)
    b_wrapper = WrappedTensorType(b)

    broadcast_pre_shape = TensorWrapper.get_broadcast_pre_shape(
        (a_wrapper, b_wrapper))

    a_broadcast = a_wrapper.unsqueeze_to_broadcast(broadcast_pre_shape)
    b_broadcast = b_wrapper.unsqueeze_to_broadcast(broadcast_pre_shape)

    assert True
