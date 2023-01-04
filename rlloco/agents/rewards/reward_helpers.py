import torch


def squared_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """Calculates the squared norm of a tensor

    Args:
        x (torch.Tensor): Arbitrarily shaped tensor
        dim (int, optional): Dimension to calculate the norm over. Defaults to -1.

    Returns:
        torch.Tensor: The squared norm of the tensor across the given dimension
    """
    return torch.sum(torch.pow(x, 2), dim=dim)
