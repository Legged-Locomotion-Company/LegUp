import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dls_invkin(j: torch.Tensor, e: torch.Tensor, lam: torch.float = 0.0001):
    """Executes Dampened Least Squares algorithm on inputs

    Args:
        j (torch.Tensor): a (NUM_ENVS x 4 x 3 x 3) tensor with the jacobian for each of the 4 feet.
        e (torch.Tensor): a (NUM_ENVS x 4 x 3) tensor with the error for each of the 4 feet.
        lam (torch.float): a scalar weight on the dampening for the optimization. The higher the lambda the slower it will be, but the more stable.

    Returns:
        torch.Tensor: a (NUM_ENVS x 4 x 3) tensor with the command delta to be applied to each foot
    """

    j_T = j.transpose( -1, -2)
    lam_eye = torch.eye(3, device=device) * (lam ** 2)

    jj_T_lameye = j @ j_T + lam_eye

    # TODO: replace this with a more stable version using torch.linalg.solve but that's too much brainpower rn lol
    jj_T_lameye_inv = torch.linalg.pinv(jj_T_lameye)

    u = torch.einsum('BFij,BFjk,BFk->BFi', j_T, jj_T_lameye_inv, e)
    return u

def pinv_invkin(j: torch.Tensor, e:torch.Tensor, multiplier=0.25):
    """Executes the pseudo inverse algorithm on inputs (Do not use this alg it sucks, just here for testing)
    Args:
        j (torch.Tensor): a (NUM_ENVS x 4 x 3 x 3) tensor with the jacobian for each of the 4 feet.
        e (torch.Tensor): a (NUM_ENVS x 4 x 3) tensor with the error for each of the 4 feet.
        multiplier (float, optional): a scalar weight on the result so that it doesn't overshoot as much

    Returns:
        torch.Tensor: a (NUM_ENVS x 4 x 3) tensor with the command delta to be applied to each foot
    """
    return torch.linalg.solve(j,e) * multiplier