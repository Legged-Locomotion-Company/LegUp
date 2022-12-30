import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dls_invkin(j: torch.Tensor, e: torch.Tensor, lam: torch.float = 0.05):
    """Executes Dampened Least Squares algorithm on inputs

    Args:
        j (torch.Tensor): a (NUM_ENVS x 4 x 3 x 3) tensor with the jacobian for each of the 4 feet.
        e (torch.Tensor): a (NUM_ENVS x 4 x 3) tensor with the error for each of the 4 feet.
        lam (torch.float): a scalar weight on the dampening for the optimization. The higher the lambda the slower it will be, but the more stable.

    Returns:
        torch.Tensor: a (NUM_ENVS x 4 x 3) tensor with the command delta to be applied to each foot
    """
    j_T = torch.transpose(j, -1, -2)
    lam_eye = torch.eye(3).to(device) * (lam ** 2)

    jj_T_lameye = j @ j_T + lam_eye

    inv_jj_T_lameye_e = torch.linalg.solve(jj_T_lameye, e)

    u = torch.einsum('BFij,BFj->BFi', j_T, inv_jj_T_lameye_e)
    return u