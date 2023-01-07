import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def invert_ht_batch(ht, out=None):
    """Inverts a 3d 4x4 homogenous transform. Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.
    Args:
        ht (torch.Tensor): A (NUM_ENVS x 4 x 4) stack of homogenous transforms
        out (torch.Tensor, optional): A (NUM_ENVS x 4 x 4) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) stack of inverted homogenous transforms. Will be equal to torch.invert(ht) for each ht
    """

    NUM_ENVS = ht.shape[0]

    if out is None:
        out = torch.eye(4, device=device).repeat(NUM_ENVS, 1, 1)

    elif out.shape != (NUM_ENVS, 4, 4):
        raise ValueError("Out must be a (NUM_ENVS x 4 x 4) tensor.")

    inverted = torch.eye(4, device=device).repeat(NUM_ENVS, 1, 1)

    # TODO This sucks make it better
    inverted[:, :3, :3] = ht[:, :3, :3].transpose(1, 2)
    inverted[:, :3, 3] = - \
        torch.einsum('Bij,Bj->Bi', inverted[:, :3, :3], ht[:, :3, 3])

    return inverted


def invert_ht(ht):
    """Inverts a 3d 4x4 homogenous transform. Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.
    Args:
        ht (torch.Tensor): A (4x4) matrix representing the homogenous transform
    Returns:
        torch.Tensor: A (4 x 4) matrix representing the inverted homogenous transform. Will be equal to torch.invert(ht) for each ht
    """

    out = torch.eye(4, device=device)

    out[:3, :3] = out.T[:3, :3]
    out[:3, 3] = - out[:3, :3] @ ht[:3, 3]

    return out


def translation_ht_batch(vs, out=None):
    """Creates a batch of 3d homogenous transform that translates by a vector
    Args:
        vs (torch.Tensor): A (NUM_ENVS x 3) tensor representing the translation vector
        out (torch.Tensor, optional): A (NUM_ENVS x 4 x 4) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing the homogenous transform
    """
    NUM_ENVS, _ = vs.shape

    if out is None:
        out = torch.eye(4, device=device).repeat(NUM_ENVS, 1, 1)
    elif out.shape != (NUM_ENVS, 4, 4):
        raise ValueError("Out must be a (NUM_ENVS x 4 x 4) tensor.")
    else:
        out[:] = torch.eye(3, device=device)

    out[:, 0, 3] = vs[:, 0]
    out[:, 1, 3] = vs[:, 1]
    out[:, 2, 3] = vs[:, 2]

    return out


def translation_ht(v):
    """Creates a 3d homogenous transform that translates by a vector
    Args:
        v (torch.Tensor): A (3) tensor representing the translation vector
    Returns:
        torch.Tensor: A (4 x 4) tensor representing the homogenous transform
    """

    out = torch.eye(4, device=device)

    out[0, 3] = v[0]
    out[1, 3] = v[1]
    out[2, 3] = v[2]

    return out


def skew_symmetric(omega):
    """Converts a 3d vector to a skew symmetric matrix
    Args:
        omega (torch.Tensor): A (3) element vector representing a 3d vector
    Returns:
        torch.Tensor: A (3 x 3) tensor representing a skew symmteric matrix
    """

    result = torch.zeros((3, 3), device=device)

    result[0, 1] = -omega[2]
    result[0, 2] = omega[1]
    result[1, 0] = omega[2]
    result[1, 2] = -omega[0]
    result[2, 0] = -omega[1]
    result[2, 1] = omega[0]

    return result


def skew_symmetric_batch(omegas, out=None):
    """Converts a 3d vector to a skew symmetric matrix
    Args:
        omega (torch.Tensor): A (NUM_ENVS x 3) element vector representing a stack of 3d vectors
        out (torch.Tensor, optional): A (NUM_ENVS x 3 x 3) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 3 x 3) tensor representing a stack of skew symmetric matrices
    """
    NUM_ENVS = omegas.shape[0]

    if out is None:
        out = torch.zeros((NUM_ENVS, 3, 3), device=device)
    elif out.shape != (NUM_ENVS, 3, 3):
        raise ValueError("Out must be a (NUM_ENVS x 3 x 3) tensor.")
    else:
        out[:, :, :] = 0.0

    out[:, 0, 1] = -omegas[:, 2]
    out[:, 0, 2] = omegas[:, 1]
    out[:, 1, 0] = omegas[:, 2]
    out[:, 1, 2] = -omegas[:, 0]
    out[:, 2, 0] = -omegas[:, 1]
    out[:, 2, 1] = omegas[:, 0]

    return out


def unskew(skew):
    """Converts a skew symmetric matrix to a 3d vector
    Args:
        skew (torch.Tensor): A (3 x 3) tensor representing a skew symmetric matrix
    Returns:
        torch.Tensor: A (3) element vector
    """

    # NUM_ENVS, _, _ = skew.shape

    result = torch.zeros((3), device=device)

    result[0] = skew[2, 1]
    result[1] = skew[0, 2]
    result[2] = skew[1, 0]

    return result

def screwvec_to_mat(screw):
    """Converts a screw axis to the matrix form
    Page 104 of MR
    Args:
        screw (torch.Tensor): A (6) element vector representing a screw axis
    Returns:
        torch.Tensor: A (4 x 4) tensor representing a screw matrix
    """

    out = torch.zeros((4, 4), device=device)

    out[:3, :3] = skew_symmetric(screw[:3])
    out[:3, 3] = screw[3:]
    out[3, 3] = 0.0

    return out


def screwvec_to_mat_batch(screws, out=None):
    """Converts a screw axis to the matrix form WARNING UNTESTED TODO: Actually implement this
    Page 104 of MR
    Args:
        screws (torch.Tensor): A (NUM_ENVS x 6) element vector representing a stack of screw axes
        out (torch.Tensor, optional): A (NUM_ENVS x 4 x 4) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of screw matrices
    """

    NUM_ENVS = screws.shape[0]

    if out is None:
        out = torch.zeros((NUM_ENVS, 4, 4), device=device)
    elif out.shape != (NUM_ENVS, 4, 4):
        raise ValueError("Out must be a (NUM_ENVS x 4 x 4) tensor.")
    else:
        out[:, :, :] = 0.0

    twist = screws[:3]
    v = screws[3:]

    out[:3, :3] = skew_symmetric(twist)
    out[:3, 3] = v

    return out


def ht_adj_batch(T, out=None):
    """Computes the adjoint of a homogenous transform
    Args:
        T (torch.Tensor): A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
        out (torch.Tensor, optional): A (NUM_ENVS x 6 x 6) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 6 x 6) tensor representing a stack of adjoints
    """

    NUM_ENVS = T.shape[0]

    if out is None:
        out = torch.zeros((NUM_ENVS, 6, 6), device=device)
    elif out.shape != (NUM_ENVS, 6, 6):
        raise ValueError("Out must be a (NUM_ENVS x 6 x 6) tensor.")
    else:
        out[:, :, :] = 0.0

    out[:, :3, :3] = T[:, :3, :3]
    out[:, :3, 3:] = skew_symmetric_batch(T[:, :3, 3]) @ T[:, :3, :3]
    out[:, 3:, 3:] = T[:, :3, :3]

    return out


def ht_adj(T):
    """Computes the adjoint of a homogenous transform
    Args:
        T (torch.Tensor): A (4 x 4) tensor representing a homogenous transform
    Returns:
        torch.Tensor: A (6 x 6) adjoint representation of the matrix T
    """

    # NUM_ENVS, _, _ = T.shape

    result = torch.zeros((6, 6), device=device)

    result[:3, :3] = T[:3, :3]
    result[:3, 3:] = skew_symmetric(T[:3, 3]) @ T[:3, :3]
    result[3:, 3:] = T[:3, :3]

    return result


def screwmat_to_ht_batch(screwmat, thetas, out=None):
    """Converts a screw axis matrix and magnitude to a homogenous transform. Should be equivalent to expm(screwmat * thetas)
    Args:
        screwmat (torch.Tensor): A (4 x 4) tensor representing a screw matrix
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
    """

    NUM_ENVS = thetas.shape[0]

    if out is None:
        out = torch.zeros((NUM_ENVS, 4, 4), device=device)
    elif out.shape != (NUM_ENVS, 4, 4):
        raise ValueError("Out must be a (NUM_ENVS x 4 x 4) tensor.")
    else:
        out[:, :, :] = 0.0  # torch.eye(4, device=device)

    twist_skew = screwmat[:3, :3]
    v = screwmat[:3, 3]

    twist = unskew(twist_skew)

    if (not twist.norm().allclose(torch.tensor(0.0, device=device))):
        scratch = out[:, :3, :3]
        # Term 1
        scratch = torch.eye(3, device=device).expand(
            NUM_ENVS, 3, 3) * thetas.expand(3, 3, -1).tranpose(0, -1)

        # Term 2
        scratch += (1 - torch.cos(thetas)).expand(3, 3, -
                                                  1).transpose(0, -1) * twist_skew.expand(NUM_ENVS, 3, 3)

        # Term 3
        scratch += (thetas - torch.sin(thetas)).expand(3, 3, 1).transpose(-1, 0) * \
            (twist_skew @ twist_skew).expand(NUM_ENVS, 3, 3)

        torch.matmul(scratch, v.expand(NUM_ENVS, 3, 1), out=out[:, :3, 3])

        out[:, :3, :3] = twist_skew_to_rot(twist_skew, thetas)
        out[:, 3, 3] = 1.0

    else:
        # TODO: Fix this garbage
        out[:, :3, 3] = v.expand(
            NUM_ENVS, 3) * thetas.expand(3, -1).transpose(-1, 0)

    return out

    # rotation = twists.norm(dim=1) != 0
    # no_rotation = twists.norm(dim=1) == 0

    # twist_skew = skew_symmetric(twists)[rotation]

    # result = torch.eye(4, device=device).repeat(NUM_ENVS, 1, 1)
    # result[rotation, :3, :3] = twist_to_rot(twists, thetas)

    # term1 = torch.eye(3, device=device).repeat(NUM_ENVS, 1, 1)[rotation] * thetas[rotation]
    # term2 = (1 - torch.cos(thetas))[rotation] * twist_skews
    # term3 = (thetas - torch.sin(thetas))[rotation] * twist_skews @ twist_skews

    # result[rotation, :3, 3] =  torch.einsum('Bij,Bj->Bi', (term1 + term2 + term3), v)

    # if (torch.sum(no_rotation) > 0):
    #     result[no_rotation,:3,3] = v[no_rotation] * thetas[no_rotation]

    # return result


def screwvec_to_ht_batch(screws, thetas, out=None):
    """Converts a screw axis to a homogenous transform
    This is equivalent to the matrix exponential of the skew symmetric matrix of the screw axis times the theta
    Args:
        screws (torch.Tensor): A (NUM_ENVS x 6) element vector representing a stack of screw axes
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
    """

    screwmats = screwvec_to_mat_batch(screws)

    return screwmat_to_ht_batch(screwmats, thetas)


def twist_skew_to_rot(twist_skew, thetas):
    """Converts a twist skew matrix and angle to a rotation matrix
    Args:
        twist_skew (torch.Tensor): A (3 x 3) tensor representing a skew symmetric matrix
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
    Returns:
        torch.Tensor: A (NUM_ENVS x 3 x 3) tensor representing a stack of rotation matrices
    """

    NUM_ENVS = len(thetas)

    term1 = torch.eye(3, device=device)
    # this line is so bad omg
    term2 = torch.sin(thetas).repeat(3, 3, 1).transpose(-1,
                                                        0).expand(NUM_ENVS, 3, 3) * twist_skew.expand(NUM_ENVS, 3, 3)
    term3 = (1 - torch.cos(thetas).repeat(3, 3, 1).transpose(-1, 0).expand(NUM_ENVS, 3, 3)) * \
        (twist_skew @ twist_skew).expand(NUM_ENVS, 3, 3)

    return term1 + term2 + term3


def twist_to_rot(twists, thetas):
    """Converts a twist axis and angle to a rotation matrix
    Args:
        twists (torch.Tensor): A (NUM_ENVS x 3) element vector representing a stack of screw axes
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
    Returns:
        torch.Tensor: A (NUM_ENVS x 3 x 3) tensor representing a stack of rotation matrices
    """

    twist_skews = skew_symmetric(twists)

    return twist_skew_to_rot(twist_skews, thetas)


def screw_axis(axis, q):
    """Creates a screw axis from a rotation axis and a point along the axis
    Args:
        axis (torch.Tensor): A (3) element vector representing a rotation axis
        q (torch.Tensor): A (3) element vector representing a point along the axis
    Returns:
        torch.Tensor: A (6) vector representing a screw axis
    """

    result = torch.zeros(6, device=device)
    result[:3] = axis
    result[3:] = torch.cross(-axis, q)

    return result


def rot_to_twist(rot):
    """Converts a rotation matrix to a twist axis
    Args:
        rot (torch.Tensor): A (NUM_ENVS x 3 x 3) element tensor representing a stack of rotation matrices
    Returns:
        torch.Tensor: A (NUM_ENVS x 3) tensor representing a stack of twist axes
    """

    # NOT SURE IF I NEED THIS, LEVING IT HERE IN CASE. NOT DOING IT CUZ IT'S ANNOYING

    # NUM_ENVS, _, _ = rot.shape

    # # batch trace
    # trace = torch.einsum('Bii->B', rot)
    # theta = torch.acos((trace - 1) / 2)

    # omega_hat_skew = 1/ (2 * torch.sin(theta)) * (rot - rot.transpose(1, 2))
    # omega_hat = unskew(omega_hat_skew)

    # eye_mask = rot == torch.eye(3).expand(NUM_ENVS, 3, 3)

    # return theta * omega_hat
