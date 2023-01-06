import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def invert_ht_batch(ht):
    """Inverts a 3d 4x4 homogenous transform. Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.

    Args:
        ht (torch.Tensor): A (NUM_ENVS x 4 x 4) stack of homogenous transforms

    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) stack of inverted homogenous transforms. Will be equal to torch.invert(ht) for each ht
    """

    NUM_ENVS, _, _ = ht.shape

    inverted = torch.eye(4, device=device).repeat(NUM_ENVS, 1, 1)

    # TODO This sucks make it better
    inverted[:, :3, :3] = ht[:, :3, :3].transpose(1, 2)
    inverted[:, :3, 3] = - \
        torch.einsum('Bij,Bj->Bi', inverted[:, :3, :3], ht[:, :3, 3])

    return inverted


def invert_ht(ht):
    """Inverts a 3d 4x4 homogenous transform. Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.

    Args:
        ht (torch.Tensor): The homogenous transform

    Returns:
        torch.Tensor: returns the inverted homogenous transform. Will be equal to torch.invert(ht)
    """

    inverted = torch.eye(4, device=device)

    # TODO This sucks make it better
    inverted[:3, :3] = ht.T[:3, :3]
    inverted[:3, 3] = - inverted[:3, :3] @ ht[:3, 3]

    return inverted


def xrot(theta):
    sth = torch.sin(theta)
    cth = torch.cos(theta)

    result = torch.eye(4, device=device).repeat(len(theta), 1, 1)

    result[:, 1, 1] = cth
    result[:, 1, 2] = -sth
    result[:, 2, 1] = sth
    result[:, 2, 2] = cth

    return result


def yrot(theta):
    sth = torch.sin(theta)
    cth = torch.cos(theta)

    result = torch.eye(4, device=device).repeat(len(theta), 1, 1)

    result[:, 0, 0] = cth
    result[:, 0, 2] = sth
    result[:, 2, 0] = -sth
    result[:, 2, 2] = cth

    return result


def translation_ht(x, y, z, NUM_ENVS=None):
    # if NUM_ENVS == None:
    #     NUM_ENVS, = len(x)
    # pos = torch.tensor([x, y, z], device= device)

    ht = torch.eye(4, device=device)
    ht[0, 3] = x
    ht[1, 3] = y
    ht[2, 3] = z
    return ht


def skew_symmetric(omega):
    """Converts a 3d vector to a skew symmetric matrix

    Args:
        omega (torch.Tensor): A (3) element vector representing a 3d vector

    Returns:
        torch.Tensor: A (3 x 3) tensor representing a skew symmteric matrix
    """
    # NUM_ENVS, _ = omega.shape

    result = torch.zeros((3, 3), device=device)

    result[0, 1] = -omega[2]
    result[0, 2] = omega[1]
    result[1, 0] = omega[2]
    result[1, 2] = -omega[0]
    result[2, 0] = -omega[1]
    result[2, 1] = omega[0]

    return result


def skew_symmetric_batch(omegas):
    """Converts a 3d vector to a skew symmetric matrix

    Args:
        omega (torch.Tensor): A (NUM_ENVS x 3) element vector representing a stack of 3d vectors

    Returns:
        torch.Tensor: A (NUM_ENVS x 3 x 3) tensor representing a stack of skew symmetric matrices
    """
    NUM_ENVS, _ = omegas.shape

    result = torch.zeros((NUM_ENVS, 3, 3), device=device)

    result[:, 0, 1] = -omegas[:, 2]
    result[:, 0, 2] = omegas[:, 1]
    result[:, 1, 0] = omegas[:, 2]
    result[:, 1, 2] = -omegas[:, 0]
    result[:, 2, 0] = -omegas[:, 1]
    result[:, 2, 1] = omegas[:, 0]

    return result


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


def screwvec_to_mat(screws):
    """Converts a screw axis to the matrix form
    Page 104 of MR

    Args:
        screws (torch.Tensor): A (NUM_ENVS x 6) element vector representing a stack of screw axes
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles

    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of screw matrices
    """

    # NUM_ENVS, _ = screws.shape

    twist = screws[:3]
    v = screws[3:]

    result = torch.zeros((4, 4), device=device)
    result[:3, :3] = skew_symmetric(twist)
    result[:3, 3] = v

    return result


def ht_adj_batch(T):
    """Computes the adjoint of a homogenous transform

    Args:
        T (torch.Tensor): A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms

    Returns:
        torch.Tensor: A (NUM_ENVS x 6 x 6) tensor representing a stack of adjoints
    """

    NUM_ENVS, _, _ = T.shape

    result = torch.zeros((NUM_ENVS, 6, 6), device=device)

    result[:, :3, :3] = T[:, :3, :3]
    result[:, :3, 3:] = skew_symmetric_batch(T[:, :3, 3]) @ T[:, :3, :3]
    result[:, 3:, 3:] = T[:, :3, :3]

    return result


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


def screwmat_to_ht(screwmats, thetas):
    """Converts a screw axis matrix and magnitude to a homogenous transform

    Args:
        screwmats (torch.Tensor): A (4 x 4) tensor representing a screw matrix
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles

    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
    """

    NUM_ENVS = len(thetas)

    result = torch.zeros((NUM_ENVS, 4, 4), device=device)

    twist_skew = screwmats[:3, :3]
    v = screwmats[:3, 3]

    twist = unskew(twist_skew)

    result = torch.eye(4, device=device).repeat(NUM_ENVS, 1, 1)

    if (not twist.norm().allclose(torch.tensor(0.0, device=device))):
        result[:, :3, :3] = twist_skew_to_rot(twist_skew, thetas)

        # TODO: Fix this garbage
        term1 = torch.eye(3, device=device).expand(NUM_ENVS, 3, 3) * thetas.repeat(3,3,1).transpose(-1,0).expand(NUM_ENVS,3,3)
        term2 = (1 - torch.cos(thetas)).repeat(3,3,1).transpose(-1,0).expand(NUM_ENVS,3,3) * twist_skew.expand(NUM_ENVS, 3, 3)
        term3 = (thetas - torch.sin(thetas)).repeat(3,3,1).transpose(-1,0).expand(NUM_ENVS,3,3) * \
            (twist_skew @ twist_skew).expand(NUM_ENVS, 3, 3)

        result[:, :3, 3] = torch.einsum(
            'Bij,j->Bi', (term1 + term2 + term3), v)

    else:
        # TODO: Fix this garbage
        result[:, :3, 3] = v.expand(NUM_ENVS,3) * thetas.repeat(3,1).transpose(-1,0).expand(NUM_ENVS,3)

    return result

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


def screw_to_ht(screws, thetas):
    """Converts a screw axis to a homogenous transform
    This is equivalent to the matrix exponential of the skew symmetric matrix of the screw axis times the theta

    Args:
        screws (torch.Tensor): A (NUM_ENVS x 6) element vector representing a stack of screw axes
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles

    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
    """

    screwmats = screwvec_to_mat(screws)

    return screwmat_to_ht(screwmats, thetas)


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
    term2 = torch.sin(thetas).repeat(3,3,1).transpose(-1,0).expand(NUM_ENVS,3,3) * twist_skew.expand(NUM_ENVS, 3, 3)
    term3 = (1 - torch.cos(thetas).repeat(3,3,1).transpose(-1,0).expand(NUM_ENVS,3,3)) * \
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
