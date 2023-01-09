import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
This function contains basic math functions for the general use in kinematics in vectorized forms
Functions implemented are 

invert_ht_batch
invert_ht

translation_ht_batch
translation_ht

skew_symmetric_batch
skew_symmetric

unskew_batch
unskew

screwvec_to_mat_batch
screwvec_to_mat

ht_adj_batch
ht_adj

screwmat_to_ht_batch
screwmat_to_ht: TODO

screwvec_to_ht_batch
screwvec_to_ht: TODO

twist_skew_to_rot_batch
twist_skew_to_rot: TODO

twist_to_rot_batch
twist_to_rot:W TODO

screw_axis_batch: TODO
screw_axis

rot_to_twist_batch: TODO
rot_to_twist: TODO
"""


def invert_ht_batch(ht, out=None):
    """Inverts a 3d 4x4 homogenous transform.
    Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.

    Args:
        ht (torch.Tensor): A (NUM_ENVS x 4 x 4) stack of homogenous transforms
        out (torch.Tensor, optional): A (NUM_ENVS x 4 x 4) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) stack of inverted homogenous transforms. Will be equal to torch.invert(ht) for each ht
    """

    NUM_ENVS = ht.shape[0]

    if out is None:
        out = torch.clone(ht)
    elif out.shape != (NUM_ENVS, 4, 4):
        raise ValueError("Out must be a (NUM_ENVS x 4 x 4) tensor.")
    else:
        out[:] = ht[:]

    # TODO This sucks make it better
    out[:, :3, :3] = out[:, :3, :3].transpose(1, 2)[:, :3, :3]
    out[:, :3, 3] = out[:, :3, :3] @ out[:, :3, 3] + \
        out[:, :3, :3] @ out[:, :3, 3]
    out[:, :3, 3] *= -1

    return out


def invert_ht(ht):
    """Inverts a 3d 4x4 homogenous transform. Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.
    Args:
        ht (torch.Tensor): A (4x4) matrix representing the homogenous transform
    Returns:
        torch.Tensor: A (4 x 4) matrix representing the inverted homogenous transform. Will be equal to torch.invert(ht) for each ht
    """

    out = ht.clone()

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
        out[:] = 0.0

    out[:, 0, 1] = -omegas[:, 2]
    out[:, 0, 2] = omegas[:, 1]
    out[:, 1, 0] = omegas[:, 2]
    out[:, 1, 2] = -omegas[:, 0]
    out[:, 2, 0] = -omegas[:, 1]
    out[:, 2, 1] = omegas[:, 0]

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


def unskew_batch(skews, out=None):
    """Converts a skew symmetric matrix to a 3d vector
    Args:
        skews (torch.Tensor): A (NUM_ENVS x 3 x 3) tensor representing a stack of skew symmetric matrices
        out (torch.Tensor, optional): A (NUM_ENVS x 3) tensor to store the result in. Defaults to None.
    Returns:
        torch.Tensor: A (NUM_ENVS x 3) element vector
    """
    NUM_ENVS = skews.shape[0]

    if out is None:
        out = torch.zeros((NUM_ENVS, 3), device=device)
    elif out.shape != (NUM_ENVS, 3):
        raise ValueError("Out must be a (NUM_ENVS x 3) tensor.")
    else:
        out[:] = 0.0

    out[:, 0] = skews[:, 2, 1]
    out[:, 1] = skews[:, 0, 2]
    out[:, 2] = skews[:, 1, 0]

    return out


def unskew(skew):
    """Converts a skew symmetric matrix to a 3d vector
    Args:
        skew (torch.Tensor): A (3 x 3) tensor representing a skew symmetric matrix
    Returns:
        torch.Tensor: A (3) element vector
    """

    result = torch.zeros((3), device=device)

    result[0] = skew[2, 1]
    result[1] = skew[0, 2]
    result[2] = skew[1, 0]

    return result


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

    out[:, :3, :3] = skew_symmetric(twist)
    out[:, :3, 3] = v

    return out


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
        out[:] = 0.0

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
        out (torch.Tensor, optional): A (NUM_ENVS x 4 x 4) tensor to store the result in. Defaults to None.

    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
    """

    NUM_ENVS = thetas.shape[0]

    if out is None:
        out = torch.zeros((NUM_ENVS, 4, 4), device=device)
    elif out.shape != (NUM_ENVS, 4, 4):
        raise ValueError("Out must be a (NUM_ENVS x 4 x 4) tensor.")

    twist_skew = screwmat[:3, :3]
    v = screwmat[:3, 3]

    twist = unskew(twist_skew)

    if (not twist.norm() == 0.0):
        scratch = out[:, :3, :3]
        # Term 1
        scratch = torch.einsum(
            'ij,B->Bij', torch.eye(3, device=device), thetas)

        # Term 2
        scratch += torch.einsum('ij,B->Bij', twist_skew, 1 - torch.cos(thetas))

        # Term 3
        scratch += torch.einsum('ij,jk,B->Bik', twist_skew,
                                twist_skew, thetas - torch.sin(thetas))

        out[:, :3, 3] = torch.einsum('Bij,j->Bi', scratch, v)

        twist_skew_to_rot_batch(twist_skew, thetas, out=out[:, :3, :3])
        out[:, 3, 3] = 1.0

    else:
        out[:] = torch.eye(4, device=device)
        # TODO: Fix this garbage
        out[:, :3, 3] = torch.einsum('i,B->Bi', v, thetas)

    return out


def screwmat_to_ht(screwmat, theta):
    """Converts a screw axis matrix and magnitude to a homogenous transform. Should be equivalent to expm(screwmat * theta)
    Args:
        screwmat (torch.Tensor): A (4 x 4) tensor representing a screw matrix
        theta (float or torch.Tensor): A scalar representing a screw angle
    Returns:
        torch.Tensor: A (4 x 4) tensor representing a homogenous transform
    """

    if (not torch.is_tensor(theta)):
        theta = torch.tensor([theta], device=device)

    return screwmat_to_ht_batch(screwmat, theta).squeeze(0)


def screwvec_to_ht_batch(screw, thetas, out=None):
    """Converts a screw axis to a homogenous transform
    This is equivalent to the matrix exponential of the skew symmetric matrix of the screw axis times the theta
    Args:
        screw (torch.Tensor): A (6) element vector representing a stack of screw axes
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
        out (torch.Tensor, optional): A (NUM_ENVS x 4 x 4) tensor to store the result in. Defaults to None.

    Returns:
        torch.Tensor: A (NUM_ENVS x 4 x 4) tensor representing a stack of homogenous transforms
    """

    screwmat = screwvec_to_mat(screw)

    return screwmat_to_ht_batch(screwmat, thetas)


def twist_skew_to_rot_batch(twist_skew, thetas, out=None):
    """Converts a twist skew matrix and angle to a rotation matrix
    This is equation 3.51 on page 71 of Modern Robotics by Kevin Lynch
    Args:
        twist_skew (torch.Tensor): A (3 x 3) tensor representing a skew symmetric matrix
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
        out (torch.Tensor, optional): A (NUM_ENVS x 3 x 3) tensor to store the result in. Defaults to None.

    Returns:
        torch.Tensor: A (NUM_ENVS x 3 x 3) tensor representing a stack of rotation matrices
    """

    NUM_ENVS = thetas.shape[0]

    if (out is None):
        out = torch.zeros(NUM_ENVS, 3, 3, device=device)
    elif (out.shape != (NUM_ENVS, 3, 3)):
        raise ValueError("Out tensor must be of shape (NUM_ENVS, 3, 3)")

    out[:] = torch.eye(3, device=device)

    # this line is so bad omg
    out += torch.einsum('ij,B->Bij', twist_skew, torch.sin(thetas))
    out += torch.einsum('ij,jk,B->Bik', twist_skew, twist_skew, 1 - torch.cos(thetas))

    return out


def twist_to_rot_batch(twist, thetas, out=None):
    """Converts a twist axis and angle to a rotation matrix

    Args:
        twists (torch.Tensor): A (3) element vector representing a twist axis
        thetas (torch.Tensor): A (NUM_ENVS) element vector representing a stack of screw angles
        out (torch.Tensor): A (NUM_ENVS x 3 x 3) tensor to write the result to

    Returns:
        torch.Tensor: A (NUM_ENVS x 3 x 3) tensor representing a stack of rotation matrices
    """

    twist_skews = skew_symmetric(twist)

    return twist_skew_to_rot_batch(twist_skews, thetas, out)


def twist_to_rot(twist, theta):
    """Converts a twist axis and angle to a rotation matrix

    Args:
        twist (torch.Tensor): A (3) element vector representing a twist axis
        theta (Float or torch.Tensor): A scalar or 1 element tensor representing a screw angle
    """

    if (not torch.is_tensor(theta)):
        theta = torch.tensor([theta], device=device)

    return twist_to_rot_batch(twist, theta).squeeze(0)


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
