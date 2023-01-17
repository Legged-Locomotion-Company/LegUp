from kin_utils import *
from inv_kin_algs import dls_invkin, pinv_invkin

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_twist_to_rot():
    twist = torch.tensor([0, 0, 1], device=device)
    theta = torch.tensor([torch.pi/2], device=device)

    val = twist_to_rot(twist, theta)

    expected = torch.matrix_exp(skew_symmetric(twist) * theta)

    return val.allclose(expected)


def test_screwvec_to_mat():
    screw = torch.tensor([0., 0., 1., 3.37, -3.37, 0.], device=device)

    mat = screwvec_to_mat(screw)

    expected = torch.tensor([[0., -1., 0., 3.37],
                             [1., 0., 0., -3.37],
                             [0., 0., 0., 0.],
                             [0., 0., 0., 0.]], device=device)

    return mat.allclose(expected)


def test_screw_to_ht():
    screw = torch.tensor([0., 0., 1., 0., -2., 0.], device=device)

    screwmat = screwvec_to_mat(screw)

    val = screwmat_to_ht(screwmat, torch.tensor([torch.pi/4], device=device))

    expected = torch.matrix_exp(screwmat * torch.pi/4)

    return val.allclose(expected)


def test_dls():
    def test_robot_fk(q, params):
        """Forward kinematics for a very simple 2R robot.

        Args:
            q (torch.Tensor): (NUM_ENVS x 2 x 1) joint angles.
            params (dict): Dictionary of parameters.

        Returns:
            torch.Tensor: 2x1 end effector position.
        """

        l1, l2 = params['l1'], params['l2']

        j1_angles, j2_angles = q[:, 0], q[:, 1]

        x = l1 * torch.cos(j1_angles) + l2 * torch.cos(j1_angles + j2_angles)
        y = l1 * torch.sin(j1_angles) + l2 * torch.sin(j1_angles + j2_angles)

        return torch.stack([x, y], dim=1)

    def test_robot_j(q, params):
        """Jacobian for a very simple 2R robot.

        Args:
            q (torch.Tensor): 2x1 joint angles.
            params (dict): Dictionary of parameters.

        Returns:
            torch.Tensor: 2x2 Jacobian matrix.
        """

        l1, l2 = params['l1'], params['l2']

        NUM_ENVS = q.shape[0]

        jacobian = torch.zeros((NUM_ENVS, 2, 2), device=device)

        jacobian[:, 0, 0] = -l1 * \
            torch.sin(q[:, 0]) - l2 * torch.sin(q[:, 0] + q[:, 1])
        jacobian[:, 0, 1] = -l2 * torch.sin(q[:, 0] + q[:, 1])
        jacobian[:, 1, 0] = l1 * \
            torch.cos(q[:, 0]) + l2 * torch.cos(q[:, 0] + q[:, 1])
        jacobian[:, 1, 1] = l2 * torch.cos(q[:, 0] + q[:, 1])

        return jacobian

    params = {'l1': 1., 'l2': 1.}

    target = torch.tensor([-0.5, -0.5], device=device)

    q = torch.tensor([[0, 0], [0.5, 0.5]], device=device)

    for _ in range(100):
        J = test_robot_j(q, params)
        fk = test_robot_fk(q, params)
        e = target - fk
        q += dls_invkin(J, e, 0.1)

    return test_robot_fk(q, params).allclose(target)


if __name__ == '__main__':
    print("TESTING twist_to_rot")
    if test_twist_to_rot():
        print("twist_to_rot PASSED!")
    else:
        print("twist_to_rot FAILED :(")

    print("TESTING screwvec_to_mat")
    if test_screwvec_to_mat():
        print("screwvec_to_mat PASSED!")
    else:
        print("screwvec_to_mat FAILED :(")

    print("TESTING screw_to_ht")
    if test_screw_to_ht():
        print("screw_to_ht PASSED!")
    else:
        print("screw_to_ht FAILED :(")

    print("TESTING dls")
    if test_dls():
        print("dls PASSED!")
    else:
        print("dls FAILED :(")
