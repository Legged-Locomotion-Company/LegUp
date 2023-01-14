import torch
import cProfile

from legup.robots.mini_cheetah.kinematics.mini_cheetah_kin_torch import get_home_position, mini_cheetah_dls_invkin, build_jacobian_and_fk_all_feet, build_jacobian_and_fk, mini_cheetah_pinv_invkin

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_mc_fk_j():
    q = torch.zeros(1, 12, device=device)

    jacobian, foot_pos = build_jacobian_and_fk(q, 0)

    expected = torch.tensor([[[0.0000, -0.4025, -0.1940],
                              [0.4025,  0.0000,  0.0000],
                              [0.0680,  0.0000,  0.0000]]]).to(device)

    if (not jacobian[:, 3:].allclose(expected)):
        return False

    q[0, 0] = torch.pi/2

    jacobian, foot_pos = build_jacobian_and_fk(q, 0)

    expected = torch.tensor([[[0.0000, -0.4025, -0.1940],
                              [-0.0680,  0.0000,  0.0000],
                              [0.4025,  0.0000,  0.0000]]]).to(device)

    if (not jacobian[:, 3:].allclose(expected)):
        return False

    q[0, 2] = -torch.pi/4

    jacobian, foot_pos = build_jacobian_and_fk(q, 0)

    expected = torch.tensor([[[0.0000, -0.4025, -0.1372],
                              [-0.0680,  0.0000,  0.1372],
                              [0.4025,  0.0000,  0.0000]]], device=device)

    if (not jacobian[:, 3:].allclose(expected, 1e-3)):
        return False

    q[0, 3] = -torch.pi/2

    jacobian, foot_pos = build_jacobian_and_fk(q, 1)

    expected = torch.tensor([[[0.0000, 0.4025, 0.1940],
                              [-0.0680,  0.0000,  0.0000],
                              [-0.4025,  0.0000,  0.0000]]], device=device)

    if (not jacobian[:, 3:].allclose(expected)):
        return False

    q[0, 9] = torch.pi/2
    q[0, 10] = -torch.pi/4
    q[0, 11] = torch.pi/4

    expected = torch.tensor([[[0.0000, 0.2846, 0.1940],
                              [0.0680,  -0.2846,  0.0000],
                              [0.4025,  0.0000,  0.0000]]], device=device)

    jacobian, foot_pos = build_jacobian_and_fk(q, 3)

    if (not jacobian[:, 3:].allclose(expected, 1e-4)):
        return False

    return True


def test_mc_fk_all_feet():
    home_pos = get_home_position().reshape(1, -1).repeat(2048, 1)

    positions = build_jacobian_and_fk_all_feet(home_pos)[1].reshape(-1, 12)

    goal = torch.tensor([[0.2132,  0.1170, -0.2804,  0.1923, -0.1170, -0.2804, -0.1923,  0.1170,
                          -0.2804, -0.2132, -0.1170, -0.2804]], device=device)

    return torch.allclose(positions, goal, 1e-3)


def test_mc_dls():
    """Tests the mini_cheetah_dls_invkin function.

    Returns:
        boolean: True if the function worked and False if not
    """

    NUM_ENVS = 2**6

    home_pos = get_home_position().reshape(1, -1).expand(NUM_ENVS, -1)
    q = home_pos + torch.randn_like(home_pos, device=device)*2

    goal = torch.zeros(NUM_ENVS, 12, device=device)

    goal_foot_pos = build_jacobian_and_fk_all_feet(home_pos)[1]

    def dist():
        _, foot_positions = build_jacobian_and_fk_all_feet(q)

        error = (foot_positions - goal_foot_pos)[1]
        per_foot_error_norms = error.norm(dim=-1)
        per_foot_mean_error = per_foot_error_norms.mean(dim=0)

        return per_foot_mean_error

    i = 0
    MAX_ITERS = 100

    while dist() > 1e-5 and i < 100:
        q = mini_cheetah_dls_invkin(q, goal)
        i += 1

    if (i < MAX_ITERS):
        print(
            f"mini_cheetah_dls_invkin converged in {i} iterations: dist = {dist()}")
        return True
    else:
        print(
            f"mini_cheetah_dls_invkin did not converge in 1000 iterations. Final dist: {dist()}")
        return False


def profile_fk():
    q = torch.zeros(2048, 12, device=device)

    def run(): return build_jacobian_and_fk_all_feet(q)

    return cProfile.runctx('run()', globals(), locals())


if __name__ == '__main__':

    # profile_result = profile_fk()

    print("TESTING build_jacobian_and_fk")
    if test_mc_fk_j():
        print("build_jacobian_and_fk PASSED!")
    else:
        print("build_jacobian_and_fk FAILED :(")

    print("TESTING build_jacobian_and_fk_all_feet")
    if test_mc_fk_all_feet():
        print("build_jacobian_and_fk_all_feet PASSED!")
    else:
        print("build_jacobian_and_fk_all_feet FAILED :(")

    print("TESTING mini_cheetah_dls_invkin FUNCTION")
    if test_mc_dls():
        print("mini_cheetah_dls_invkin PASSED!")
    else:
        print("mini_cheetah_dls_invkin FAILED :(")
