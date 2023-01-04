import torch
import cProfile

from rlloco.robots.mini_cheetah.kinematics.mini_cheetah_kin_torch import get_home_position, mini_cheetah_dls_invkin, build_jacobian_and_fk_all_feet, build_jacobian_and_fk, mini_cheetah_pinv_invkin

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

    expected = torch.tensor([[[0.0000, -0.4025, 0.0000],
                              [-0.0680,  0.0000,  0.1940],
                              [0.4025,  0.0000,  0.0000]]]).to(device)

    if (not jacobian[:, 3:].allclose(expected)):
        return False
    else:
        return True


def test_mc_fk_all_feet():
    home_pos = get_home_position().reshape(1, -1).repeat(2048, 1)

    positions = build_jacobian_and_fk_all_feet(home_pos)[1].reshape(-1, 12)

    goal = torch.tensor([[0.2132,  0.1170, -0.2804,  0.2132, -0.1170, -0.2804, -0.1923,  0.1170,
                          -0.2804, -0.1923, -0.1170, -0.2804]], device=device)

    return torch.allclose(positions, goal, 1e-3)


def test_mc_dls():
    """Tests the mini_cheetah_dls_invkin function.

    Returns:
        boolean: True if the function worked and False if not
    """

    home_pos = get_home_position().reshape(1, -1).expand(2**6, -1)
    q = home_pos + torch.randn_like(home_pos).to(device)*2

    _, target_foot_pos = build_jacobian_and_fk_all_feet(home_pos)
    goal = target_foot_pos.reshape(-1, 12)

    def dist():
        _, foot_positions = build_jacobian_and_fk_all_feet(q)

        return torch.mean(torch.norm(foot_positions-target_foot_pos, dim=-1), dim=0)

    for _ in range(100):
        print(f"Total Dist: {dist()}")
        q = mini_cheetah_dls_invkin(q, goal)
        # q = mini_cheetah_pinv_invkin(q, goal)

    # print(q)
    # print(goals)

    return torch.sum(dist()) < 1e-5

def profile_fk():
    q = torch.zeros(2048, 12, device=device)

    run = lambda: build_jacobian_and_fk_all_feet(q)

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
