import torch

from mini_cheetah_kin_torch import get_home_position, mini_cheetah_dls_invkin, build_jacobian_and_fk_all_feet

def test_mc_dls():
    """Tests the mini_cheetah_dls_invkin function.

    Returns:
        boolean: True if the function worked and False if not
    """

    home_pos = get_home_position.reshape(1,-1)
    q = home_pos + torch.randn_like(home_pos).cuda()*2

    goals = build_jacobian_and_fk_all_feet(home_pos).reshape(-1, 12)

    for _ in range(100):
        q = mini_cheetah_dls_invkin(q, goals)

    return torch.allclose(q, goals)

if __name__ == '__main__':
    print("TESTING mini_cheetah_dls_invkin FUNCTION")
    assert(test_mc_dls, "mini_cheetah_dls_invkin problem :(")
    print("mini_cheetah_dls_invkin PASSED!")