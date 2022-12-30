import torch

from rlloco.robots.common.kinematics.inv_kin_algs import dls_invkin

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_home_position():
    return torch.tensor([0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6, 0, -0.8, 1.6]).to(device)


def invert_ht(ht):
    """Inverts a 3d 4x4 homogenous transform. Using this function is a lot better because it does not need the iterative algorithm that is used in regular matrix inversion.

    Args:
        ht (torch.Tensor): The homogenous transform

    Returns:
        torch.Tensor: returns the inverted homogenous transform. Will be equail to torch.invert(ht)
    """

    # TODO This sucks make it better
    ht[:, :3, :3] = ht[:, :3, :3].transpose(1, 2)
    ht[:, :3, 3] = (ht[:, :3, :3] @ ht[:, :3, 3].unsqueeze(2)).squeeze(2)

    return ht


def xrot(theta):
    sth = torch.sin(theta)
    cth = torch.cos(theta)

    result = torch.eye(4).repeat(len(theta), 1, 1)

    result[:, 1, 1] = cth
    result[:, 1, 2] = -sth
    result[:, 2, 1] = sth
    result[:, 2, 2] = cth

    return result


def yrot(theta):
    sth = torch.sin(theta)
    cth = torch.cos(theta)

    result = torch.eye(4).repeat(len(theta), 1, 1)

    result[:, 0, 0] = cth
    result[:, 0, 2] = sth
    result[:, 2, 0] = -sth
    result[:, 2, 2] = cth

    return result

def translation_ht(x, y, z):
    pos = torch.tensor([x, y, z], device= device)

def translation_ht(position: torch.Tensor):
    """Instantiates a new 4x4 homogenous transform tensor which is a pure

    Args:
        pos (torch.Tensor): a (3) vec with (x,y,z) coords for the position to make in the ht

    Returns:
        torch.Tensor: a (4x4) tensor homogenous transform 
    """

    ht = torch.eye(4, device= device)
    ht[0,0:3] = position

    return ht


def build_jacobian_and_fk(q_vec, leg):
    # create the multipliers depending on the leg
    front = 1 if (leg == 0 or leg == 1) else -1
    left = 1 if (leg == 0 or leg == 2) else -1

    NUM_ENVS, _ = q_vec.shape

    # Create the homogenous transforms for the joints
    torso_to_hip = torch.tensor([
        [1, 0, 0, 0.14775 * front],
        [0, 1, 0, 0.049 * left],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32).repeat(NUM_ENVS, 1, 1)
    # [0.055, -0.019, 0]
    hip_to_shoulder = xrot(q_vec[:, 3*leg + 0]) @ translation_ht(0.55*front, 0.19*left, 0)
    # hip_to_shoulder = xrot(q_vec[:, 3*leg + 0]) @ torch.tensor([
    #     [1, 0, 0, 0.055 * front],
    #     [0, 1, 0, 0.019 * left],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    # [0, -0.049, -0.2085]
    shoulder_to_knee = yrot(q_vec[:, 3*leg + 1]) @ translation_ht(0, 0.49*left, -0.2085)
    # shoulder_to_knee = yrot(q_vec[:, 3*leg + 1]) @ torch.tensor([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0.049 * left],
    #     [0, 0, 1, -0.2085],
    #     [0, 0, 0, 1]
    # ])
    # [0, 0, -0.194]
    knee_to_foot = yrot(q_vec[:, 3*leg + 2]) @ translation_ht(0, 0, -1.94)
    # knee_to_foot = yrot(q_vec[:, 3*leg + 2]) @ torch.tensor([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, -0.194],
    #     [0, 0, 0, 1]
    # ])

    # create the axes
    hip_axis = torch.tensor(
        [1, 0, 0], dtype=torch.float32).repeat(len(q_vec), 1)
    shoulder_axis = torch.tensor(
        [0, 1, 0], dtype=torch.float32).repeat(len(q_vec), 1)
    knee_axis = torch.tensor(
        [0, 1, 0], dtype=torch.float32).repeat(len(q_vec), 1)

    # create joint to foots
    shoulder_to_foot = shoulder_to_knee @ knee_to_foot
    hip_to_foot = hip_to_shoulder @ shoulder_to_foot
    torso_to_foot = torso_to_hip @ hip_to_foot

    # create joint to base
    hip_to_torso = invert_ht(torso_to_hip)
    shoulder_to_torso = invert_ht(hip_to_shoulder) @ hip_to_torso
    knee_to_torso = invert_ht(shoulder_to_knee) @ shoulder_to_torso

    # cross the axes to get local gradients
    local_hip_g = torch.one((len(q_vec), 4))
    local_shoulder_g = torch.ones((len(q_vec), 4))
    local_knee_g = torch.ones((len(q_vec), 4))

    # print(torch.cross(hip_axis[1], hip_to_foot[1, :3, 3]))
    local_hip_g[:, :3] = torch.cross(hip_axis, hip_to_foot[:, :3, 3], dim=1)
    local_shoulder_g[:, :3] = torch.cross(
        shoulder_axis, shoulder_to_foot[:, :3, 3], dim=1)
    local_knee_g[:, :3] = torch.cross(knee_axis, knee_to_foot[:, :3, 3], dim=1)

    # rotate the local gradients into the global frame
    hip_g = (hip_to_torso[:, :3, :3] @
             local_hip_g.unsqueeze(2)[:, :3]).squeeze(2)[:, :3]
    shoulder_g = (shoulder_to_torso[:, :3, :3] @
                  local_shoulder_g.unsqueeze(2)[:, :3]).squeeze(2)[:, :3]
    knee_g = (knee_to_torso[:, :3, :3] @
              local_knee_g.unsqueeze(2)[:, :3]).squeeze(2)[:, :3]

    # stack the gradients to get the jacobian
    jacobian = torch.stack((hip_g, shoulder_g, knee_g), dim=2)

    return jacobian.to(device), torso_to_foot[:, :3, 3].to(device)


def build_jacobian_and_fk_all_feet(q_vec):
    FL_J, FL_r = build_jacobian_and_fk(q_vec, 0)
    FR_J, FR_r = build_jacobian_and_fk(q_vec, 1)
    BL_J, BL_r = build_jacobian_and_fk(q_vec, 2)
    BR_J, BR_r = build_jacobian_and_fk(q_vec, 3)

    all_jacobs = torch.stack((FL_J, FR_J, BL_J, BR_J), dim=1)
    all_positions = torch.stack((FL_r, FR_r, BL_r, BR_r), dim=1)

    return all_jacobs, all_positions


def use_ik(q_vec, goals, ik_alg):
    """This function unpacks the stacked up vectors into the feet for the ik function, and restacks them at the end.

    Args:
        q_vec (torch.Tensor): a (NUM_ENVS x 12) vector which contains the joint positions of the robots joints.
        goals (torch.Tensor): a (NUM_ENVS x 12) vector which contains the goal (x,y,z) positions for the feet relative to their home position all stacked up in one vector.
        ik_alk ((torch.tensor, torch.tensor) -> torch.tensor)

    Returns:
        torch.Tensor: returns a (NUM_ENVS x 12) vector which contains the new targets for each of the robot's joints
    """

    # Here we dissect the goals tensor to create a (NUM_ENVS x 4 x 3) vector which contains the joint targets for each of the robots 4 feet separately
    goals_per_foot = torch.reshape(goals, (-1, 4, 3))

    # Here we find the foot errors and the jacobian

    jacobian, absolute_foot_pos = build_jacobian_and_fk_all_feet(q_vec)
    _, home_pos = build_jacobian_and_fk_all_feet(get_home_position().reshape((1,-1)))
    relative_foot_pos = absolute_foot_pos - home_pos
    error = goals_per_foot - relative_foot_pos

    per_foot_errors = ik_alg(jacobian, error)

    return q_vec + per_foot_errors.reshape((-1, 12))

def mini_cheetah_dls_invkin(q_vec, goals):
    """Simply executes the dls_invkin algorithm on the mini_cheetah robot

    Args:
        q_vec (torch.Tensor): a (NUM_ENVS x 12) vector which contains the joint positions of the robots joints.
        goals (torch.Tensor): a (NUM_ENVS x 12) vector which contains the goal (x,y,z) positions for the feet relative to their home position all stacked up in one vector.

    Returns:
        torch.Tensor: returns a (NUM_ENVS x 12) vector which contains the new targets for each of the robot's joints
    """

    return use_ik(q_vec, goals, dls_invkin)


def inv_kin_torch(q_vec, goals):
    # get positions and jacobians
    j0, f0 = build_jacobian_and_fk(q_vec, 0)
    j1, f1 = build_jacobian_and_fk(q_vec, 1)
    j2, f2 = build_jacobian_and_fk(q_vec, 2)
    j3, f3 = build_jacobian_and_fk(q_vec, 3)

    # calculate position deltas
    d0 = goals[:, :3] - f0
    d1 = goals[:, 3:6] - f1
    d2 = goals[:, 6:9] - f2
    d3 = goals[:, 9:12] - f3

    # mag_sq_d0 = torch.sum(torch.pow(d0, 2), dim = -1)
    # mag_sq_d1 = torch.sum(torch.pow(d1, 2), dim = -1)
    # mag_sq_d2 = torch.sum(torch.pow(d2, 2), dim = -1)
    # mag_sq_d3 = torch.sum(torch.pow(d3, 2), dim = -1)

    print(
        f"mag_d: {torch.norm(d0)}, {torch.norm(d1)}, {torch.norm(d2)}, {torch.norm(d3)}")

    # print("Pinv: ", torch.pinverse(j0[0]))

    # build joint deltas
    jd0 = torch.concat(
        [torch.pinverse(j0_i) @ d0_i for j0_i, d0_i in zip(j0, d0)], dim=-1).view(-1, 3)  # * mag_sq_d0
    jd1 = torch.concat(
        [torch.pinverse(j1_i) @ d1_i for j1_i, d1_i in zip(j1, d1)], dim=-1).view(-1, 3)  # * mag_sq_d1
    jd2 = torch.concat(
        [torch.pinverse(j2_i) @ d2_i for j2_i, d2_i in zip(j2, d2)], dim=-1).view(-1, 3)  # * mag_sq_d2
    jd3 = torch.concat(
        [torch.pinverse(j3_i) @ d3_i for j3_i, d3_i in zip(j3, d3)], dim=-1).view(-1, 3)  # * mag_sq_d3

    # get new joint positions given the new joint deltas
    # q_delta = torch.clamp(torch.concat([jd0, jd1, jd2, jd3], dim=1), -0.1, 0.1)
    q_delta = torch.concat([jd0, jd1, jd2, jd3], dim=1)
    # print(f"goals: {goals}")
    # print(f"q_delta: {q_delta}")
    # print(q_delta)
    return q_vec + q_delta/2
