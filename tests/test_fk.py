import torch

from legup.common.kinematics import Link, Joint
from legup.common.spatial.spatial import Transform, Position, Direction, Screw


def test_planar_2r_kinematics():
    """Test that the robot kinematics work for a 2r planar robot"""

    ee_link = Link("ee_link")

    fixed_wrist = Joint.make_fixed(
        name="wrist",
        origin=Position.from_iter([1, 0, 0]).make_transform(),
        child_link=ee_link,
    )

    forearm_link = Link("forearm",
                        child_joints=[fixed_wrist])

    elbow_joint = Joint.make_revolute(
        name="elbow",
        origin=Position.from_iter([1, 0, 0]).make_transform(),
        axis=Direction.from_list([0, 0, 1]),
        child_link=forearm_link)

    upper_arm_link = Link("upper_arm",
                          child_joints=[elbow_joint])

    shoulder_joint = Joint.make_revolute("shoulder",
                                         origin=Transform.zero(),
                                         axis=Direction.from_list([0, 0, 1]),
                                         child_link=upper_arm_link)

    base_link = Link(name="base",
                     child_joints=[shoulder_joint])

    kinematics = base_link.make_kinematics(
        query_link_names=["ee_link", "forearm"])

    def stupid_kinematics_and_jacobian(joint_angles):
        """Improved kinematics for a 2r planar robot"""

        angle_0 = joint_angles[..., 0]
        angle_1 = joint_angles[..., 1]
        angle_sum = angle_0 + angle_1

        sin_0 = torch.sin(angle_0)
        cos_0 = torch.cos(angle_0)

        sin_01 = torch.sin(angle_sum)
        cos_01 = torch.cos(angle_sum)

        ee_pos = torch.zeros(
            joint_angles.shape[:-1] + (2,), device=joint_angles.device)
        ee_pos[..., 0] = cos_0 + cos_01
        ee_pos[..., 1] = sin_0 + sin_01

        ee_jacobian = torch.zeros(
            (*joint_angles.shape, 2), device=joint_angles.device)
        ee_jacobian[..., 0, 0] = -sin_0 - sin_01
        ee_jacobian[..., 0, 1] = -sin_01
        ee_jacobian[..., 1, 0] = cos_0 + cos_01
        ee_jacobian[..., 1, 1] = cos_01

        return ee_pos, ee_jacobian

    joint_angles = \
        torch.tensor(
            [[0, 0],
             [0, torch.pi/2],
             [torch.pi/2, 0],
             [torch.pi/2, torch.pi/2],
             [torch.pi/4, 0],
             [torch.pi/4, torch.pi/4],
             [torch.pi/4, torch.pi/2]],
            device=base_link.device,
            dtype=torch.float)

    extra_random_joint_angles = \
        torch.rand((100, 2), device=base_link.device,
                   dtype=torch.float) * 2 * torch.pi

    joint_angles = torch.cat((joint_angles, extra_random_joint_angles), dim=0)

    expected_pos, expected_jacobian = \
        stupid_kinematics_and_jacobian(joint_angles)

    result = kinematics(joint_angles)

    assert result.jacobian.tensor[..., 0, 3:5, :].allclose(
        expected_jacobian, atol=1e-6)
    assert result.transform.tensor[..., 0, :2,
                                   3].allclose(expected_pos, atol=1e-6)


def forward_kinematics_and_jacobian_3d_rrr(joint_angles):

    joint_angles_pre_shape = joint_angles.shape[:-1]

    theta1, theta2, theta3 = \
        joint_angles[..., 0], joint_angles[..., 1], joint_angles[..., 2]

    T_0_1 = torch.tensor([0, 0, 0], dtype=torch.float)
    T_1_2 = torch.tensor([0, 0, 0.1], dtype=torch.float)
    T_2_3 = torch.tensor([1, 0, 0], dtype=torch.float)
    T_3_4 = torch.tensor([1, 0, 0], dtype=torch.float)

    # Here I create a rotation matrix for the z axis shoulder joint
    R_0_1 = torch.zeros(joint_angles_pre_shape + (3, 3))

    R_0_1[..., 0, 0] = torch.cos(theta1)
    R_0_1[..., 0, 1] = -torch.sin(theta1)
    R_0_1[..., 1, 0] = torch.sin(theta1)
    R_0_1[..., 1, 1] = torch.cos(theta1)
    R_0_1[..., 2, 2] = 1

    # Here I create a rotation matrix for the y axis shoulder joint
    R_1_2 = torch.zeros(joint_angles_pre_shape + (3, 3))

    R_1_2[..., 0, 0] = torch.cos(theta2)
    R_1_2[..., 0, 2] = torch.sin(theta2)
    R_1_2[..., 1, 1] = 1
    R_1_2[..., 2, 0] = -torch.sin(theta2)
    R_1_2[..., 2, 2] = torch.cos(theta2)

    # Here I create a rotation matrix for the y axis elbow joint
    R_2_3 = torch.zeros(joint_angles_pre_shape + (3, 3))

    R_2_3[..., 0, 0] = torch.cos(theta3)
    R_2_3[..., 0, 2] = torch.sin(theta3)
    R_2_3[..., 1, 1] = 1
    R_2_3[..., 2, 0] = -torch.sin(theta3)
    R_2_3[..., 2, 2] = torch.cos(theta3)

    # Now I can calculate the position of the end effector
    T_2_4 = T_2_3 + torch.einsum('...ij,...j->...i', R_2_3, T_3_4)
    T_1_4 = T_1_2 + torch.einsum('...ij,...j->...i', R_1_2, T_2_4)
    T_0_4 = T_0_1 + torch.einsum('...ij,...j->...i', R_0_1, T_1_4)

    # Now I can calculate the jacobian

    # Now I rotate all of the axes and partial transforms to the space frame

    s_ax_0_1 = R_0_1 @ torch.tensor([0, 0, 1], dtype=torch.float)
    s_ax_1_2 = R_0_1 @ R_1_2 @ torch.tensor([0, 1, 0], dtype=torch.float)
    s_ax_2_3 = R_0_1 @ R_1_2 @ R_2_3 @ torch.tensor(
        [0, 1, 0], dtype=torch.float)

    g_T_0_4 = T_0_4
    g_T_1_4 = torch.einsum('...ij,...j->...i', R_0_1, T_1_4)
    g_T_2_4 = torch.einsum('...ij,...j->...i', R_0_1 @ R_1_2, T_2_4)
    g_T_3_4 = torch.einsum('...ij,...j->...i', R_0_1 @ R_1_2 @ R_2_3, T_3_4)

    # Now I calculate the cross product of the shoulder joint axis with
    # the end effector position
    jacobian = torch.zeros(
        joint_angles.shape[:-1] + (3, 3), device=joint_angles.device)

    jacobian[..., :, 0] = torch.cross(s_ax_0_1, g_T_1_4)
    jacobian[..., :, 1] = torch.cross(s_ax_1_2, g_T_2_4)
    jacobian[..., :, 2] = torch.cross(s_ax_2_3, g_T_3_4)

    return T_0_4, jacobian


def test_3d_rrr_kinematics_against_expected():
    """Test that the robot kinematics work for a 3D RRR robot"""

    ee_link = Link("ee_link")

    fixed_wrist = Joint.make_fixed(
        name="wrist",
        origin=Position.from_iter([1, 0, 0]).make_transform(),
        child_link=ee_link,
    )

    forearm_link = Link("forearm",
                        child_joints=[fixed_wrist])

    elbow_joint = Joint.make_revolute(
        name="elbow",
        origin=Position.from_iter([1, 0, 0]).make_transform(),
        axis=Direction.from_list([0, 1, 0]),
        child_link=forearm_link)

    upper_arm_link = Link("upper_arm",
                          child_joints=[elbow_joint])

    shoulder_y_joint = Joint.make_revolute(
        name="shoulder_y",
        origin=Position.from_iter([0, 0, 0.1]).make_transform(),
        axis=Direction.from_list([0, 1, 0]),
        child_link=upper_arm_link)

    upper_base_link = Link("upper_base",
                           child_joints=[shoulder_y_joint])

    shoulder_z_joint = Joint.make_revolute("shoulder_z",
                                           origin=Transform.zero(),
                                           axis=Direction.from_list([0, 0, 1]),
                                           child_link=upper_base_link)

    base_link = Link(name="base",
                     child_joints=[shoulder_z_joint])

    # robot = Robot("3d_rrr_shoulder_z_shoulder_y",
    #               base_link=base_link)

    kinematics = base_link.make_kinematics(query_link_names=["ee_link"])

    joint_angles = torch.tensor(
        [[0, 0, 0],
         [0, torch.pi/2, 0],
         [torch.pi/2, 0, 0],
         [torch.pi/2, torch.pi/2, 0],
         [torch.pi/4, 0, 0],
         [torch.pi/4, torch.pi/4, 0],
         [torch.pi/4, torch.pi/2, 0]],
        device=base_link.device,
        dtype=torch.float)

    extra_random_joint_angles = torch.rand((100, 3), device=base_link.device,
                                           dtype=torch.float) * 2 * torch.pi

    joint_angles = torch.cat((joint_angles, extra_random_joint_angles), dim=0)

    expected_pos, expected_jacobian = forward_kinematics_and_jacobian_3d_rrr(
        joint_angles)

    result = kinematics(joint_angles)

    # Convert the 4x4 transform matrix to xyz position
    result_pos = result.transform.tensor[..., 0, :3, 3]

    # Extract the last 3 rows of the 6xnum_dofs jacobian
    result_jacobian = result.jacobian.tensor[..., 0, 3:, :]

    assert result_pos.allclose(expected_pos.to(
        device=result_pos.device), atol=1e-6)
    assert result_jacobian.allclose(
        expected_jacobian.to(device=result_jacobian.device), atol=1e-6)
