def Kinematics:
    """
    This class precalculates the necessary basics for the robot kinematics,
    and contains the functions that then do the kinematics
    """

    def __init__(self, base_link):
        """This function precalculates the basics for kinematics and unpacks the robot structure

        Args:
            robot_structure (RobotStructure): The robot structure of the robot
        """

        self.joint_zero_transforms = []

        next_joint_idx = 0

        for joint in robot_structure.joints:
            self.joint_zero_transforms.append(
                joint.get_zero_transform(next_joint_idx))
            next_joint_idx += joint.num_dofs

class RobotLink:
    """
    This class stores a link of the robot with child joints and POIs
    """

    def __init__(self, joints, pois):
        """This function initializes the link

        Args:
            robot (Robot): The robot
        """

        self.structure = []

class RobotJoint:
    """
    This class stores a joint of the robot with child links
    """

    def __init__(self, chid_link, joint_screw_ax, joint_ht):
        """This function initializes the joint

        Args:
            chid_link (RobotLink): The child link of the joint
            joint_screw_ax (torch.Tensor): a (6) vector containing the screw axis of the joint
            joint_ht (torch.Tensor): a (6) vector containing the twist vector of the joint
        """

        self.structure = []