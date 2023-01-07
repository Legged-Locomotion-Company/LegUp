from kin_utils import screw_axis

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
    This class stores a joint of the robot with a child link and a position
    """

    def __init__(self, child_link, joint_screw_ax):
        """This function initializes the joint

        Args:
            child_link (RobotLink): The child link of the joint
            joint_screw_ax (torch.Tensor): a (6) vector containing the screw axis of the joint
        """

        self.child_link = child_link
        self.screw_ax = joint_screw_ax

class RobotJointRev(RobotJoint):
    """
    This class stores a revolute joint of the robot with a child link and a position
    """

    def __init__(self, child_link, ax, origin):
        """This function initializes the joint

        Args:
            child_link (RobotLink): The child link of the joint
            ax (torch.Tensor): a (3) vector containing the axis of the joint
            origin (torch.Tensor): a (3) vector containing the origin of the joint
        """

        screw_ax = screw_axis(origin, ax)

        super().__init__(child_link, screw_ax)