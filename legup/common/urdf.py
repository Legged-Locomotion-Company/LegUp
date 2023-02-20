from typing import Dict, Union, List

from urdfpy import URDF

# Path: urdfpy.py


def create_robot_from_urdf(urdf_path):
    urdf_robot = URDF.load(urdf_path)

    num_dofs = len(urdf_robot.actuated_joints)

    

    robot_tree: Dict[str, List[str, any]] = {}
