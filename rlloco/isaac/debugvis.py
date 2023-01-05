import numpy as np
from isaacgym.gymutil import WireframeSphereGeometry, draw_lines
from isaacgym import gymapi

def sep(name = "", length = 110):
    name = f" {name} "
    if len(name) == 2:
        name = ''

    num = length - len(name)
    l, r = int(np.floor(num / 2)), int(np.ceil(num / 2))
    print(f"{'-' * l}{name}{'-' * r}")

class DebugInfo:
    def __init__(self, sim_ctx):
        self.gym = sim_ctx.gym
        self.sim = sim_ctx.sim
        self.asset_handle = sim_ctx.asset_handle
        self.env_handles = sim_ctx.env_handles
        self.actor_handles = sim_ctx.actor_handles

        self.red_sphere = WireframeSphereGeometry(0.05, color = (1, 0, 0))
        self.green_sphere = WireframeSphereGeometry(0.05, color = (0, 1, 0))
        self.blue_sphere = WireframeSphereGeometry(0.05, color = (0, 0, 1))
    
    def print_all_info(self):
        self.print_rb_info()
        self.print_joint_info()
        self.print_dof_info()
    
    def draw(self, viewer, actor_handle, env_handle, dyn, idx):
        rb_pos = dyn.get_rb_position()[idx] # (num_rb, 3)
        rb_contact_forces = dyn.get_contact_forces()[idx] # (num_rb, 3)

        for rb_idx in range(len(rb_pos)):
            pos = rb_pos[rb_idx]
            if rb_idx == 0: # [0]
                pass
            elif rb_idx % 3 == 1:
                draw_lines(self.red_sphere, self.gym, viewer, env_handle, gymapi.Transform(p = gymapi.Vec3(*pos))) # [1, 4, 7, 10] = abduct
            elif rb_idx % 3 == 2:
                draw_lines(self.blue_sphere, self.gym, viewer, env_handle, gymapi.Transform(p = gymapi.Vec3(*pos))) # [2, 5, 8, 11] = thigh
            elif rb_idx % 3 == 0:
                draw_lines(self.green_sphere, self.gym, viewer, env_handle, gymapi.Transform(p = gymapi.Vec3(*pos))) # [3, 6, 9, 12] = shank



    
    def visualize(self, viewer, dyn):
        for idx, (actor_handle, env_handle) in enumerate(zip(self.actor_handles, self.env_handles)):
            self.draw(viewer, actor_handle, env_handle, dyn, idx)
        
    def print_rb_info(self):
        sep('Rigid Bodies')

        props = self.gym.get_actor_rigid_body_properties(self.env_handles[0], self.actor_handles[0])
        names = self.gym.get_asset_rigid_body_names(self.asset_handle)
        for idx, (rb_prop, rb_name) in enumerate(zip(props, names)):
            print(f"Rigid Body {idx}: {rb_name}")
            # print(f"\tCenter-of-mass: {rb_prop.com}")
            # print(f"\tflags: {rb_prop.flags}")
            # print(f"\tinertia: {rb_prop.inertia}")
            # print(f"\tinverse of inertia: {rb_prop.invInertia}")
            # print(f"\tMass: {rb_prop.mass} kg")
            # print(f"\tinverse of mass: {rb_prop.invMass}")
            # print()
        
        sep()
    
    def print_joint_info(self):
        sep("Joints")

        names = self.gym.get_asset_joint_names(self.asset_handle)
        for asset_idx, asset_name in enumerate(names):
            joint_type = self.gym.get_asset_joint_type(self.asset_handle, asset_idx)
            type_name = self.gym.get_joint_type_string(joint_type)
            print(f"Asset {asset_idx}: {asset_name} ({type_name})")            

        sep()
    
    def print_dof_info(self):
        sep("Degrees of Freedom")

        names = self.gym.get_asset_dof_names(self.asset_handle)
        props = self.gym.get_asset_dof_properties(self.asset_handle)
        dof_strs = ["DOF_MODE_NONE (free)", "DOF_MODE_POS (position)", "DOF_MODE_VEL (velocity)", "DOF_MODE_EFFORT (torque)"]
        
        for dof_idx, dof_name in enumerate(names):
            dof_type = self.gym.get_asset_dof_type(self.asset_handle, dof_idx)
            type_name = self.gym.get_dof_type_string(dof_type)
            has_lim, lower, upper, mode, kp, kd, vel, effort, friction, armature = props[dof_idx]
            mode = dof_strs[mode]

            print(f"DOF {dof_idx}: {dof_name} ({type_name})")
            print(f"\tHas Limits: {has_lim}")
            print(f"\tLower Limit: {lower}")
            print(f"\tUpper Limit: {upper}")
            print(f"\tDOF Drive Mode: {mode}")
            print(f"\tDrive Stiffness: {kp}")
            print(f"\tDrive Damping: {kd}")
            print(f"\tMax Velocity: {vel}")
            print(f"\tMax Torque: {effort}")
            print(f"\tDOF Friction: {friction}")
            print(f"\tDOF armature: {armature}")

        sep()