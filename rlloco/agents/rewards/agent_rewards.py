from agents.rewards.rewards import *
from agents.rewards.make_reward import Reward


class WildAnymal:
    def __init__(self):
        pass

    def __call__(self, env, action):
        # get the desired and actual velocities
        v_des = env.get_desired_velocity()
        v_act = env.get_actual_velocity()
