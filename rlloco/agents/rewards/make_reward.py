from agents.rewards.agent_rewards import WildAnymal


def make_reward(agent, robot_config):
    if agent == 'wild_anymal':
        return WildAnymal()
