from rlloco.isaacgym.environment import IsaacGymEnvironment

def main():
    env = IsaacGymEnvironment(1, 1. / 60., '/home/rohan/Documents/LLC/rlloco/assets', 'mini-cheetah.urdf')
    env.should_render = True

    '''
    for i in range(100000):
        if i % 1000 == 0:
            env.reset()

        env.step()
        env.render()
    '''


if __name__ == '__main__':
    main()