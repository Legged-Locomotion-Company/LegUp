from rlloco.isaacgym.environment import IsaacGymEnvironment
import cv2

def main():
    env = IsaacGymEnvironment(4, True, '/home/rohan/Documents/rlloco/rlloco/assets', 'mini-cheetah.urdf')

    for i in range(100000):
        if i % 10000 == 0:
            env.reset()

        env.step()

        # cv2.imshow('Mini Cheetah', env.render())
        # cv2.waitKey(1)
        


if __name__ == '__main__':
    main()