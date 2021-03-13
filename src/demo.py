import numpy as np
import suite
import matplotlib.pyplot as plt

steps = 500

if __name__ == "__main__":

    # initialize the task
    env = suite.make(
        "PandaPegIn",
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()
    # env.viewer.set_camera(camera_id=0)
    
    action = [0, 0, 0]
    for i in range(4):
        obs, reward, done, _, plot_list = env.step(action)

    action = [1, 1, 1]
    obs, reward, done, _, plot_list = env.step(action)
    
    # plt.subplot(131)
    # plt.xlabel('steps', size=12)
    # plt.ylabel('x', size=12)
    # plt.plot(range(steps),np.array(plot_list)[:,0]) 
    # plt.subplot(132)
    # plt.xlabel('steps', size=12)
    # plt.ylabel('y', size=12)
    # plt.plot(range(steps),np.array(plot_list)[:,1]) 
    # plt.subplot(133)
    # plt.xlabel('steps', size=12)
    # plt.ylabel('z', size=12)
    # plt.plot(range(steps),np.array(plot_list)[:,2]) 

    # plt.show()
    while True:
        env.step( [0, 0.1, 0.1] )

    # while True:
    #     env.step( [0, 0, -1] )
    #     if abs(env.us_pos[2] -env.ep_pos[2]) < 0.001:
    #         print("yes!") 
    #         break
    # while True:
    #     env.step( [0, 0.5, -0.1] )
