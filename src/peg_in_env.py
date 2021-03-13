import numpy as np
import suite

class PegInEnv:

    def __init__(self, env_name, *args, **kwargs):
        self.env = suite.make( env_name, *args, **kwargs)
        # self.env.viewer.set_camera(camera_id=0)

        self.observation_space = np.zeros(26)
        self.action_space = np.zeros(3)

        

    def reset(self):
        unprocessed_obs = self.env.reset()
        processed_obs = self.__process_observation(unprocessed_obs)
        return processed_obs

    def step(self, action):
        # print('action:',action)
        unprocessed_obs, reward, done, _, self.plot_list = self.env.step(action)
        processed_obs = self.__process_observation(unprocessed_obs)
        # self.env.render()
        info = {}
        return processed_obs, reward, done, info

    def __process_observation(self, obs):
        #need to modify
        processed_obs = np.concatenate([obs["joint_pos"], obs["joint_vel"], obs["prev-act"], obs["ee_force"], obs["ee_torque"], obs["delta_pos"]])        
        # print(processed_obs)
        # processed_obs = np.concatenate([obs["joint_pos"], obs["joint_vel"], obs["prev-act"]])        
        return processed_obs



