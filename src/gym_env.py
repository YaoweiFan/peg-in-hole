import numpy as np
import gym
from gym import spaces
import suite

class PegInEnv(gym.Env):

    def __init__(self, env_name, *args, **kwargs):
        super(PegInEnv, self).__init__()

        self.env = suite.make( env_name, *args, **kwargs)
        # self.env.viewer.set_camera(camera_id=0)

        #action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        #observation space
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)



    def reset(self):
        print(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id("endpoint")])
        print(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id("underside")])
        #记录回合步数
        self.ep_steps = 0

        unprocessed_obs = self.env.reset()
        processed_obs = self._process_observation(unprocessed_obs)

        return processed_obs

    def step(self, action):
        self.ep_steps = self.ep_steps + 1
        unprocessed_obs, reward, done, _, self.plot_list = self.env.step(action)
        processed_obs = self._process_observation(unprocessed_obs)
        # self.env.render()
        info = {}
        if self.ep_steps == 512:
            done = 1
        return processed_obs, reward, bool(done), info

    def render(self):
        pass

    def close(self):
        pass

    def _process_observation(self, obs):
        #need to modify
        processed_obs = np.concatenate([obs["joint_pos"], obs["joint_vel"], obs["prev-act"], obs["ee_force"], obs["ee_torque"], obs["delta_pos"]])        
        # print(processed_obs)
        # processed_obs = np.concatenate([obs["joint_pos"], obs["joint_vel"], obs["prev-act"]])      
        return processed_obs
