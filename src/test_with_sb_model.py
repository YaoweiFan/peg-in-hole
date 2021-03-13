from stable_baselines import PPO1
import gym_env
import numpy as np

env = gym_env.PegInEnv(
        "PandaPegIn",
        has_offscreen_renderer=True,
        has_renderer=True,
        use_camera_obs=False,
        control_freq=100,
    )
model = PPO1.load("rl_model_198656_steps")

obs = env.reset()
# while True:
#     end_pos = env.env.sim.data.site_xpos[env.env.sim.model.site_name2id("endpoint")]
#     action = ([ 0.4755, -0.0352, 1.0216] - end_pos)*10
#     obs, reward, dones, info = env.step(action)
#     if np.linalg.norm([0.4755, -0.0352, 1.0216] - end_pos) < 0.001:
#         break
# print ("ok")
while True:
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    env.render()