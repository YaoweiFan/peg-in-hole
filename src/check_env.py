from stable_baselines.common.env_checker import check_env
import gym_env

env = gym_env.PegInEnv(
        "PandaPegIn",
        has_offscreen_renderer=True,
        # has_renderer=True,
        use_camera_obs=False,
        control_freq=100,
    )
check_env(env, warn=True)