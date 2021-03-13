from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
import gym_env
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./tf_model_logs/')
# Separate evaluation env
eval_env = gym_env.PegInEnv(
        "PandaPegIn",
        has_offscreen_renderer=True,
        # has_renderer=True,
        use_camera_obs=False,
        control_freq=100,
    )

eval_callback = EvalCallback(eval_env, best_model_save_path='./tf_model_logs/best_model',
                             log_path='./tf_model_logs/best_model_results', eval_freq=10000)
# Create the callback list
callback = CallbackList([checkpoint_callback, eval_callback])

env = gym_env.PegInEnv(
        "PandaPegIn",
        has_offscreen_renderer=True,
        # has_renderer=True,
        use_camera_obs=False,
        control_freq=100,
    )

model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=5,
                optim_stepsize=3e-4, optim_batchsize=5, gamma=0.99, lam=0.95, schedule='linear', tensorboard_log='runs', verbose=1)

model.learn(total_timesteps=500000, callback=callback)
