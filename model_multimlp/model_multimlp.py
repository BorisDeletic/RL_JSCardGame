import gymnasium as gym
import cardenv
import os
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from typing import Callable


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


checkpoint_dir = "checkpoints/ppo_jscardgame_multi"
periodic_checkpoint_dir = "./checkpoints/"
env_id = "JSCardGame-v0"

# env = gym.make(env_id)
# env = gym.wrappers.FlattenObservation(env)

def train():
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
    # vec_env = make_vec_env(env_id, n_envs=num_cpu)


    eval_callback = EvalCallback(env, eval_freq=1e5 // num_cpu, n_eval_episodes=1000, log_path="./logs/")
    checkpoint_callback = CheckpointCallback(save_freq=1e6 // num_cpu, save_path='./checkpoints/')

    # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128, 128])
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs)
    # model = PPO.load('./checkpoints/rl_model_1000000_steps', env, tensorboard_log="./logs/", verbose=1) #, device="mps"
    print(model.policy)
    model.learn(total_timesteps=1e7, callback=[eval_callback, checkpoint_callback])

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(6):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()

    model.save(checkpoint_dir)
    # stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    # env.save(stats_path)
    env.close()


def load_model():
    num_cpu = 1
    # vec_env = make_vec_env(env_id, n_envs=1, env_kwargs=dict(render_mode='ansi'))
    # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork', env_kwargs=dict(render_mode='ansi'))

    env = gym.make(env_id, render_mode='ansi')
    env = gym.wrappers.FlattenObservation(env)

    model = PPO.load('./rl_model_10000000_steps', env) #, device="mps"
    mean_reward, std_reward = evaluate_policy(model, env)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # vec_env = model.get_env()
    obs, info = env.reset()
    # state = model.env.render(mode="ansi")
    # print(state)

    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(env.print_action(action))
        state = env.render()
        print(state)


train()
# load_model()