import gymnasium as gym
import cardenv
import os

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
        env = gym.wrappers.FlattenObservation(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


checkpoint_dir = "checkpoints/ppo_jscardgame"
periodic_checkpoint_dir = "./model_checkpoints/"
env_id = "JSCardGame-v0"

# env = gym.make(env_id)
# env = gym.wrappers.FlattenObservation(env)


def train():
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
    # vec_env = make_vec_env(env_id, n_envs=num_cpu)

    eval_callback = EvalCallback(env, eval_freq=1e5 // num_cpu, n_eval_episodes=100, log_path="./logs/")
    checkpoint_callback = CheckpointCallback(save_freq=1e6 // num_cpu, save_path='./model_checkpoints/')

    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model = PPO.load('./model_checkpoints/rl_model_10000000_steps', env, tensorboard_log="./logs/", verbose=1) #, device="mps"
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
    model = PPO.load(checkpoint_dir)
    mean_reward, std_reward = evaluate_policy(model, env)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # env.

train()