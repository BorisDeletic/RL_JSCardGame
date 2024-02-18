import gymnasium as gym
import torch as th
import cardenv

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


model_name = "ppo_jscardgame"
replay_buffer = "replay_buffer"
checkpoint_dir = "checkpoints/"
env_id = "JSCardGame-v0"

# env = gym.make(env_id)
# env = gym.wrappers.FlattenObservation(env)

def create_new_model():
    num_cpu = 20  # Number of processes to use
    # Create the vectorized environment
    vec_env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # vec_env = make_vec_env(env_id, n_envs=num_cpu, env_kwargs={'render_mode':'ansi'})

    eval_callback = EvalCallback(vec_env, eval_freq=1e5 // num_cpu, n_eval_episodes=1000, log_path="./logs/")

    policy_kwargs = dict(activation_fn=th.nn.ReLU)
    model = PPO("MultiInputPolicy",
                vec_env,
                n_steps=32,
                batch_size=64,
                gae_lambda=0.9,
                gamma=0.995,
                n_epochs=10,
                ent_coef=0.00688,
                learning_rate=9.21e-05,
                clip_range=0.4,
                vf_coef=0.989,
                verbose=1,
                tensorboard_log="./logs/",
                policy_kwargs=policy_kwargs)

    print(model.policy)
    model.learn(total_timesteps=1e6, callback=[eval_callback])

    model.save(checkpoint_dir + model_name)
    # model.save_replay_buffer(checkpoint_dir + replay_buffer)

def train(checkpoint = None):
    print("CHECKPOINT: {}".format(checkpoint))
    num_cpu = 20  # Number of processes to use
    # Create the vectorized environment
    # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
    vec_env = make_vec_env(env_id,
                           n_envs=num_cpu,
                           env_kwargs={'render_mode':None},
                           vec_env_cls=SubprocVecEnv,
                           vec_env_kwargs={'start_method': 'fork'})

    # vec_env = make_vec_env(env_id,
    #                        n_envs=num_cpu,
    #                        env_kwargs={'render_mode':None},
    #                        vec_env_cls=DummyVecEnv)

    eval_callback = EvalCallback(vec_env,
                                 eval_freq=1e5 // num_cpu,
                                 n_eval_episodes=1000,
                                 log_path="./logs/")
    checkpoint_callback = CheckpointCallback(save_freq=1e6 // num_cpu,
                                             save_path='./checkpoints/',
                                             name_prefix=model_name,
                                             save_replay_buffer=True)

    checkpoint_name = model_name if checkpoint is None else checkpoint

    model = PPO.load(checkpoint_dir + checkpoint_name, vec_env, tensorboard_log="./logs/", verbose=1) #, device="mps"
    # model.load_replay_buffer(checkpoint_dir + replay_buffer)

    print(model.policy)
    model.learn(total_timesteps=1e7, callback=[eval_callback, checkpoint_callback])


def load_model(checkpoint):
    num_cpu = 1
    # vec_env = make_vec_env(env_id, n_envs=1, env_kwargs=dict(render_mode='ansi'))
    # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork', env_kwargs=dict(render_mode='ansi'))

    env = gym.make(env_id, render_mode='ansi')
    # env = gym.wrappers.FlattenObservation(env)

    model = PPO.load(checkpoint_dir + checkpoint, env) #, device="mps"
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


# create_new_model()
checkpoint = "ppo_jscardgame_10000000_steps"
# checkpoint = "ppo_jscardgame"
train(checkpoint)
# load_model(checkpoint)