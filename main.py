import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np

env = gym.make('JSCardGame-v0', render_mode='ansi')
env = FlattenObservation(env)

print(env.observation_space)
observation, info = env.reset()

for _ in range(5):
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    action = np.zeros(52)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    state = env.render()
    print(state)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

#
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}


