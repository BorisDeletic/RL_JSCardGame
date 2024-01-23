import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import cardenv

env = gym.make('JSCardGame-v0', render_mode='ansi')
observation, info = env.reset()

for _ in range(5):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(observation)
    print("REWARD: ", reward)
    print("TERMINATED: ", terminated)
    print()

    if terminated or truncated:
        observation, info = env.reset()

env.close()

#
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}


