import mani_skill.envs
import gymnasium as gym
"""
ManiSkill environments are created by gymnasium’s make function. 
The result is by default a “batched” environment where every input and output is batched.
 Note that this is not standard gymnasium API.
  If you want the standard gymnasium environment / vectorized environment API see the next sections.
"""


N = 4
env = gym.make("PickCube-v1", num_envs=N)
env.action_space # shape (N, D)
env.observation_space # shape (N, ...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )

print(obs)
print(rew)
print(terminated)
print(truncated)
print(info)
