import gymnasium as gym
import highway_env

from stable_baselines3 import PPO

from reward_wrapper import LLMRewardWrapper

env = gym.make("highway-v0")

env = LLMRewardWrapper(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

model.learn(
    total_timesteps=100000
)

model.save("ppo_highway_qwen_reward")
