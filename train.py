import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from reward_wrapper import LLMRewardWrapper

env = gym.make("highway-v0")
env = LLMRewardWrapper(env, llm_interval=50)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",        # ← GPU به جای CPU
    n_steps=512,          # buffer کوچک‌تر = آپدیت سریع‌تر در Colab
    batch_size=64,
    n_epochs=5,
)

model.learn(total_timesteps=100_000)
model.save("ppo_highway_qwen_reward")
