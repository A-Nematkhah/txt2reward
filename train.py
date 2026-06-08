import gymnasium as gym
import highway_env  # noqa: F401 — registers highway-v0
from stable_baselines3 import PPO
from reward_wrapper import LLMRewardWrapper

# ── Environment config ────────────────────────────────────────────────────────
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": False,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 20,
    "duration": 40,
    "reward_speed_range": [20, 30],
}


def make_env():
    env = gym.make("highway-v0", config=ENV_CONFIG)
    # llm_every=200: LLM is called ~500 times over 100k steps instead of 2000
    env = LLMRewardWrapper(env, llm_every=200)
    return env


# ── Training ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",           # MlpPolicy runs faster on CPU (no GPU transfer overhead)
        n_steps=2048,           # larger rollout = fewer updates = faster wall-clock
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=100_000)
    model.save("ppo_highway_qwen_reward")
    print("Model saved to ppo_highway_qwen_reward.zip")

    env.close()
