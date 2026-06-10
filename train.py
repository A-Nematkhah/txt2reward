import gymnasium as gym
import highway_env                          # noqa: F401 — env registration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from reward_wrapper import LLMRewardWrapper

# ── تنظیمات محیط ─────────────────────────────────────────────────────────────
ENV_CONFIG = {
    # [FIX-PERF] vehicles_count پیش‌فرض ۵۰ بود → هر step 750 vehicle update!
    "vehicles_count":       15,
    # [FIX-PERF] simulation_frequency پیش‌فرض ۱۵ بود → ۱۵ physics substep/step
    "simulation_frequency":  5,
    "policy_frequency":      1,
    "duration":             40,
    "lanes_count":           4,
    "observation": {
        "type":             "Kinematics",
        "vehicles_count":    5,
        "features":         ["presence", "x", "y", "vx", "vy"],
        "normalize":        True,
        "absolute":         False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "reward_speed_range": [20, 30],
    "collision_reward":   -1.0,
    "high_speed_reward":   0.4,
    "right_lane_reward":   0.1,
    "lane_change_reward":  0.0,
}


def make_env(rank: int = 0, llm_interval: int = 50):
    """factory تابع برای SubprocVecEnv."""
    def _init():
        env = gym.make("highway-v0", config=ENV_CONFIG)
        env = LLMRewardWrapper(env, llm_interval=llm_interval)
        env = Monitor(env)
        return env
    return _init


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",    type=int,   default=100_000)
    parser.add_argument("--n-envs",       type=int,   default=4,
                        help="تعداد محیط‌های موازی")
    parser.add_argument("--llm-interval", type=int,   default=50,
                        help="هر چند step یه‌بار LLM صدا بزنیم")
    parser.add_argument("--no-precompute", action="store_true",
                        help="pre-fill cache رو skip کن")
    args = parser.parse_args()

    # ── Pre-fill cache (همه 192 state قبل از training) ────────────────────────
    if not args.no_precompute:
        print("🔄 Pre-filling reward cache ...")
        from llm_judge import prefill_cache_sync
        prefill_cache_sync()

    # ── محیط‌های موازی ────────────────────────────────────────────────────────
    # [FIX-PERF] SubprocVecEnv: n_envs محیط موازی در process‌های جدا
    # از DummyVecEnv در Colab هم میشه استفاده کرد اگه multiprocessing مشکل داشت
    env_fns = [make_env(rank=i, llm_interval=args.llm_interval)
               for i in range(args.n_envs)]

    try:
        vec_env = SubprocVecEnv(env_fns)
        print(f"✅ SubprocVecEnv با {args.n_envs} محیط")
    except Exception as e:
        print(f"⚠️  SubprocVecEnv شکست خورد ({e})، از DummyVecEnv استفاده میکنم")
        vec_env = DummyVecEnv(env_fns)

    # ── مدل PPO ───────────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cuda",
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        tensorboard_log="./tb_logs/",
    )

    print(f"\n🚀 Training شروع شد — {args.timesteps:,} timesteps، "
          f"{args.n_envs} envs، llm_interval={args.llm_interval}")
    model.learn(total_timesteps=args.timesteps)
    model.save("ppo_highway_qwen_reward")
    vec_env.close()
    print("\n✅ مدل ذخیره شد: ppo_highway_qwen_reward.zip")
