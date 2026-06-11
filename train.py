import os
import shutil
import gymnasium as gym
import highway_env                          # noqa: F401 — registers highway-v0
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from reward_wrapper import LLMRewardWrapper

# ── Environment configuration ─────────────────────────────────────────────────
ENV_CONFIG = {
    # Reduced from default 50 — each step was simulating 750 vehicle updates
    "vehicles_count":       15,
    # Reduced from default 15 — each step was running 15 physics substeps
    "simulation_frequency":  5,
    "policy_frequency":      1,
    "duration":             40,
    "lanes_count":           4,
    "observation": {
        "type":           "Kinematics",
        "vehicles_count":  5,
        "features":       ["presence", "x", "y", "vx", "vy"],
        "normalize":      True,
        "absolute":       False,
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
    """Return an env factory function suitable for SubprocVecEnv / DummyVecEnv."""
    def _init():
        env = gym.make("highway-v0", config=ENV_CONFIG)
        env = LLMRewardWrapper(env, llm_interval=llm_interval)
        env = Monitor(env)
        return env
    return _init


# ── Google Drive sync callback ────────────────────────────────────────────────

class DriveSyncCallback(BaseCallback):
    """
    After every `sync_freq` steps, copies the latest SB3 checkpoint and the
    reward cache to Google Drive so progress survives a Colab disconnect.
    """

    def __init__(self, drive_dir: str, sync_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.drive_dir = drive_dir
        self.sync_freq = sync_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.sync_freq == 0:
            self._sync()
        return True  # returning False would stop training

    def _sync(self) -> None:
        os.makedirs(self.drive_dir, exist_ok=True)

        # Copy all checkpoint files produced by CheckpointCallback
        for fname in os.listdir("."):
            if fname.startswith("ppo_highway") and fname.endswith(".zip"):
                shutil.copy(fname, os.path.join(self.drive_dir, fname))

        # Copy reward cache
        if os.path.exists("reward_cache.pkl"):
            shutil.copy(
                "reward_cache.pkl",
                os.path.join(self.drive_dir, "reward_cache.pkl"),
            )

        print(f"[drive] Synced at step {self.num_timesteps:,} → {self.drive_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on highway-v0 with LLM rewards")
    parser.add_argument("--timesteps",     type=int,   default=100_000)
    parser.add_argument("--n-envs",        type=int,   default=4)
    parser.add_argument("--llm-interval",  type=int,   default=50)
    parser.add_argument("--no-precompute", action="store_true",
                        help="Skip cache pre-fill (use when cache already exists)")
    parser.add_argument("--resume",        type=str,   default=None,
                        metavar="PATH",
                        help="Path to a .zip checkpoint to resume training from")
    parser.add_argument("--drive-dir",     type=str,
                        default="/content/drive/MyDrive/txt2reward",
                        help="Google Drive folder for checkpoints and cache")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                        help="Save a checkpoint every N steps")
    args = parser.parse_args()

    # ── Restore cache from Drive if available and not already present ─────────
    drive_cache = os.path.join(args.drive_dir, "reward_cache.pkl")
    if not os.path.exists("reward_cache.pkl") and os.path.exists(drive_cache):
        shutil.copy(drive_cache, "reward_cache.pkl")
        print(f"[train] Restored reward_cache.pkl from {args.drive_dir}")

    # ── Pre-fill the reward cache ─────────────────────────────────────────────
    if not args.no_precompute:
        print("[train] Pre-filling reward cache ...")
        from llm_judge import prefill_cache_sync
        prefill_cache_sync()
        # Immediately back up to Drive
        if os.path.exists(args.drive_dir) or args.drive_dir.startswith("/content/drive"):
            os.makedirs(args.drive_dir, exist_ok=True)
            shutil.copy("reward_cache.pkl", drive_cache)
            print(f"[train] Cache backed up to {drive_cache}")

    # ── Build parallel environments ───────────────────────────────────────────
    env_fns = [make_env(rank=i, llm_interval=args.llm_interval)
               for i in range(args.n_envs)]

    try:
        vec_env = SubprocVecEnv(env_fns)
        print(f"[train] Using SubprocVecEnv with {args.n_envs} workers")
    except Exception as e:
        print(f"[train] SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv(env_fns)

    # ── Build or restore the PPO model ────────────────────────────────────────
    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        print(f"[train] Resumed from checkpoint: {args.resume}")
    else:
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

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=".",
        name_prefix="ppo_highway",
    )
    drive_sync_cb = DriveSyncCallback(
        drive_dir=args.drive_dir,
        sync_freq=args.checkpoint_freq,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(
        f"\n[train] Starting — {args.timesteps:,} timesteps | "
        f"{args.n_envs} envs | llm_interval={args.llm_interval} | "
        f"checkpoint every {args.checkpoint_freq:,} steps"
    )
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.resume is None,
        callback=[checkpoint_cb, drive_sync_cb],
    )

    # ── Final save ────────────────────────────────────────────────────────────
    model.save("ppo_highway_qwen_reward")
    shutil.copy(
        "ppo_highway_qwen_reward.zip",
        os.path.join(args.drive_dir, "ppo_highway_qwen_reward.zip"),
    )
    vec_env.close()
    print(f"\n[train] Done. Final model saved locally and to {args.drive_dir}")
