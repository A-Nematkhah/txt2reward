# txt2reward

**LLM-as-Reward for Highway Reinforcement Learning**

Train a highway driving agent with PPO using a language model (Groq / llama-3.3-70b) as the reward signal instead of hand-crafted reward functions.

---

## Idea

Rather than defining rewards with manual formulas, we ask an LLM to evaluate the current driving state and return a score. This approach is known as **text-to-reward**.

```
env state  ──▶  LLM Judge  ──▶  reward signal  ──▶  PPO update
(speed, lane,   (Groq API)      [-1, +1]
 front dist)
```

---

## Project Structure

```
txt2reward/
├── train.py               # Entry point — PPO training on highway-v0
├── reward_wrapper.py      # Gym wrapper: queries LLM every N steps (async)
├── llm_judge.py           # Groq client, async cache, score → reward
├── prompts.py             # System prompt for the driving evaluator
├── precompute_rewards.py  # Pre-fill reward cache before training
├── requirements.txt       # Python dependencies
└── colab_setup.ipynb      # Ready-to-run Google Colab notebook
```

---

## Requirements

- Python 3.10+
- [Groq API key](https://console.groq.com) — free tier is sufficient
- GPU for training (Colab T4 works fine)

```bash
pip install -r requirements.txt
```

---

## Quickstart on Google Colab

**Fastest path:** open `colab_setup.ipynb`.

1. Set the runtime to GPU (Runtime → Change runtime type → T4 GPU)
2. Paste your `GROQ_API_KEY` in cell 2
3. Run cells in order

---

## Local Setup

```bash
export GROQ_API_KEY="gsk_xxxxxxxx"

# Step 1: pre-fill the reward cache (~2 minutes)
python precompute_rewards.py

# Step 2: train
python train.py --timesteps 100000 --n-envs 4
```

### CLI flags for `train.py`

| Flag | Default | Description |
|---|---|---|
| `--timesteps` | `100000` | Total environment steps |
| `--n-envs` | `4` | Number of parallel environments |
| `--llm-interval` | `50` | Steps between LLM reward queries |
| `--no-precompute` | off | Skip cache pre-fill (use if cache already exists) |

---

## How It Works

### Reward Wrapper

`LLMRewardWrapper` extracts three features from the Kinematics observation every `llm_interval` steps:

| Feature | Description |
|---|---|
| `speed_kmh` | Ego vehicle longitudinal speed |
| `lane_index` | Current lane (0–3) |
| `front_distance` | Distance to the closest vehicle ahead (metres) |

### LLM Judge (Async)

The Groq call runs in a background thread — the training loop is **never blocked**. On a cache miss, the current step returns `0.0` as the LLM bonus and the result is written to the cache for all future visits to that state.

The LLM returns a score from 0 to 5:

| Score | Meaning |
|---|---|
| 0 | Crash risk (very high speed + very close front vehicle) |
| 1 | Dangerous |
| 2 | Poor |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent (speed ~25–30 km/h, distance > 20 m, middle lane) |

The score is normalised to `[-1, +1]` and added on top of the environment's default reward.

### Caching

Each state is discretised into a `(speed_bin, lane, distance_bin)` key before querying. Only **192 unique states** exist in the discretised space — after running `precompute_rewards.py`, training runs completely offline with zero HTTP calls.

The cache is persisted to `reward_cache.pkl` and checkpointed every 50 new entries. All cache operations are thread-safe.

---

## Performance

The following changes were made to fix the original ~9 FPS:

| Issue | Before | After |
|---|---|---|
| `vehicles_count` | 50 | 15 |
| `simulation_frequency` | 15 Hz | 5 Hz |
| Groq API call | blocking | async (ThreadPoolExecutor) |
| `precompute` cache | disconnected from training | unified module-level dict |
| `step_count` reset | never reset | reset on `env.reset()` |
| Parallelism | 1 env | 4 envs (SubprocVecEnv) |

**Expected FPS after fixes: ~80–150** (vs ~9 before)

---

## Output Files

| File | Description |
|---|---|
| `ppo_highway_qwen_reward.zip` | Trained PPO model |
| `tb_logs/` | TensorBoard training logs |
| `reward_cache.pkl` | Persisted LLM reward cache |

---

## Known Limitations

- The reward signal only uses three features (speed, lane, front distance). Richer context (neighbouring vehicles, heading, acceleration) would improve reward quality.
- Groq free tier is rate-limited to ~30 req/min. `precompute_rewards.py` throttles at 0.5 s/request to stay within the limit.
- Training on CPU is impractically slow.

---

## References

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Groq](https://console.groq.com)
- [Text2Reward paper](https://arxiv.org/abs/2309.11489)
