# txt2reward

**LLM-as-Reward for Highway Reinforcement Learning**

Train a highway driving agent with PPO using a language model (Qwen3-4B) as the reward signal instead of hand-crafted reward functions.

---

## Idea

Rather than defining rewards with manual formulas, we ask an LLM to evaluate the current driving state and return a score. This approach is known as **text-to-reward**.

```
env state  ──▶  LLM Judge  ──▶  reward signal  ──▶  PPO update
(speed, lane,   (Qwen3-4B)      [-1, +1]
 front dist)
```

---

## Project Structure

```
txt2reward/
├── train.py            # Entry point — PPO on highway-v0
├── reward_wrapper.py   # Gym wrapper that queries LLM every N steps
├── llm_judge.py        # Qwen loader, caching, score → reward conversion
├── prompts.py          # System prompt for the driving evaluator
├── requirements.txt    # Dependencies
└── colab_setup.ipynb   # Ready-to-run Google Colab notebook
```

---

## Requirements

- Python 3.10+
- GPU with at least 8 GB VRAM (for Qwen3-4B in float16)
- Qwen3-4B model weights downloaded locally

```bash
pip install -r requirements.txt
```

---

## Quickstart on Google Colab

The fastest way to run is via `colab_setup.ipynb`.

1. Place the Qwen3-4B model in your Google Drive at:
   ```
   MyDrive/models/qwen3-4b/
   ```

2. Open the notebook in Colab with a GPU runtime (T4 or better)

3. Run cells in order:
   - Install dependencies
   - Mount Drive
   - Clone repo
   - Quick LLM judge sanity check
   - Start training

---

## Local Setup

Set the model path via environment variable, then run:

```bash
export MODEL_PATH="/path/to/your/qwen3-4b"
python train.py
```

---

## How It Works

### Reward Wrapper

`LLMRewardWrapper` extracts three features from the Kinematics observation every `llm_every=50` steps:

| Feature | Description |
|---|---|
| `speed_kmh` | Ego vehicle longitudinal speed |
| `lane_index` | Current lane (0–3) |
| `front_distance` | Distance to closest vehicle ahead |

### LLM Judge

Qwen3-4B receives the driving state and returns a score from 0 to 5:

| Score | Meaning |
|---|---|
| 0 | Crash risk (very high speed + very close to front vehicle) |
| 1 | Dangerous |
| 2 | Poor |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent (speed ~25–30 km/h, distance > 20 m, middle lane) |

The score is normalized to `[-1, +1]` and added on top of the environment's default reward.

### Caching

To minimize LLM calls, each state is discretized into a key before querying. Results are persisted in `reward_cache.pkl` and checkpointed every 50 new entries.

---

## Output

After training completes:
- Trained model saved to `ppo_highway_qwen_reward.zip`
- TensorBoard logs in `./tb_logs/`
- Reward cache in `reward_cache.pkl`

---

## Known Limitations

- First visit to any new discretized state requires a full LLM inference (slow)
- Model path must be set via the `MODEL_PATH` environment variable
- Training on CPU is impractically slow

---

## References

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-4B)
- [Text2Reward (paper)](https://arxiv.org/abs/2309.11489)
