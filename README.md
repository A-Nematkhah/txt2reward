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
├── train.py               # Entry point — PPO on highway-v0
├── reward_wrapper.py      # Gym wrapper that queries LLM every N steps (async)
├── llm_judge.py           # Groq client, async cache, score → reward
├── prompts.py             # System prompt for the driving evaluator
├── precompute_rewards.py  # Pre-fill reward cache before training
├── requirements.txt       # Dependencies
└── colab_setup.ipynb      # Ready-to-run Google Colab notebook
```

---

## Requirements

- Python 3.10+
- [Groq API key](https://console.groq.com) (free tier کافیه)
- GPU برای training (Colab T4 کافیه)

```bash
pip install -r requirements.txt
```

---

## Quickstart on Google Colab

**سریع‌ترین روش:** `colab_setup.ipynb` رو باز کن.

1. Runtime رو روی GPU بذار (T4 یا بهتر)
2. GROQ_API_KEY رو در cell 2 وارد کن
3. Cellها رو به ترتیب اجرا کن

---

## Local Setup

```bash
export GROQ_API_KEY="gsk_xxxxxxxx"

# مرحله ۱: cache رو از پیش پر کن (حدود ۲ دقیقه)
python precompute_rewards.py

# مرحله ۲: training
python train.py --timesteps 100000 --n-envs 4
```

---

## How It Works

### Reward Wrapper

`LLMRewardWrapper` هر `llm_interval=50` استپ سه feature از observation میگیره:

| Feature | Description |
|---|---|
| `speed_kmh` | سرعت طولی ego vehicle |
| `lane_index` | lane فعلی (0–3) |
| `front_distance` | فاصله به نزدیک‌ترین ماشین جلو |

### LLM Judge (Async)

LLM call **بلاکینگ نیست** — در یه thread جدا اجرا میشه. اگه cache miss اتفاق بیفته، همون step بدون LLM bonus میگذره و cache برای دفعات بعد پر میشه.

| Score | Meaning |
|---|---|
| 0 | Crash risk |
| 1 | Dangerous |
| 2 | Poor |
| 3 | Acceptable |
| 4 | Good |
| 5 | Excellent |

Score نرمالایز میشه به `[-1, +1]` و به reward محیط اضافه میشه.

### Caching

State به یه discrete key تبدیل میشه. فقط **192 state منحصربه‌فرد** وجود داره — بعد از `precompute_rewards.py`، training کاملاً بدون HTTP call ران میشه.

### Performance Fixes

| مشکل | قبل | بعد |
|---|---|---|
| `vehicles_count` | 50 | 15 |
| `simulation_frequency` | 15 | 5 |
| LLM call | blocking | async (ThreadPool) |
| precompute cache | جدا از training | یکپارچه |
| step_count reset | نمیشد | موقع `reset()` ریست میشه |
| Parallelism | 1 env | 4 env (SubprocVecEnv) |

---

## Output

- مدل: `ppo_highway_qwen_reward.zip`
- TensorBoard logs: `./tb_logs/`
- Reward cache: `reward_cache.pkl`

---

## References

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Groq](https://console.groq.com)
- [Text2Reward (paper)](https://arxiv.org/abs/2309.11489)
