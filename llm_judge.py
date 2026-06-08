import os
import re
import pickle
import atexit
import torch
from prompts import SYSTEM_PROMPT

MODEL_PATH = "/content/drive/MyDrive/models/qwen3-4b"
CACHE_FILE = "reward_cache.pkl"

# ── Lazy-loaded model (loaded only on first use) ──────────────────────────────
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("[txt2reward] Loading Qwen model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("[txt2reward] Model loaded.")
    return _tokenizer, _model


# ── Persistent cache ──────────────────────────────────────────────────────────
def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(reward_cache, f)


reward_cache = _load_cache()
atexit.register(_save_cache)   # always save on clean exit / crash


# ── State discretisation ──────────────────────────────────────────────────────
def discretize(speed, lane, distance):
    speed_bin    = int(speed    // 5)
    lane_bin     = int(lane)
    distance_bin = int(min(distance, 100) // 10)   # cap at 100 m
    return (speed_bin, lane_bin, distance_bin)


# ── LLM query ────────────────────────────────────────────────────────────────
def query_qwen(speed, lane, distance):
    tokenizer, model = _load_model()

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Speed:{speed:.1f} km/h\n"
        f"Lane:{lane}\n"
        f"FrontDistance:{distance:.1f} m\n"
        "Score:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Robust parse: find the first standalone digit 0-5
    match = re.search(r"\b([0-5])\b", text)
    score = int(match.group(1)) if match else 3   # default: acceptable

    # Normalise to [-1, +1]
    return (score - 2.5) / 2.5


# ── Public API ────────────────────────────────────────────────────────────────
def judge_state(speed: float, lane: int, front_distance: float) -> float:
    """Return a normalised reward in [-1, +1] for the given driving state."""
    key = discretize(speed, lane, front_distance)

    if key in reward_cache:
        return reward_cache[key]

    reward = query_qwen(speed, lane, front_distance)
    reward_cache[key] = reward

    # Periodic checkpoint every 50 new entries
    if len(reward_cache) % 50 == 0:
        _save_cache()

    return reward
