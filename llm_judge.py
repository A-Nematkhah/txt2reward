import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import SYSTEM_PROMPT

MODEL_PATH = "/content/drive/MyDrive/models/qwen3-4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)

CACHE_FILE = "reward_cache.pkl"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        reward_cache = pickle.load(f)
else:
    reward_cache = {}

def discretize(speed, lane, distance):
    speed_bin = int(speed // 5)
    lane_bin = int(lane)
    distance_bin = int(distance // 10)
    return (speed_bin, lane_bin, distance_bin)

def query_qwen(speed, lane, distance):
    prompt = f"""
{SYSTEM_PROMPT}

Speed:{speed:.0f}
Lane:{lane}
FrontDistance:{distance:.0f}

Score:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    score = 3

    for i in range(6):
        if str(i) in text:
            score = i
            break

    return (score - 2.5) / 2.5

def judge_state(speed, lane, front_distance):
    key = discretize(speed, lane, front_distance)

    if key in reward_cache:
        return reward_cache[key]

    reward = query_qwen(speed, lane, front_distance)

    reward_cache[key] = reward

    if len(reward_cache) % 50 == 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(reward_cache, f)

    return reward
