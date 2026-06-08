import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompts import SYSTEM_PROMPT

MODEL_PATH = "/content/drive/MyDrive/models/qwen3-4b"

# ── 4-bit quantization: نصف VRAM، inference سریع‌تر ──────────────────────────
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="cuda:0",          # مستقیم روی GPU، نه auto
)
model.eval()                       # غیرفعال کردن dropout

# ── Prompt ثابت رو یک‌بار tokenize کن ──────────────────────────────────────
_PREFIX = f"{SYSTEM_PROMPT}\nSpeed:"
_PREFIX_IDS = tokenizer(_PREFIX, return_tensors="pt").input_ids.to("cuda:0")
_PREFIX_LEN  = _PREFIX_IDS.shape[1]

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_FILE = "reward_cache.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        reward_cache = pickle.load(f)
else:
    reward_cache = {}

def discretize(speed, lane, distance):
    """گرانول‌بندی دقیق‌تر برای cache hit بیشتر"""
    speed_bin    = int(speed    // 5)
    lane_bin     = int(lane)
    distance_bin = int(distance // 10)
    return (speed_bin, lane_bin, distance_bin)

@torch.inference_mode()           # سریع‌تر از no_grad، بدون overhead
def query_qwen(speed, lane, distance):
    suffix = f"{speed:.0f}\nLane:{lane}\nFrontDistance:{distance:.0f}\nScore:"
    suffix_ids = tokenizer(suffix, return_tensors="pt",
                           add_special_tokens=False).input_ids.to("cuda:0")

    input_ids = torch.cat([_PREFIX_IDS, suffix_ids], dim=1)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,            # KV-cache فعال
    )

    # فقط token جدید رو decode کن
    new_token = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_token, skip_special_tokens=True).strip()

    score = 3  # مقدار پیش‌فرض acceptable
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

    # هر ۵۰ entry جدید، cache رو روی دیسک ذخیره کن
    if len(reward_cache) % 50 == 0:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(reward_cache, f)

    return reward
