import os
import pickle
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompts import SYSTEM_PROMPT

MODEL_PATH = "/content/drive/MyDrive/models/qwen3-4b"

# ── 4-bit quantization ────────────────────────────────────────────────────────
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
    device_map="cuda:0",
)
model.eval()

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_FILE = "reward_cache.pkl"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        reward_cache = pickle.load(f)
else:
    reward_cache = {}

def discretize(speed, lane, distance):
    return (int(speed // 5), int(lane), int(distance // 10))

@torch.inference_mode()
def query_qwen(speed, lane, distance):
    # thinking mode رو غیرفعال کن با /no_think
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Speed:{speed:.0f} Lane:{lane} FrontDistance:{distance:.0f}\n"
        f"Reply with ONLY a single digit 0-5. /no_think\nScore:"
    )

    # attention_mask رو صریح بساز تا warning نده
    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    input_ids      = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        # Qwen3 thinking mode رو با max_new_tokens بزرگ‌تر bypass می‌کنیم
        # و بعد عدد رو از کل output extract می‌کنیم
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # فقط بخش جدید رو decode کن (بعد از prompt)
    new_tokens = outputs[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # thinking block رو حذف کن: <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # اولین عدد ۰ تا ۵ رو پیدا کن
    match = re.search(r"[0-5]", text)
    score = int(match.group()) if match else 3   # fallback: acceptable

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
