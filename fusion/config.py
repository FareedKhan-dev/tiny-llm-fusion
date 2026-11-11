"""config.py: the models, prices, budgets and addresses every other file reads."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    id: str                 # the Hugging Face id we serve
    lab: str                # who trained it
    role: str               # what the system asks of it
    temperature: float      # sampling temperature for its candidates
    max_len: int            # vLLM --max-model-len
    lora: bool = False      # keeps LoRA slots for adapters


MODELS = {
    "qwen": Spec("Qwen/Qwen3.5-4B", "Qwen", "proposer, aggregator", 0.7, 16384),
    "llama": Spec("meta-llama/Llama-3.2-3B-Instruct", "Meta", "proposer", 0.8, 8192, lora=True),
    "gemma": Spec("google/gemma-4-E2B-it", "Google", "proposer", 0.8, 8192, lora=True),
    "phi": Spec("microsoft/Phi-4-mini-instruct", "Microsoft", "proposer", 0.7, 8192),
    "reward": Spec("Skywork/Skywork-Reward-V2-Llama-3.2-3B", "Skywork", "verifier", 0.0, 8192),
}
FRONTIER = "claude-opus-5-5"                        # the yardstick, called through the Anthropic SDK

CREDITS = {"l40s_hour": 0.86, "opus_in_mtok": 4.0, "opus_out_mtok": 20.0}   # one VM hour, a million tokens
BUDGET = {   # per task, by router tier
    1: {"tokens": 2_000, "gpu_s": 4.0, "credits": 0.002},
    2: {"tokens": 12_000, "gpu_s": 18.0, "credits": 0.006},
    3: {"tokens": 40_000, "gpu_s": 60.0, "credits": 0.02},
}
RUN_CREDITS = 400.0                                 # a hard ceiling for one whole evaluation run
SEED = 2026                                         # every split and sample draws from this seed
DATA, RESULTS = Path("data"), Path("results")
GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:9000/v1")   # every model call goes through it
GATEWAY_KEY = os.environ.get("GATEWAY_KEY", "")     # read once, never printed
CLOUD = {"url": os.environ.get("GPU_CLOUD_URL", ""), "key": os.environ.get("GPU_CLOUD_KEY", ""),
         "flavor": "1x-l40s-48g", "region": os.environ.get("GPU_REGION", "region-b")}
