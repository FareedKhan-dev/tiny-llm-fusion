"""platform/serve.py: what each GPU VM downloads, and the vllm serve flags it starts with."""
import os
import shlex
from pathlib import Path

from fusion.config import GATEWAY, GATEWAY_KEY, MODELS

BOOT = Path(__file__).with_name("bootstrap.sh")


def boot_script():
    """bootstrap.sh with the control address and two keys written in, as every VM's user_data."""
    text = BOOT.read_text()
    fields = {"@CONTROL@": GATEWAY.removesuffix("/v1"), "@GATEWAY_KEY@": GATEWAY_KEY, "@HF_TOKEN@": os.environ["HF_TOKEN"]}
    for name, value in fields.items():
        text = text.replace(name, shlex.quote(value))
    return text


def job(model):
    """(repos, flags) for one VM. A standby downloads every model and waits to be promoted."""
    if model == "standby":
        return " ".join(s.id for s in MODELS.values()), ""
    return MODELS[model].id, " ".join(launch_args(model))


def launch_args(name):
    """The vllm serve flags for one model and its role."""
    spec = MODELS[name]
    args = [spec.id, "--served-model-name", name, "--max-model-len", str(spec.max_len), "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching", "--gpu-memory-utilization", "0.90"]
    if spec.lora:                                # slots for the DPO adapters
        args += ["--enable-lora", "--max-lora-rank", "16", "--max-loras", "2"]
    if "aggregator" in spec.role:                # an aggregator copies its prompt
        args += ["--speculative-config", '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}']
    if spec.role == "verifier":                  # the reward model only scores
        args += ["--runner", "pooling"]
    return args
