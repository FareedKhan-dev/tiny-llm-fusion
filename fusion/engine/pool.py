"""engine/pool.py: the four proposers, one call to any of them, and how often they miss together."""
import math
import time
from dataclasses import dataclass
from itertools import combinations

from fusion.platform.gateway import forward
from fusion.platform.telemetry import record
from fusion.tools.graders import extract

POOL = ["qwen", "llama", "gemma", "phi"]          # the reward model only scores
PROMPT = {                                        # one instruction per family, after every question
    "math": "Solve it step by step and put the final answer in \\boxed{}.",
    "knowledge": "Think step by step, then end with: The answer is (X).",
    "reasoning": "Think step by step, then end with: So the answer is <answer>.",
    "code": "Return the complete function in one python code fence.",
}


@dataclass
class Candidate:
    model: str
    text: str
    answer: str | None                           # None for a format failure
    tokens: int                                  # completion tokens, the GPU work it cost
    seconds: float


async def ask(model, task, temperature=0.0, lane="single", messages=None):
    """One reply from one model through the gateway, the engine's only model call."""
    messages = messages or [{"role": "user", "content": f"{task.prompt}\n\n{PROMPT[task.family]}"}]
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 2048}
    t0 = time.perf_counter()
    r = (await forward("/v1/chat/completions", body, lane)).json()
    text = r["choices"][0]["message"]["content"]
    cand = Candidate(model, text, extract(task.family, text), r["usage"]["completion_tokens"], time.perf_counter() - t0)
    record(task.id, cand)                        # one span per call, for cost
    return cand


def overlap(right):
    """rho for every pair of models: the correlation of their misses (right[m] is 0 or 1 per task)."""
    rho = {}
    for a, b in combinations(right, 2):
        x, y = [1 - v for v in right[a]], [1 - v for v in right[b]]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        cov = sum((p - mx) * (q - my) for p, q in zip(x, y)) / len(x)
        rho[a, b] = cov / math.sqrt(mx * (1 - mx) * my * (1 - my))
    return rho


def effective_voters(n, rho):
    return n / (1 + (n - 1) * rho)               # the design effect from survey sampling
