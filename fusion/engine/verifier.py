"""engine/verifier.py: picks among candidates. Code must pass the question's own examples, then the reward model ranks."""
import asyncio
import re

from fusion.platform.gateway import forward
from fusion.tools.sandbox import run


async def reward(task, cand):
    """The reward model's logit for one candidate, from vLLM's pooling route. Higher is better."""
    msgs = [{"role": "user", "content": task.prompt}, {"role": "assistant", "content": cand.text}]
    return (await forward("/pooling", {"model": "reward", "messages": msgs}, "fanout")).json()["data"][0]["data"][0]


def examples(task):
    """A code question's shown examples as asserts: doctest lines, or MBPP's given assert."""
    pairs = re.findall(r">>> (.+)\n\s*(.+)", task.prompt)
    return [f"assert {call} == {want}" for call, want in pairs] + re.findall(r"^assert .+$", task.prompt, re.M)


async def best_of_n(task, cands, tests=True):
    """(candidate, logit). Code that fails a shown example is out, unless every candidate fails."""
    pool = [c for c in cands if c.answer is not None]
    if task.family == "code" and tests:
        pool = [c for c in pool if run(c.answer + "\n" + "\n".join(examples(task)), timeout=10).ok] or pool
    if not pool:
        return None, float("-inf")
    logits = await asyncio.gather(*[reward(task, c) for c in pool])
    return max(zip(pool, logits), key=lambda x: x[1])
