"""platform/telemetry.py: who used the GPUs, by the tokens each task used."""
from collections import defaultdict

from fusion.config import CREDITS

SPANS = []                                       # (task id, model, tokens, seconds) per call


def record(task_id, cand):
    SPANS.append((task_id, cand.model, cand.tokens, cand.seconds))


def attribute(rented_s):
    """task id -> credits, idle time included. The shares add up to what the fleet cost."""
    used = defaultdict(int)
    for task_id, _, tokens, _ in SPANS:
        used[task_id] += tokens
    total, rate = sum(used.values()), CREDITS["l40s_hour"] / 3600
    return {task_id: rented_s * rate * n / total for task_id, n in used.items()}
