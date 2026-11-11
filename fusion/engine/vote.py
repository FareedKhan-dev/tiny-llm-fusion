"""engine/vote.py: a weighted vote over candidates, after answers that mean the same thing are merged."""
from math_verify import parse, verify

WEIGHT = {"qwen": 1.33, "llama": 0.43, "gemma": 0.59, "phi": 0.91, "moa": 1.95}   # log odds of dev accuracy


def same(task, a, b):
    """Math answers match by value, so 14/3 and 4.6667 are one answer. The rest by text."""
    if task.family == "math":
        return verify(parse(f"${a}$"), parse(f"${b}$"))
    return a.strip().lower() == b.strip().lower()


def tally(task, cands, weight=WEIGHT):
    """(winner, margin, shares): the heaviest answer, its lead over the runner-up, every share."""
    groups = []                                  # [answer as first seen, total weight]
    for c in (c for c in cands if c.answer is not None):
        g = next((g for g in groups if same(task, g[0], c.answer)), None)
        if g:
            g[1] += weight.get(c.model, 1.0)
        else:
            groups.append([c.answer, weight.get(c.model, 1.0)])
    if not groups:
        return None, 0.0, {}
    total, groups = sum(w for _, w in groups), sorted(groups, key=lambda g: -g[1])
    shares = {a: w / total for a, w in groups}
    return groups[0][0], shares[groups[0][0]] - (groups[1][1] / total if len(groups) > 1 else 0.0), shares
