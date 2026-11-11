"""eval/harness.py: the sealed run, every arm on the same tasks, every number with an interval."""
import numpy as np
from scipy.stats import binomtest

from fusion.config import SEED
from fusion.tools.graders import check

ACCESS = []                                      # (arm, task id) per sealed read


def grade(arm, tasks, answers):
    """1 or 0 per task for one arm, with every read of the seal logged."""
    ACCESS.extend((arm, t.id) for t in tasks)
    return np.array([check(t, a) for t, a in zip(tasks, answers)], float)


def interval(right, groups, reps=10000, seed=SEED):
    """A 95% bootstrap interval that resamples whole groups, such as MMLU subjects."""
    rng, ids = np.random.default_rng(seed), sorted(set(groups))
    by = {g: right[[i for i, x in enumerate(groups) if x == g]] for g in ids}
    means = [np.concatenate([by[g] for g in rng.choice(ids, len(ids))]).mean() for _ in range(reps)]
    return np.percentile(means, [2.5, 97.5])


def mcnemar(a, b):
    """Exact McNemar p value for two arms, from the tasks where they disagree."""
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0


def gap_closed(single, fused, frontier):
    return (fused - single) / (frontier - single)
