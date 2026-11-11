"""engine/router.py: decides how much of the engine a question gets, from one cheap probe."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from fusion.engine.pool import PROMPT
from fusion.platform.gateway import forward
from fusion.tools.graders import extract

COST = {1: 0.0004, 2: 0.0021, 3: 0.0045}         # credits per task: probe, 8 samples, everything
LAMBDA = 6.0                                      # the accuracy one credit must buy, from dev


async def probe(task):
    """Qwen's greedy answer with token log probabilities: tier 1's answer and the router's evidence."""
    body = {"model": "qwen", "messages": [{"role": "user", "content": f"{task.prompt}\n\n{PROMPT[task.family]}"}],
            "temperature": 0.0, "max_tokens": 2048, "logprobs": True}
    choice = (await forward("/v1/chat/completions", body, "probe")).json()["choices"][0]
    lp = [t["logprob"] for t in choice["logprobs"]["content"]]
    answer = extract(task.family, choice["message"]["content"])
    x = [task.family == f for f in ("math", "knowledge", "reasoning", "code")]
    return answer, np.array(x + [len(task.prompt) / 1000, len(lp) / 1000, np.mean(lp), min(lp), answer is None], float)


def fit(X, right):
    """One logistic model per tier: P(tier t is right | probe features), fit on dev."""
    return {t: LogisticRegression(max_iter=1000).fit(X, right[t]) for t in COST}


def route(models, x, lam=LAMBDA):
    """The tier with the best predicted accuracy minus lam times its cost."""
    score = {t: m.predict_proba(x[None])[0, 1] - lam * COST[t] for t, m in models.items()}
    return max(score, key=score.get)
