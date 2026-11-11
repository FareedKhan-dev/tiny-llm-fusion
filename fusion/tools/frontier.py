"""tools/frontier.py: the yardstick. Claude Opus 5.5 answers the same prompts, priced in credits."""
import time

import anthropic

from fusion.config import CREDITS, FRONTIER
from fusion.engine.pool import PROMPT
from fusion.tools.graders import extract

CLIENT = anthropic.Anthropic()                   # reads ANTHROPIC_API_KEY


def params(task):
    # medium effort, set explicitly. No fallbacks: a refusal scores as wrong
    return {"model": FRONTIER, "max_tokens": 16000, "output_config": {"effort": "medium"},
            "messages": [{"role": "user", "content": f"{task.prompt}\n\n{PROMPT[task.family]}"}]}


def score(task, msg):
    """(answer, credits, refused) for one finished message, at the live price."""
    text = "".join(b.text for b in msg.content if b.type == "text")
    usage = msg.usage.input_tokens * CREDITS["opus_in_mtok"] + msg.usage.output_tokens * CREDITS["opus_out_mtok"]
    return extract(task.family, text), round(usage / 1e6, 6), msg.stop_reason == "refusal"


def live(task):
    t0 = time.perf_counter()                     # one timed call, for the latency sample
    return CLIENT.messages.create(**params(task)), time.perf_counter() - t0


def batch(tasks, poll=60):
    """One Message Batch at half the price. Returns {task id: message}, None where it failed."""
    reqs = [{"custom_id": t.id.replace("/", "--"), "params": params(t)} for t in tasks]   # ids allow no "/"
    b = CLIENT.messages.batches.create(requests=reqs)
    while CLIENT.messages.batches.retrieve(b.id).processing_status != "ended":
        time.sleep(poll)
    return {r.custom_id.replace("--", "/"): r.result.message if r.result.type == "succeeded" else None
            for r in CLIENT.messages.batches.results(b.id)}
