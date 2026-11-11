"""engine/sampler.py: one question to every proposer at once, with a repair call for a lost answer."""
import asyncio
from fusion.config import MODELS
from fusion.engine.cache import cached
from fusion.engine.pool import POOL, PROMPT, ask
from fusion.platform.gateway import ADAPTER
from fusion.tools.pal import pal_answer

REPAIR = "Give only the final answer, in the required format."


async def one(model, task, i):                   # sample i of one model, cached
    first = [{"role": "user", "content": f"{task.prompt}\n\n{PROMPT[task.family]}"}]
    async def fresh():
        c = await ask(model, task, MODELS[model].temperature, "fanout")
        if c.answer is None:                     # a follow-up is cheaper than a lost sample
            again = first + [{"role": "assistant", "content": c.text}, {"role": "user", "content": REPAIR}]
            fix = await ask(model, task, 0.0, "fanout", again)
            c.answer, c.tokens = fix.answer, c.tokens + fix.tokens
        return c
    return await cached(fresh, ADAPTER.get(model, model), first, MODELS[model].temperature, i)   # an adapter is a new model


async def sample(task, per_model=4, pal=0, models=POOL):   # per_model samples from each model, plus pal programs
    calls = [one(m, task, k * per_model + i) for k, m in enumerate(models) for i in range(per_model)]
    return await asyncio.gather(*calls, *[pal_answer(m, task, "fanout") for m in POOL[:pal]])
