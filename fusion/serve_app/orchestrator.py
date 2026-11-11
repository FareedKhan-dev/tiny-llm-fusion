"""serve_app/orchestrator.py: the whole team behind one OpenAI-compatible model name, fusion-4x3b."""
import asyncio
import time
from dataclasses import replace

from fastapi import FastAPI, Request

from fusion.data import Task
from fusion.engine.moa import moa
from fusion.engine.router import probe, route
from fusion.engine.sampler import sample
from fusion.engine.verifier import best_of_n
from fusion.engine.pool import POOL
from fusion.engine.vote import WEIGHT, tally
from fusion.platform import episodes, gateway
from fusion.platform.governor import Governor
from fusion.tools import frontier
from fusion.tools.retrieval import augment


class FusionEngine:
    def __init__(self, router, calibrator, escape=0.0, store=False, off=()):    # escape: a threshold, 0 is off
        self.router, self.cal, self.escape, self.store, self.gov = router, calibrator, escape, store, Governor()
        self.off = set(off)                      # parts switched off for one ablation row

    async def solve(self, task):
        """(answer, trace): probe, route, sample, aggregate, vote or verify, then a calibrated confidence."""
        if task.family == "knowledge" and "retrieval" not in self.off:
            task = replace(task, prompt=augment(task))
        answer, x = await probe(task)
        tier = 3 if "router" in self.off else route(self.router, x)
        p, trace = float(self.router[1].predict_proba(x[None])[0, 1]), {"tier": tier}
        if tier > 1:
            pal = 2 if tier == 3 and task.family == "math" and "pot" not in self.off else 0
            cands = await sample(task, 2 if tier == 2 else 4, pal, ["qwen"] * 4 if "diversity" in self.off else POOL)
            trace["governor"] = self.gov.charge(task.id, tier, sum(c.tokens for c in cands))
            if tier == 3 and trace["governor"] == "ok" and "moa" not in self.off:
                cands.append(await moa(task, cands))
            answer, margin, shares = tally(task, cands, dict.fromkeys(WEIGHT, 1.0) if "weights" in self.off else WEIGHT)
            if task.family == "code" or margin < 0.3:            # split votes and all code go to the verifier
                pick, _ = await best_of_n(task, cands, tests="tests" not in self.off)
                answer = pick.answer if pick else answer
            p = float(self.cal.predict([shares.get(answer, 0.0)])[0])
            if self.store:
                episodes.add(task, cands)
        if p < self.escape:                                      # the escape hatch: Opus answers instead
            answer = frontier.score(task, (await asyncio.to_thread(frontier.live, task))[0])[0]
        return answer, trace | {"confidence": round(p, 3), "escalated": p < self.escape}

    async def run_stream(self, tasks, width=64):
        """Many tasks, at most width in flight. The training pass and the sealed run use this."""
        gate = asyncio.Semaphore(width)
        async def one(t):
            async with gate:
                return await self.solve(t)
        return await asyncio.gather(*map(one, tasks))


API, ENGINE = FastAPI(), None                    # the public door, on its own port


@API.post("/v1/chat/completions")
async def chat(request: Request):
    """OpenAI-compatible: messages in, an answer and a fusion block out."""
    body = await request.json()
    task = Task(f"api/{time.time_ns()}", "api", body.get("family", "reasoning"), body["messages"][-1]["content"], "")
    answer, trace = await ENGINE.solve(task)
    return {"id": task.id, "object": "chat.completion", "created": int(time.time()), "model": "fusion-4x3b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "fusion": trace}


async def lifecycle(rec, scaler, every=15, quiet_after=600):
    """The control loop: sync the gateway, scale on waiting calls, release everything after ten quiet minutes."""
    last_busy = time.time()
    while True:
        rec.refresh()
        gateway.sync(rec.vms)
        now = time.time()
        if any(gateway.WAITING.values()) or any(r.inflight for rs in gateway.REPLICAS.values() for r in rs):
            last_busy = now
        want = {m: scaler.desired(m, max(len(rs), 1), gateway.WAITING[m], now) for m, rs in gateway.REPLICAS.items()}
        quiet = now - last_busy > quiet_after
        rec.tick({m: 0 for m in want} | {"standby": 0} if quiet else want | {"standby": 1},
                 idle=lambda v: not any(r.inflight for r in gateway.REPLICAS.get(v.model, []) if v.ip in r.url))
        await asyncio.sleep(every)
