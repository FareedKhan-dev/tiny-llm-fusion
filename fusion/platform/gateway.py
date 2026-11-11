"""platform/gateway.py: one OpenAI-compatible door to every replica, with a reserved lane and a circuit breaker."""
import asyncio
import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fusion.config import GATEWAY_KEY, MODELS
from fusion.platform.fleet import control

SLOTS, RESERVED = 48, 12        # calls per replica, and the share bulk calls never take
BULK = ("fanout", "batch")      # lanes that leave RESERVED slots free
HTTP = httpx.AsyncClient(timeout=180, headers={"Authorization": f"Bearer {GATEWAY_KEY}"})
REPLICAS, WAITING = {m: [] for m in MODELS}, {m: 0 for m in MODELS}   # WAITING is what the autoscaler reads
ADAPTER = {}                    # model -> its DPO adapter, once promoted


@dataclass
class Replica:
    model: str
    url: str                                         # http://10.0.3.17:8000
    inflight: int = 0
    fails: int = 0                                   # failures in a row
    down_until: float = 0.0                          # an open breaker skips it until then
    every: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(SLOTS))
    bulk: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(SLOTS - RESERVED))


def sync(vms):                                       # mirror the reconciler: serving VMs in, departed VMs out
    for m in REPLICAS:
        live = {f"http://{v.ip}:8000" for v in vms if v.model == m and v.state == "serving"}
        kept = [r for r in REPLICAS[m] if r.url in live]
        REPLICAS[m] = kept + [Replica(m, u) for u in sorted(live - {r.url for r in kept})]


def pick(model, prefix, tried=()):                   # one prefix, one home replica, unless busy
    up = [r for r in REPLICAS[model] if r.down_until < time.time() and r not in tried] or REPLICAS[model]
    home = max(up, key=lambda r: hashlib.sha1((r.url + prefix).encode()).digest())    # rendezvous hashing
    least = min(up, key=lambda r: r.inflight)
    return home if home.inflight < least.inflight + 8 else least


async def forward(path, body, lane="single"):
    """One call. Bulk lanes leave RESERVED slots free, and three failures open the breaker."""
    tried = []
    for _ in range(3):                                   # retry on an untried replica
        r = pick(body["model"], str(body.get("messages") or body.get("input"))[:400], tried)
        tried.append(r)
        WAITING[r.model] += 1
        async with (r.bulk if lane in BULK else nullcontext()), r.every:
            WAITING[r.model] -= 1
            r.inflight += 1
            try:
                resp = await HTTP.post(r.url + path, json={**body, "model": ADAPTER.get(r.model, r.model)})
                if resp.status_code < 500:               # a 4xx is our mistake, so no retry
                    r.fails = 0
                    return resp
            except httpx.TransportError:
                pass
            finally:
                r.inflight -= 1
            r.fails += 1
            r.down_until = time.time() + 30 if r.fails >= 3 else r.down_until   # open for 30 s, then try it again
    raise HTTPException(503, f"{body['model']}: every attempt failed")


app = FastAPI()
app.include_router(control.router)                   # the VMs' control routes


@app.post("/{path:path}")
async def proxy(path: str, request: Request):
    """The OpenAI-compatible door. An X-Lane header picks the lane, and single is the default."""
    if request.headers.get("authorization") != f"Bearer {GATEWAY_KEY}":
        raise HTTPException(401)
    resp = await forward("/" + path, await request.json(), request.headers.get("x-lane", "single"))
    return JSONResponse(resp.json(), resp.status_code)
