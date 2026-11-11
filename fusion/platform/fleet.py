"""platform/fleet.py: the GPU control plane. It rents VMs, keeps one warm, scales them and gives them back."""
import math
import time
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from fusion.config import CLOUD
from fusion.platform.serve import job

API = {"create": ("POST", "/vms"), "get": ("GET", "/vms/{id}"), "delete": ("DELETE", "/vms/{id}")}   # your cloud's paths
STATES = ("requested", "booting", "pulling", "ready", "serving", "draining", "gone")   # "ready": a standby, weights on disk
ASSIGN, REPORTED = {}, {}       # vm id -> its model, vm id -> the last state it reported


class CloudClient:
    """A thin REST client for any GPU cloud with create, get and delete calls."""

    def __init__(self, url=CLOUD["url"], key=CLOUD["key"]):
        self.http = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {key}"}, timeout=60)

    def call(self, op, body=None, **ids):
        method, path = API[op]
        r = self.http.request(method, path.format(**ids), json=body)
        r.raise_for_status()
        return r.json() if r.content else {}

    def create(self, name, user_data):
        return self.call("create", {"name": name, "flavor": CLOUD["flavor"], "region": CLOUD["region"],
                                    "image": "ubuntu-24.04-cuda-13.0", "volume_gb": 200, "user_data": user_data})


@dataclass
class VM:
    id: str
    model: str                                       # a MODELS key, or "standby"
    state: str = "requested"
    ip: str = ""                                     # private address, from the cloud
    events: list = field(default_factory=list)       # (state, seconds since it was requested)
    born: float = field(default_factory=time.time)

    def move(self, state):
        """A VM only ever moves forward through STATES, and every move is timed."""
        if STATES.index(state) <= STATES.index(self.state):
            raise ValueError(f"{self.id}: {self.state} cannot go to {state}")
        self.state = state
        self.events.append((state, round(time.time() - self.born, 1)))


@dataclass
class Autoscaler:
    target: float = 24.0         # requests waiting per replica that we accept
    hi: int = 3                  # at most this many replicas of one model
    up_cool: float = 60.0        # seconds between two scale-outs of a model
    down_cool: float = 300.0     # seconds after any change before a scale-in
    last: dict = field(default_factory=dict)

    def desired(self, model, replicas, waiting, now):
        # the Kubernetes HPA rule, clamped to 1..hi
        want = max(1, min(self.hi, math.ceil(waiting / self.target)))
        since = now - self.last.get(model, float("-inf"))
        if (want > replicas and since >= self.up_cool) or (want < replicas and since >= self.down_cool):
            self.last[model] = now
            return want
        return replicas                              # inside a cooldown we hold what we have


class Reconciler:
    """Desired replicas in, cloud calls out. It ticks every 15 seconds, like a Kubernetes controller."""

    def __init__(self, cloud, boot, max_vms=8):
        self.cloud, self.boot, self.max_vms, self.vms = cloud, boot, max_vms, []

    def rent(self, model):
        vm = VM(self.cloud.create(f"fusion-{model}", self.boot)["id"], model)
        ASSIGN[vm.id] = model                        # the VM fetches this once booted
        self.vms.append(vm)

    def promote(self, model):
        """Hand the warm standby to a model. It serves in under a minute."""
        spare = next((v for v in self.vms if v.model == "standby" and v.state == "ready"), None)
        if spare:
            spare.model, ASSIGN[spare.id] = model, model
            self.rent("standby")                     # and order the next standby straight away
        return spare is not None

    def tick(self, desired, idle):
        for model, n in desired.items():
            live = [v for v in self.vms if v.model == model and v.state in STATES[:5]]
            for _ in range(n - len(live)):
                if sum(v.state != "gone" for v in self.vms) < self.max_vms and not self.promote(model):
                    self.rent(model)
            for v in live[n:]:
                v.move("draining")                   # it finishes the requests it holds, then goes
        for v in self.vms:
            if v.state == "draining" and idle(v):
                self.cloud.call("delete", id=v.id)
                v.move("gone")

    def refresh(self):
        # the cloud says up and where, the VM says the rest
        for v in [v for v in self.vms if v.state in STATES[:4]]:      # only these can still move forward here
            info = self.cloud.call("get", id=v.id)
            v.ip = info.get("ip") or v.ip
            seen = REPORTED.get(v.id) or ("booting" if info.get("status") == "active" else None)
            if seen and STATES.index(seen) > STATES.index(v.state):
                v.move(seen)


control = FastAPI()                              # the two routes a VM calls


@control.get("/control/assign/{vm_id}", response_class=PlainTextResponse)
def assignment(vm_id: str, skip: str = ""):
    model = ASSIGN.get(vm_id)
    if model in (None, skip):
        raise HTTPException(404)                 # the VM's curl retries until it gets one
    return "\n".join([model, *job(model)])       # model, repos and vLLM flags


@control.post("/control/state/{vm_id}/{state}")
def report(vm_id: str, state: str):
    if state in ("pulling", "ready", "serving"):  # what a VM can see from inside
        REPORTED[vm_id] = state
    return {"ok": True}
