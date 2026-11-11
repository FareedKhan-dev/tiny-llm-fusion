"""engine/cache.py: answers we already paid for, replayed free when a call repeats."""
import hashlib
import json
import shelve

CACHE = shelve.open("data/result_cache")          # on disk, warm after a restart


async def cached(call, model, messages, temperature, i):
    """The stored candidate for this exact call, or a fresh one, stored on the way out."""
    key = hashlib.sha256(json.dumps([model, messages, temperature, i], sort_keys=True).encode()).hexdigest()
    if key not in CACHE:
        CACHE[key] = await call()
    return CACHE[key]
