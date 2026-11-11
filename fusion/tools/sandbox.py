"""tools/sandbox.py: runs code nobody reviewed in its own process, with a time limit and a memory cap."""
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass
class Run:
    ok: bool                                     # exit code 0
    out: str
    err: str
    seconds: float
    killed: bool = False                         # True when the time limit ended it


def run(code, stdin="", timeout=5.0, mem_mb=512):
    """python -I in an empty directory, with a memory cap and no new processes or threads."""
    def cap():                                   # runs in the child, just before Python starts
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], input=stdin, capture_output=True, text=True,
                               timeout=timeout, cwd=tmp, env={"OMP_NUM_THREADS": "1"}, preexec_fn=cap)   # numpy, one thread
            return Run(p.returncode == 0, p.stdout, p.stderr[-2000:], round(time.time() - t0, 2))
        except subprocess.TimeoutExpired:
            return Run(False, "", "timed out", timeout, killed=True)
