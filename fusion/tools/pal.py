"""tools/pal.py: program-aided reasoning. The model writes a program, and the sandbox does the arithmetic."""
from fusion.engine.pool import ask
from fusion.tools.graders import extract
from fusion.tools.sandbox import run

PAL_SYSTEM = ("Write a Python function solution() that returns the final answer as a number, in one python code "
              "fence. Use only the standard library.")


async def pal_answer(model, task, lane="single"):
    """Ask for a program, run it, and keep what it returns as the answer."""
    msgs = [{"role": "system", "content": PAL_SYSTEM}, {"role": "user", "content": task.prompt}]
    cand = await ask(model, task, 0.0, lane, msgs)
    result = run((extract("code", cand.text) or "") + "\nprint(solution())", timeout=5)
    cand.answer = result.out.strip() if result.ok else None     # a crash or a timeout is a format failure
    return cand
