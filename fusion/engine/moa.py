"""engine/moa.py: Mixture-of-Agents. One model reads the team's answers and writes its own final one."""
from fusion.engine.pool import PROMPT, ask

AGG_SYSTEM = ("Several models answered the question below, and some of them may be wrong. Check each answer, keep "
              "what is right, fix what is wrong, and give one final answer in the required format.")


async def moa(task, cands, aggregator="qwen", k=6):
    """Show the aggregator one reply per distinct answer, most common first, and return its answer."""
    seen, shown = set(), []
    for c in sorted(cands, key=lambda c: -sum(d.answer == c.answer for d in cands)):
        if c.answer not in seen and len(shown) < k:
            seen.add(c.answer)
            shown.append(f"Answer {len(shown) + 1}:\n{c.text[-1500:]}")    # the end of a reply holds its conclusion
    msgs = [{"role": "system", "content": AGG_SYSTEM},
            {"role": "user", "content": task.prompt + "\n\n" + "\n\n".join(shown) + "\n\n" + PROMPT[task.family]}]
    agg = await ask(aggregator, task, 0.0, "fanout", msgs)
    agg.model = "moa"                            # the aggregate gets its own vote weight
    return agg
