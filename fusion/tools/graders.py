"""tools/graders.py: pulls the final answer out of a reply and checks it against the gold."""
import re

from math_verify import parse, verify

from fusion.tools.sandbox import run

FENCE = "`" * 3                                  # a code fence
PATTERNS = {                                     # family -> pattern, the last match wins
    "math": r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}",           # a \boxed{} answer, one level of braces deep
    "knowledge": r"answer is \(?([ABCD])\)?",                  # "The answer is (C)"
    "code": FENCE + r"(?:python)?\n(.*?)" + FENCE,             # the last python fence
    "reasoning": r"answer is (.+?)\.?$",                       # BBH: "So the answer is (B)."
}


def extract(family, text):
    found = re.findall(PATTERNS[family], text.strip(), re.S | re.M)
    return found[-1].strip() if found else None               # None is a format failure


def check(task, answer):
    """Math by value, code by running its tests in the sandbox, everything else by exact text."""
    if answer is None:
        return False
    if task.family == "math":
        return verify(parse(f"${task.gold}$"), parse(f"${answer}$"))   # 1/2, 0.5 and \frac{1}{2} are one value
    if task.family == "code":
        return run(answer + "\n\n" + task.gold, timeout=10).ok        # a code task's gold is its test program
    return answer.lower() == task.gold.strip().lower()
