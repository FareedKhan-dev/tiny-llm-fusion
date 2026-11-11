"""platform/episodes.py: every graded candidate, kept for training. A sealed task is refused at the door."""
import json
import sqlite3

from fusion.config import DATA, SEED
from fusion.tools.graders import check

SEALED = set(json.loads((DATA / f"sealed_manifest_seed{SEED}.json").read_text())["ids"])
DB = sqlite3.connect(DATA / "episodes.db")
DB.execute("CREATE TABLE IF NOT EXISTS ep (task TEXT, prompt TEXT, model TEXT, answer TEXT, ok INT, reply TEXT)")


def add(task, cands):
    if task.id in SEALED:                        # the seal guard
        raise PermissionError(f"{task.id} is sealed")
    DB.executemany("INSERT INTO ep VALUES (?, ?, ?, ?, ?, ?)",
                   [(task.id, task.prompt, c.model, c.answer, int(check(task, c.answer)), c.text) for c in cands])
    DB.commit()
