"""platform/governor.py: soft and hard caps per task, and a ceiling for the run."""
from dataclasses import dataclass, field

from fusion.config import BUDGET, RUN_CREDITS

PER_TOKEN = 1.9e-6                               # credits per token, fully loaded, from dev


@dataclass
class Governor:
    spent: float = 0.0                           # credits charged across the whole run
    tasks: dict = field(default_factory=dict)    # task id -> credits charged so far

    def charge(self, task_id, tier, tokens):
        """'ok', 'soft' past the tier's budget, or 'hard' past twice that budget or the run's ceiling."""
        credits = tokens * PER_TOKEN
        self.tasks[task_id] = self.tasks.get(task_id, 0.0) + credits
        self.spent += credits
        if self.tasks[task_id] > 2 * BUDGET[tier]["credits"] or self.spent > RUN_CREDITS:
            return "hard"                        # return the best answer so far, flagged
        return "soft" if self.tasks[task_id] > BUDGET[tier]["credits"] else "ok"   # soft: finish, start nothing new
