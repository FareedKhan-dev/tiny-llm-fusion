"""data.py: the benchmarks, the sealed test slices and the pools we may tune and train on."""
import hashlib
import json
import random
from dataclasses import dataclass, field

from datasets import get_dataset_config_names, load_dataset

from fusion.config import DATA, SEED


@dataclass
class Task:
    id: str                  # "gsm8k/test/2", the same in every run
    bench: str               # gsm8k, math500, mmlu, humaneval_plus, mbpp_plus or bbh
    family: str              # math, knowledge, code or reasoning
    prompt: str              # the question exactly as the benchmark wrote it
    gold: str                # the reference answer, or the tests for code
    meta: dict = field(default_factory=dict)   # its subject or BBH task, and MBPP id


FAMILY = {"gsm8k": "math", "math500": "math", "mmlu": "knowledge", "humaneval_plus": "code", "mbpp_plus": "code",
          "bbh": "reasoning"}
SOURCES = {   # name -> (repo, configs, split, benchmark, reader)
    "gsm8k": ("openai/gsm8k", ["main"], "test", "gsm8k", "gsm8k"),
    "math500": ("HuggingFaceH4/MATH-500", [None], "test", "math500", "math500"),
    "mmlu": ("cais/mmlu", ["all"], "test", "mmlu", "mmlu"),
    "humaneval_plus": ("evalplus/humanevalplus", [None], "test", "humaneval_plus", "heplus"),
    "mbpp_plus": ("evalplus/mbppplus", [None], "test", "mbpp_plus", "mbppplus"),
    "bbh": ("lukaemon/bbh", "*", "test", "bbh", "bbh"),                 # 27 tasks, one config each
    "gsm8k_train": ("openai/gsm8k", ["main"], "train", "gsm8k", "gsm8k"),
    "math_full": ("EleutherAI/hendrycks_math", "*", "test", "math500", "math"),   # 7 subjects
    "math_train": ("EleutherAI/hendrycks_math", "*", "train", "math500", "math"),
    "mmlu_val": ("cais/mmlu", ["all"], "validation", "mmlu", "mmlu"),
    "mmlu_aux": ("cais/mmlu", ["all"], "auxiliary_train", "mmlu", "mmlu"),
    "mbpp_train": ("google-research-datasets/mbpp", ["full"], "train", "mbpp_plus", "mbpp"),
}

READ = {   # reader -> (prompt, gold, group) from one row
    "gsm8k": lambda r, c: (r["question"], r["answer"].split("####")[-1].strip(), "gsm8k"),   # it ends "#### 70000"
    "math500": lambda r, c: (r["problem"], r["answer"], r["subject"]),
    "math": lambda r, c: (r["problem"], r["solution"], c),        # the grader reads the answer out of \boxed{}
    "mmlu": lambda r, c: (r["question"] + "".join(f"\n({'ABCD'[k]}) {x}" for k, x in enumerate(r["choices"])),
                          "ABCD"[r["answer"]], r["subject"]),
    "heplus": lambda r, c: (r["prompt"], f"{r['test']}\ncheck({r['entry_point']})\n", "code"),     # tests, then the call
    "mbppplus": lambda r, c: (f"{r['prompt']}\n{r['test_list'][0]}", r["test"], "code"),     # one assert shows the name
    "mbpp": lambda r, c: (f"{r['text']}\n{r['test_list'][0]}", "\n".join(r["test_list"]), "code"),
    "bbh": lambda r, c: (r["input"], r["target"], c),
}


def load(name):
    """Every row of one source as Tasks, tagged with the benchmark it serves."""
    repo, configs, split, bench, reader = SOURCES[name]
    configs = get_dataset_config_names(repo) if configs == "*" else configs
    tasks = []
    for c in configs:
        for i, r in enumerate(load_dataset(repo, c, split=split)):
            prompt, gold, group = READ[reader](r, c)
            mbpp = int(str(r["task_id"]).split("/")[-1]) if bench == "mbpp_plus" else None   # "Mbpp/2" and MBPP's 2 alike
            where = c if len(configs) > 1 else split
            tasks.append(Task(f"{name}/{where}/{i}", bench, FAMILY[bench], prompt, gold, {"group": group, "mbpp_id": mbpp}))
    return tasks


SEAL = {"gsm8k": "all", "math500": "all", "mmlu": 20, "humaneval_plus": "all", "mbpp_plus": "all", "bbh": 30}


def seal(tasks, seed=SEED):
    """The sealed test slices, drawn once, and a sha256 over their ids."""
    rng, sealed = random.Random(seed), []
    for bench, rule in SEAL.items():
        pool = [t for t in tasks if t.bench == bench]
        if rule == "all":
            sealed += pool
            continue
        groups = {}                                     # the same number from every subject
        for t in pool:
            groups.setdefault(t.meta["group"], []).append(t)
        sealed += [t for g in sorted(groups) for t in rng.sample(groups[g], rule)]
    ids = sorted(t.id for t in sealed)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    (DATA / f"sealed_manifest_seed{seed}.json").write_text(json.dumps({"sha256": digest, "ids": ids}))
    return sealed, digest


def norm(text):
    return " ".join(text.lower().split())               # case and spacing never hide a repeat


def pools(sealed, candidates):
    """Rows we may tune and train on, minus repeats of a sealed question or MBPP id."""
    sealed_ids, seen_text = {t.id for t in sealed}, {norm(t.prompt) for t in sealed}
    seen_mbpp = {t.meta["mbpp_id"] for t in sealed if t.meta.get("mbpp_id")}
    keep, dropped = [], {"text": 0, "mbpp_id": 0}
    for t in candidates:
        if t.id in sealed_ids:
            continue                                    # a sealed row itself
        if norm(t.prompt) in seen_text:
            dropped["text"] += 1
        elif t.meta.get("mbpp_id") in seen_mbpp:        # MBPP+ reworded prompts but kept ids
            dropped["mbpp_id"] += 1
        else:
            keep.append(t)
    return keep, dropped


DRAW = {   # benchmark -> (dev size, train size, dev split)
    "gsm8k": (400, 3000, "gsm8k_train"), "math500": (400, 2400, "math_full"), "mmlu": (400, 1800, "mmlu_val"),
    "mbpp_plus": (200, 162, "mbpp_train"), "bbh": (405, 1620, "bbh"),
}


def draw(pool, seed=SEED):
    """(dev, train) from the pool, per benchmark and with no task in both."""
    rng, dev, train = random.Random(seed), [], []
    for bench, (d, t, source) in DRAW.items():
        rows = [x for x in pool if x.bench == bench]
        dev += rng.sample([x for x in rows if x.id.startswith(source + "/")], d)
        taken = {x.id for x in dev}
        train += rng.sample([x for x in rows if x.id not in taken], t)
    return dev, train
