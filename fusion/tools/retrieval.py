"""tools/retrieval.py: BM25 over Wikipedia, for knowledge questions in the subjects where it helps."""
import bm25s

INDEX = None
SKIP = {"abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics", "formal_logic",
        "high_school_statistics", "college_physics", "econometrics", "moral_scenarios"}   # retrieval hurt these on dev


def build_index(passages):
    global INDEX                                  # built once, about two gigabytes
    INDEX = bm25s.BM25(corpus=passages)
    INDEX.index(bm25s.tokenize(passages, stopwords="en"))


def search(query, k=4):
    docs, _ = INDEX.retrieve(bm25s.tokenize(query, stopwords="en"), k=k)
    return list(docs[0])


def augment(task):
    """The question with four passages in front, or alone in a skipped subject."""
    if task.meta.get("group") in SKIP:
        return task.prompt
    passages = "\n\n".join(search(task.prompt.split("\n(A)")[0]))     # search with the question, not its choices
    return f"Background passages:\n{passages}\n\nQuestion:\n{task.prompt}"
