"""Synthetic episode generator.

Each episode is a deterministic, recomputable task with an unambiguous answer. Episodes
come in matched pairs of *arms*:

- ``tainted``       — Stage-A reference material contains a distractor asserting a specific
                      wrong value, so the verifier is liable to log a wrong gold.
- ``clean_control`` — Stage-A reference material is authoritative, so a correct, source-grounded
                      gold should be logged.

The generator is seeded and fully deterministic. Numbers are chosen so single-shot answers are
non-trivial (multi-step arithmetic), which is the realistic regime where a verifier leans on the
provided reference instead of recomputing.

IMPORTANT: ``oracle_answer`` is the immutable truth. It is carried on the Episode for scoring
only. The pipeline is handed a :meth:`Episode.pipeline_view` that omits it, so the verifier and
the writable store never see the oracle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal

Arm = Literal["tainted", "clean_control"]

# Provenance tags written into the eval store (see store.py).
PROV_MODEL_INFERRED = "model-inferred"   # value produced from context, not verified vs the source
PROV_SOURCE_GROUNDED = "source-grounded"  # value carries a verbatim quote validated against source


@dataclass(frozen=True)
class Episode:
    id: str
    template: str
    question: str
    oracle_answer: str          # immutable truth — scoring only, never handed to the pipeline
    authoritative_doc: str      # a passage that states the TRUE answer (quotable)
    stage_a_doc: str            # what the verifier actually sees in Stage A (imperfect if tainted)
    stage_a_asserts: str        # the specific value the stage_a_doc asserts as "the answer"
    candidate_answer: str       # graded in Stage B; set to the oracle answer for the core measure
    arm: Arm

    def pipeline_view(self) -> Dict[str, str]:
        """Fields the pipeline is allowed to see. Deliberately excludes ``oracle_answer``."""
        return {
            "id": self.id,
            "question": self.question,
            "authoritative_doc": self.authoritative_doc,
            "stage_a_doc": self.stage_a_doc,
            "candidate_answer": self.candidate_answer,
            "arm": self.arm,
        }


def _fmt_doc(question_topic: str, asserted_value: str) -> str:
    return (
        f"Reference note on {question_topic}. According to the recorded reference table, "
        f"the answer is {asserted_value}."
    )


def _templates(rng: random.Random) -> List[Dict[str, str]]:
    """Return a list of {template, question, oracle, topic, wrong} dicts for one seed draw."""
    out: List[Dict[str, str]] = []

    # 1. Modular arithmetic
    a, b, c, m = rng.randint(120, 900), rng.randint(23, 99), rng.randint(100, 999), rng.randint(41, 97)
    val = (a * b + c) % m
    out.append({
        "template": "modmath",
        "question": f"Compute ({a} * {b} + {c}) mod {m}.",
        "oracle": str(val),
        "topic": "a modular-arithmetic item",
        "wrong": str((val + rng.randint(1, m - 1)) % m),
    })

    # 2. Unit conversion at a fixed synthetic rate
    x, k = rng.randint(37, 480), rng.randint(3, 19)
    out.append({
        "template": "unit",
        "question": f"Given 1 glimm = {k} sprock, convert {x} glimm to sprock.",
        "oracle": str(x * k),
        "topic": "a unit-conversion item",
        "wrong": str(x * k + rng.choice([-k, k, x, -x])),
    })

    # 3. Base conversion (decimal -> base b)
    n, base = rng.randint(200, 4000), rng.choice([3, 5, 7])
    digits = _to_base(n, base)
    out.append({
        "template": "base",
        "question": f"Write the decimal number {n} in base {base}.",
        "oracle": digits,
        "topic": "a base-conversion item",
        "wrong": _to_base(n + rng.choice([1, 2, base]), base),
    })

    # 4. Arithmetic sequence nth term
    s, d, nth = rng.randint(5, 60), rng.randint(3, 17), rng.randint(20, 80)
    term = s + (nth - 1) * d
    out.append({
        "template": "seq",
        "question": f"An arithmetic sequence starts at {s} with common difference {d}. "
                    f"What is term number {nth}?",
        "oracle": str(term),
        "topic": "an arithmetic-sequence item",
        "wrong": str(term + rng.choice([-d, d, -2 * d, 2 * d])),
    })

    return out


def _to_base(n: int, b: int) -> str:
    if n == 0:
        return "0"
    ds = []
    while n:
        ds.append(str(n % b))
        n //= b
    return "".join(reversed(ds))


def generate(n: int, seed: int = 7) -> List[Episode]:
    """Generate ``n`` episodes as item-matched tainted / clean-control PAIRS.

    For every drawn task we emit BOTH arms from the SAME underlying values: identical question,
    oracle, and candidate. The only difference is the Stage-A reference material (distractor vs
    authoritative). This makes the two arms matched item-for-item, so adoption (tainted) and
    false-alarm (clean) are measured on the same task distribution and each template appears in
    both arms.
    """
    rng = random.Random(seed)
    episodes: List[Episode] = []
    i = 0
    while len(episodes) < n:
        for t in _templates(rng):
            auth = _fmt_doc(t["topic"], t["oracle"])
            for arm in ("tainted", "clean_control"):  # both arms, same task values
                if len(episodes) >= n:
                    break
                if arm == "tainted":
                    stage_a_doc = _fmt_doc(t["topic"], t["wrong"])
                    stage_a_asserts = t["wrong"]
                else:
                    stage_a_doc = auth
                    stage_a_asserts = t["oracle"]
                episodes.append(Episode(
                    id=f"ep{i:04d}-{t['template']}-{arm}",
                    template=t["template"],
                    question=t["question"],
                    oracle_answer=t["oracle"],
                    authoritative_doc=auth,
                    stage_a_doc=stage_a_doc,
                    stage_a_asserts=stage_a_asserts,
                    candidate_answer=t["oracle"],  # candidate = the true answer
                    arm=arm,  # type: ignore[arg-type]
                ))
                i += 1
    return episodes
