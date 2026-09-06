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
    stage_a_doc: str            # what the verifier sees in Stage A (imperfect if tainted; unused if from_self)
    stage_a_asserts: str        # the specific value the stage_a_doc asserts as "the answer"
    candidate_answer: str       # graded in Stage B; set to the oracle answer for the core measure
    arm: str                    # "tainted" | "clean_control" (distractor mode) | "self" (natural mode)
    stage_a_mode: str = "from_source"   # "from_source" (read a doc) | "from_self" (answer unaided)

    def pipeline_view(self) -> Dict[str, str]:
        """Fields the pipeline is allowed to see. Deliberately excludes ``oracle_answer``."""
        return {
            "id": self.id,
            "question": self.question,
            "authoritative_doc": self.authoritative_doc,
            "stage_a_doc": self.stage_a_doc,
            "candidate_answer": self.candidate_answer,
            "arm": self.arm,
            "stage_a_mode": self.stage_a_mode,
        }


def _fmt_doc(question_topic: str, asserted_value: str) -> str:
    return (
        f"Reference note on {question_topic}. According to the recorded reference table, "
        f"the answer is {asserted_value}."
    )


def _templates(rng: random.Random, hard: bool = False) -> List[Dict[str, str]]:
    """Return a list of {template, question, oracle, topic, wrong} dicts for one seed draw.

    ``hard=True`` scales the operands up so that a single-shot unaided answer is error-prone.
    This is used for the natural taint mode, where taint must arise from the verifier's own
    mistakes rather than a planted distractor.
    """
    out: List[Dict[str, str]] = []

    # 1. Modular arithmetic
    if hard:
        a, b, c, m = (rng.randint(400, 1800), rng.randint(60, 300),
                      rng.randint(500, 5000), rng.randint(89, 499))
    else:
        a, b, c, m = (rng.randint(120, 900), rng.randint(23, 99),
                      rng.randint(100, 999), rng.randint(41, 97))
    val = (a * b + c) % m
    out.append({
        "template": "modmath",
        "question": f"Compute ({a} * {b} + {c}) mod {m}.",
        "oracle": str(val),
        "topic": "a modular-arithmetic item",
        "wrong": str((val + rng.randint(1, m - 1)) % m),
    })

    # 2. Unit conversion at a fixed synthetic rate
    if hard:
        x, k = rng.randint(120, 1500), rng.randint(13, 89)
    else:
        x, k = rng.randint(37, 480), rng.randint(3, 19)
    out.append({
        "template": "unit",
        "question": f"Given 1 glimm = {k} sprock, convert {x} glimm to sprock.",
        "oracle": str(x * k),
        "topic": "a unit-conversion item",
        "wrong": str(x * k + rng.choice([-k, k, x, -x])),
    })

    # 3. Base conversion (decimal -> base b)
    n = rng.randint(5000, 90000) if hard else rng.randint(200, 4000)
    base = rng.choice([3, 5, 7])
    digits = _to_base(n, base)
    out.append({
        "template": "base",
        "question": f"Write the decimal number {n} in base {base}.",
        "oracle": digits,
        "topic": "a base-conversion item",
        "wrong": _to_base(n + rng.choice([1, 2, base]), base),
    })

    # 4. Arithmetic sequence nth term
    if hard:
        s, d, nth = rng.randint(20, 300), rng.randint(15, 90), rng.randint(80, 400)
    else:
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


def generate(n: int, seed: int = 7, taint_mode: str = "distractor") -> List[Episode]:
    """Generate ``n`` episodes.

    ``taint_mode="distractor"`` (default): item-matched tainted / clean-control PAIRS. For every
    drawn task we emit BOTH arms from the SAME underlying values (identical question, oracle,
    candidate); the only difference is the Stage-A reference material (distractor vs authoritative).
    Taint is planted.

    ``taint_mode="natural"``: no distractor, no planted reference. Each drawn task becomes ONE
    episode in which the verifier answers the question unaided and its own answer is logged as gold
    (``stage_a_mode="from_self"``). Taint arises only when the verifier makes its own mistake, so
    eligibility equals the verifier's unaided error rate. Tasks are harder so this rate is nonzero.
    """
    if taint_mode == "natural":
        return _generate_natural(n, seed)

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
                    arm=arm,
                ))
                i += 1
    return episodes


def _generate_natural(n: int, seed: int) -> List[Episode]:
    """Natural taint: the verifier answers unaided and its own answer becomes gold.

    Items are a mix of hard and easy: on the hard items the fast, single-shot logging step is
    usually wrong (those episodes self-poison and are eligible for adoption), on the easy items it
    is usually right (those give correct-but-unverified controls for the false-alarm measure).
    """
    rng = random.Random(seed)
    episodes: List[Episode] = []
    i = 0
    while len(episodes) < n:
        hard = (len(episodes) // 4) % 2 == 0   # alternate blocks of hard / easy tasks
        for t in _templates(rng, hard=hard):
            if len(episodes) >= n:
                break
            auth = _fmt_doc(t["topic"], t["oracle"])
            episodes.append(Episode(
                id=f"nat{i:04d}-{t['template']}-{'h' if hard else 'e'}",
                template=t["template"],
                question=t["question"],
                oracle_answer=t["oracle"],
                authoritative_doc=auth,          # used only by source/guardrail conditions + scoring
                stage_a_doc="",                  # unused in from_self mode
                stage_a_asserts="",
                candidate_answer=t["oracle"],    # candidate = the true answer
                arm="self",
                stage_a_mode="from_self",
            ))
            i += 1
    return episodes
