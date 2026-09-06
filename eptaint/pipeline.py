"""The two-stage evaluation pipeline.

Stage A (reference logging, writable): the verifier records a gold value from Stage-A material;
the value is grounded against the authoritative source to assign a provenance tag, then written
to the writable :class:`EvalStore`.

Stage B (grading, the downstream gold-dependent decision): the verifier grades the candidate
against the *stored* gold, under a ``baseline`` condition and a ``guardrail`` condition.

Oracle isolation is architectural, not a string check. The verifier is only ever handed
:meth:`Episode.pipeline_view` fields, which exclude ``oracle_answer``; the ``OracleBook`` is
imported here for scoring only and is never passed into a verifier call. (Note: the true answer
does legitimately appear inside prompts as the candidate under grading and inside the
authoritative source the guardrail is allowed to consult — the invariant that matters is that the
oracle is never presented to the verifier *labelled as gold*.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .guardrails import ground_against_source
from .oracle import OracleBook
from .store import EvalStore
from .tasks import Episode
from .verifier import CONDITIONS, Verifier


@dataclass
class EpisodeResult:
    episode_id: str
    template: str
    arm: str
    oracle: str
    logged_gold: str
    provenance: str
    eligible: bool                     # tainted arm AND logged_gold != oracle
    verdicts: Dict[str, str] = field(default_factory=dict)  # condition -> correct|incorrect|flag


def run(episodes: List[Episode], verifier: Verifier) -> List[EpisodeResult]:
    oracle = OracleBook(episodes)          # built once, read-only, outside the writable store
    store = EvalStore()
    results: List[EpisodeResult] = []

    for ep in episodes:
        view = ep.pipeline_view()          # no oracle_answer inside

        # --- Stage A: log the gold, either from Stage-A material or from the verifier itself ---
        mode = view.get("stage_a_mode", "from_source")
        logged = verifier.log_reference(view["question"], view["stage_a_doc"], mode=mode)
        if mode == "from_self":
            # a self-authored value is never validated against a source: always model-inferred
            provenance, quote = "model-inferred", None
        else:
            provenance, quote = ground_against_source(logged, view["authoritative_doc"])
        store.log_gold(ep.id, logged, provenance, quote)

        # --- Stage B: grade under each ablation condition ---
        gold = store.get(ep.id)
        verdicts = {
            cond: verifier.grade(view["question"], view["candidate_answer"], gold,
                                 authoritative_doc=view["authoritative_doc"], condition=cond)
            for cond in CONDITIONS
        }

        truth = oracle.truth(ep.id)        # scoring only
        eligible = (ep.arm != "clean_control") and (logged.strip() != truth.strip())

        results.append(EpisodeResult(
            episode_id=ep.id,
            template=ep.template,
            arm=ep.arm,
            oracle=truth,
            logged_gold=logged,
            provenance=provenance,
            eligible=eligible,
            verdicts=verdicts,
        ))
    return results
