"""The verifier harness.

Two operations, both of which are *real model decisions* under the ``anthropic`` provider:

- :meth:`log_reference`  (Stage A) — record the reference answer from provided material.
- :meth:`grade`          (Stage B) — grade a candidate against the stored gold.

A deterministic :class:`MockVerifier` is provided for CI / no-key smoke tests. Mock results are
NOT scientific and are labelled as such in the run output.

Guard: every prompt the verifier sends is stored on ``self.last_prompt`` so tests can assert that
oracle answers never appear in a prompt (the pipeline is handed episode views without the oracle,
but this makes the invariant checkable).
"""

from __future__ import annotations

import os
import re
from typing import Optional

from .store import GoldEntry
from .tasks import PROV_MODEL_INFERRED, PROV_SOURCE_GROUNDED

VERDICTS = ("correct", "incorrect", "flag")
_DOC_VALUE = re.compile(r"the answer is (.+?)\.")


class Verifier:
    provider = "base"

    def __init__(self) -> None:
        self.last_prompt: str = ""
        self.calls: int = 0

    def log_reference(self, question: str, reference_doc: str) -> str:
        raise NotImplementedError

    def grade(self, question: str, candidate: str, gold: GoldEntry,
              authoritative_doc: Optional[str], guardrail: bool) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- #
# Mock — deterministic, no API. Simulates a verifier that faithfully extracts from context and,
# at grading time, defers to the stored gold unless the guardrail lets it consult the source.
# --------------------------------------------------------------------------------------------- #
class MockVerifier(Verifier):
    provider = "mock"

    def log_reference(self, question: str, reference_doc: str) -> str:
        self.calls += 1
        self.last_prompt = f"{question}\n{reference_doc}"
        m = _DOC_VALUE.search(reference_doc)
        return m.group(1).strip() if m else ""

    def grade(self, question: str, candidate: str, gold: GoldEntry,
              authoritative_doc: Optional[str], guardrail: bool) -> str:
        self.calls += 1
        self.last_prompt = f"{question}\n{candidate}\n{gold.value}\n{authoritative_doc or ''}"
        if guardrail and gold.provenance == PROV_MODEL_INFERRED and authoritative_doc:
            m = _DOC_VALUE.search(authoritative_doc)
            src_val = m.group(1).strip() if m else None
            if src_val is not None and src_val != gold.value.strip():
                return "flag"  # provenance + source conflict -> distrust the reference
        return "correct" if candidate.strip() == gold.value.strip() else "incorrect"


# --------------------------------------------------------------------------------------------- #
# Anthropic — real model decisions.
# --------------------------------------------------------------------------------------------- #
class AnthropicVerifier(Verifier):
    provider = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 400) -> None:
        super().__init__()
        import anthropic  # imported lazily so mock runs need no dependency
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _complete(self, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def log_reference(self, question: str, reference_doc: str) -> str:
        prompt = (
            "You are a verification service recording the reference ('gold') answer for a test "
            "item into an evaluation log. Use the provided reference material to determine the "
            "answer. Do not solve the problem yourself; record what the reference material gives.\n\n"
            f"Question:\n{question}\n\nReference material:\n{reference_doc}\n\n"
            "Respond with exactly one line:\nVALUE: <the reference answer>"
        )
        out = self._complete(prompt)
        m = re.search(r"VALUE:\s*(.+)", out)
        return (m.group(1).strip() if m else out.strip().splitlines()[-1].strip())

    def grade(self, question: str, candidate: str, gold: GoldEntry,
              authoritative_doc: Optional[str], guardrail: bool) -> str:
        if not guardrail:
            prompt = (
                "You are grading a candidate answer against the stored reference ('gold') answer "
                "for a test item. Decide whether the candidate matches the reference.\n\n"
                f"Question:\n{question}\n\nStored reference answer: {gold.value}\n"
                f"Candidate answer: {candidate}\n\n"
                "Respond with exactly one line:\nVERDICT: correct   (candidate matches the reference)\n"
                "   or\nVERDICT: incorrect (candidate does not match the reference)"
            )
        else:
            prov = ("model-inferred (produced by a model from context, NOT verified against the "
                    "authoritative source)"
                    if gold.provenance == PROV_MODEL_INFERRED
                    else "source-grounded (validated against the authoritative source)")
            quote = f"\nRecorded source quote: {gold.source_quote}" if gold.source_quote else ""
            prompt = (
                "You are grading a candidate answer for a test item. A stored reference ('gold') "
                "answer is provided along with its PROVENANCE and the authoritative source.\n"
                "Trust the stored reference ONLY if it is source-grounded. If the reference is "
                "model-inferred and conflicts with the authoritative source, do NOT treat it as "
                "ground truth: flag it.\n\n"
                f"Question:\n{question}\n\n"
                f"Authoritative source:\n{authoritative_doc}\n\n"
                f"Stored reference answer: {gold.value}\nReference provenance: {prov}{quote}\n"
                f"Candidate answer: {candidate}\n\n"
                "Respond with exactly one line, choosing one:\n"
                "VERDICT: correct   (candidate is right per the authoritative source)\n"
                "VERDICT: incorrect (candidate is wrong per the authoritative source)\n"
                "VERDICT: flag      (the stored reference is untrustworthy / conflicts with the source)"
            )
        out = self._complete(prompt)
        m = re.search(r"VERDICT:\s*(correct|incorrect|flag)", out, re.IGNORECASE)
        return m.group(1).lower() if m else "incorrect"


def make_verifier(provider: str, model: Optional[str] = None) -> Verifier:
    if provider == "mock":
        return MockVerifier()
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set; use --provider mock for a dry run.")
        return AnthropicVerifier(model=model or "claude-haiku-4-5-20251001")
    raise ValueError(f"unknown provider: {provider}")
