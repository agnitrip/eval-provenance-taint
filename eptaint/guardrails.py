"""Provenance / quote-source guardrail helpers.

Provenance is treated as an *emergent* property of the logged value and the authoritative
source, not merely a label copied from the arm. Stage A logs whatever the verifier produced,
then :func:`ground_against_source` decides the tag: a logged value counts as ``source-grounded``
only if it can be validated verbatim against the authoritative source; otherwise it is
``model-inferred``. This means a verifier that logs a wrong value from imperfect context is
correctly tagged ``model-inferred`` with no source quote — the signal the Stage-B guardrail uses.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .tasks import PROV_MODEL_INFERRED, PROV_SOURCE_GROUNDED

_DOC_VALUE = re.compile(r"the answer is (.+?)\.")


def asserted_value(doc: str) -> Optional[str]:
    m = _DOC_VALUE.search(doc)
    return m.group(1).strip() if m else None


def ground_against_source(value: str, authoritative_doc: str) -> Tuple[str, Optional[str]]:
    """Return (provenance, source_quote).

    A value is ``source-grounded`` iff the authoritative source states exactly that value as the
    answer; the returned quote is the verbatim supporting span. Otherwise it is ``model-inferred``.
    """
    src_val = asserted_value(authoritative_doc)
    if src_val is not None and src_val == value.strip():
        quote = f"the answer is {src_val}."
        return PROV_SOURCE_GROUNDED, quote
    return PROV_MODEL_INFERRED, None
