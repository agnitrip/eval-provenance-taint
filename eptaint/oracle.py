"""The immutable external oracle.

The oracle holds the true answer for every episode. It is deliberately kept OUTSIDE the writable
pipeline: it is built once from the generated episodes, exposes read-only access, and is imported
only by the scorer (``metrics.py``). Neither the verifier harness nor the writable eval store
imports this module, so a self-authored value can never overwrite an oracle answer.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from .tasks import Episode


class OracleBook:
    """Read-only lookup of episode_id -> true answer.

    There is no setter. The internal dict is wrapped in a ``MappingProxyType`` so callers cannot
    mutate it, structurally enforcing "held outside the writable pipeline".
    """

    def __init__(self, episodes: Iterable[Episode]) -> None:
        self._answers: Mapping[str, str] = MappingProxyType(
            {ep.id: ep.oracle_answer for ep in episodes}
        )

    def truth(self, episode_id: str) -> str:
        return self._answers[episode_id]

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._answers

    def __len__(self) -> int:
        return len(self._answers)
