"""The writable, provenance-tagged evaluation store.

This is the mutable surface that Stage A writes gold values into and Stage B reads them from.
Every entry carries a provenance tag and (optionally) a validated source quote. The store never
sees oracle values; the only way a value gets in is via :meth:`log_gold`, which records exactly
what the verifier produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .tasks import PROV_MODEL_INFERRED, PROV_SOURCE_GROUNDED


@dataclass
class GoldEntry:
    episode_id: str
    value: str
    provenance: str                 # PROV_MODEL_INFERRED | PROV_SOURCE_GROUNDED
    source_quote: Optional[str]     # verbatim span from the authoritative source, if grounded


class EvalStore:
    def __init__(self) -> None:
        self._entries: Dict[str, GoldEntry] = {}

    def log_gold(self, episode_id: str, value: str, provenance: str,
                 source_quote: Optional[str] = None) -> None:
        assert provenance in (PROV_MODEL_INFERRED, PROV_SOURCE_GROUNDED), provenance
        self._entries[episode_id] = GoldEntry(
            episode_id=episode_id,
            value=value,
            provenance=provenance,
            source_quote=source_quote,
        )

    def get(self, episode_id: str) -> GoldEntry:
        return self._entries[episode_id]

    def has(self, episode_id: str) -> bool:
        return episode_id in self._entries
