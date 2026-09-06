"""End-to-end smoke test with the deterministic mock verifier + oracle-isolation checks.

Run: python -m pytest -q   (or: python tests/test_smoke.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eptaint import oracle as oracle_mod       # noqa: E402
from eptaint import store as store_mod          # noqa: E402
from eptaint import verifier as verifier_mod    # noqa: E402
from eptaint.metrics import summarize           # noqa: E402
from eptaint.pipeline import run                 # noqa: E402
from eptaint.tasks import generate               # noqa: E402


def test_oracle_not_imported_by_pipeline_surfaces():
    # Structural isolation: neither the verifier nor the store module imports the oracle.
    assert "OracleBook" not in dir(verifier_mod)
    assert "OracleBook" not in dir(store_mod)
    assert hasattr(oracle_mod, "OracleBook")


def test_pipeline_view_excludes_oracle():
    ep = generate(2, seed=1)[0]
    assert "oracle_answer" not in ep.pipeline_view()


def test_mock_end_to_end():
    episodes = generate(48, seed=7)
    v = verifier_mod.MockVerifier()
    results = run(episodes, v)
    s = summarize(results)

    # The mock faithfully extracts the (wrong) distractor value on tainted episodes, so every
    # tainted episode is eligible, and the baseline grader defers to the tainted gold.
    assert s["counts"]["eligible_tainted"] > 0
    assert s["eligibility_rate_tainted"] == 1.0
    assert s["taint_adoption_rate_baseline"] == 1.0          # baseline adopts taint
    assert s["taint_adoption_rate_guardrail"] == 0.0         # guardrail stops adoption
    assert s["guardrail_recall"] == 1.0                      # prevents adoption on every eligible case
    assert s["guardrail_flag_rate"] == 1.0                   # mock catches via explicit flag
    assert s["false_alarm_rate"] == 0.0                      # no false alarms on clean controls
    assert s["clean_baseline_correct_rate"] == 1.0
    print("mock summary OK:", s)


if __name__ == "__main__":
    test_oracle_not_imported_by_pipeline_surfaces()
    test_pipeline_view_excludes_oracle()
    test_mock_end_to_end()
    print("all smoke tests passed")
