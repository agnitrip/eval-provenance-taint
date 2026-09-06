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
    c = s["conditions"]

    # Every tainted episode is eligible (mock extracts the distractor value).
    assert s["counts"]["eligible_tainted"] == 24
    assert s["eligibility_rate_tainted"] == 1.0

    # Baseline defers to the tainted gold; the ablation conditions each stop adoption.
    assert c["baseline"]["taint_adoption"]["p"] == 1.0
    for cond in ("tag_only", "source_only", "guardrail"):
        assert c[cond]["taint_adoption"]["p"] == 0.0
        assert c[cond]["prevented_adoption"]["p"] == 1.0
    # tag_only and guardrail catch via explicit flag in the mock.
    assert c["tag_only"]["flag_rate"] == 1.0
    assert c["guardrail"]["flag_rate"] == 1.0
    # No false alarms on clean controls in any condition.
    for cond in c:
        assert c[cond]["false_alarm"]["p"] == 0.0
    # Wilson interval sanity: adoption CI brackets the point estimate.
    b = c["baseline"]["taint_adoption"]
    assert b["lo"] <= b["p"] <= b["hi"] and b["n"] == 24
    print("mock ablation OK")


if __name__ == "__main__":
    test_oracle_not_imported_by_pipeline_surfaces()
    test_pipeline_view_excludes_oracle()
    test_mock_end_to_end()
    print("all smoke tests passed")
