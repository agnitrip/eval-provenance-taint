# eval-provenance-taint

A self-contained synthetic benchmark for **eval-set self-contamination through provenance taint**: the failure where a
verifier writes its own model-mediated inference into an evaluation log / gold key, and a
downstream gold-dependent decision later consumes that self-authored value **as ground truth**,
within a single pipeline. It needs no adversary.

Paper: *Measuring Eval-Set Self-Contamination: A Synthetic Benchmark for Provenance Taint in On-Device
Verifier Pipelines*, IEEE ICCE 2027. Which result file backs which number: `results/MANIFEST.md`.

The benchmark measures a **real LLM verifier's** decisions against an **immutable external
oracle** that is held outside the writable pipeline. It is fully synthetic — no external dataset,
no proprietary content.

## What it measures

- **taint-adoption rate** — over *eligible* tainted episodes (episodes where the verifier's
  self-authored gold value actually diverges from the oracle), the fraction where the downstream
  grader follows the self-authored value **against the oracle** (rejects a true candidate). This
  is a real grader decision, not a scripted rate.
- **guardrail recall** — fraction of eligible tainted episodes a provenance-tag / quote-source
  guardrail catches (flags the reference or refuses to adopt it).
- **false-alarm rate** — on clean controls (gold correctly source-grounded), the fraction where
  the guardrail wrongly flags/rejects a correct reference.

All claims are bounded to the synthetic pipeline and the models tested. **No real-world
prevalence claim is made or supported by this benchmark.**

## The two-stage pipeline

Each episode is a deterministic, recomputable task (arithmetic / unit-conversion / base-conversion
/ sequence) with an unambiguous answer.

- **Stage A — reference logging (writable).** The verifier is asked to record the reference
  ("gold") answer from provided reference material, as a retrieval-grounded verifier would. In the
  **tainted arm** the material is imperfect (contains a distractor that asserts a specific wrong
  value); the verifier logs a wrong gold, tagged `model-inferred`. In the **clean-control arm** the
  material is authoritative; the verifier logs the correct gold with a validated source quote,
  tagged `source-grounded`.
- **Stage B — grading (the downstream gold-dependent decision).** The verifier grades a candidate
  answer (set to the true oracle answer) against the *stored* gold. Under `baseline` it sees only
  the stored value. Under `guardrail` it also sees the provenance tag and the authoritative source
  and is instructed to distrust a `model-inferred` reference that conflicts with the source.

The **oracle** (`eptaint/oracle.py`) is computed independently and is read only by the scorer
(`eptaint/metrics.py`). The verifier and the writable store never receive oracle values; a guard
asserts oracle values never enter a prompt.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# smoke test, deterministic mock verifier, no API key, NOT scientific:
python run.py --provider mock --n 40 --out results/mock.json

# real run (requires ANTHROPIC_API_KEY):
python run.py --provider anthropic --model claude-haiku-4-5-20251001 --n 60 --out results/haiku.json
```

## Layout

- `eptaint/tasks.py` — synthetic episode generator (matched tainted / clean-control, provenance IDs)
- `eptaint/oracle.py` — immutable external oracle, held outside the writable pipeline
- `eptaint/store.py` — writable provenance-tagged eval store
- `eptaint/verifier.py` — real LLM verifier harness (Anthropic) + deterministic mock
- `eptaint/guardrails.py` — Stage-B prompt variants (baseline vs provenance/quote-source guardrail)
- `eptaint/pipeline.py` — Stage A (log) + Stage B (grade)
- `eptaint/metrics.py` — taint-adoption / guardrail-recall / false-alarm
- `run.py` — CLI
- `tests/test_smoke.py` — mock end-to-end + oracle-isolation assertions

## License

MIT. Synthetic content only.
