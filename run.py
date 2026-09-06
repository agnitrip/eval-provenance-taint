"""CLI entry point for the eval-provenance-taint benchmark.

Examples:
    python run.py --provider mock --n 40 --out results/mock.json
    python run.py --provider anthropic --model claude-haiku-4-5-20251001 --n 60 --out results/haiku.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from eptaint.metrics import per_template, summarize
from eptaint.pipeline import run
from eptaint.tasks import generate
from eptaint.verifier import make_verifier


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="eval-provenance-taint synthetic benchmark")
    ap.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--model", default=None, help="model id for --provider anthropic")
    ap.add_argument("--n", type=int, default=40, help="number of episodes")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="write full results JSON here")
    args = ap.parse_args(argv)

    episodes = generate(args.n, seed=args.seed)
    verifier = make_verifier(args.provider, args.model)
    results = run(episodes, verifier)

    summary = summarize(results)
    report = {
        "meta": {
            "provider": verifier.provider,
            "model": getattr(verifier, "model", None),
            "n": args.n,
            "seed": args.seed,
            "verifier_calls": verifier.calls,
            "scientific": verifier.provider != "mock",
            "note": "MOCK run — deterministic stub, NOT a scientific result."
                    if verifier.provider == "mock"
                    else "Real verifier decisions. Claims bounded to this synthetic pipeline/model.",
        },
        "summary": summary,
        "per_template": per_template(results),
        "episodes": [asdict(r) for r in results],
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps({"meta": report["meta"], "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
