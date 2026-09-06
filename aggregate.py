"""Pool per-seed run files for a model and report per-condition metrics.

Usage:
    python aggregate.py results/haiku_ab_s*.json
    python aggregate.py --label "Haiku 4.5" results/haiku_ab_s7.json results/haiku_ab_s17.json ...

Pools all episodes across the given files (each seed contributes fresh synthetic tasks), recomputes
the summary with a Wilson 95% CI over the pooled eligible set, and reports the per-seed point
estimates so run-to-run stability is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
from types import SimpleNamespace

from eptaint.metrics import summarize
from eptaint.verifier import CONDITIONS


def _load_episodes(path):
    d = json.load(open(path))
    return [SimpleNamespace(**e) for e in d["episodes"]], d["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    files = []
    for f in args.files:
        files.extend(sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f])

    pooled, per_seed, model = [], [], None
    for f in files:
        eps, meta = _load_episodes(f)
        model = meta.get("model")
        pooled.extend(eps)
        s = summarize(eps)
        per_seed.append((meta.get("seed"), s))

    label = args.label or model or "model"
    agg = summarize(pooled)
    c = agg["conditions"]
    print(f"\n=== {label}  (pooled n={agg['counts']['total']} over {len(files)} seeds, "
          f"eligible={agg['counts']['eligible_tainted']}, "
          f"clean_sg={agg['counts']['clean_source_grounded']}) ===")
    print(f"{'condition':12s} {'adopt [95% CI]':22s} {'false_alarm [95% CI]':22s} flag  seeds(adopt)")
    for cond in CONDITIONS:
        a = c[cond]["taint_adoption"]; fa = c[cond]["false_alarm"]
        seed_adopts = ",".join(f"{s['conditions'][cond]['taint_adoption']['p']:.2f}"
                               for _, s in per_seed)
        print(f"{cond:12s} {a['p']:.2f} [{a['lo']:.2f},{a['hi']:.2f}]"
              f"{'':6s} {fa['p']:.2f} [{fa['lo']:.2f},{fa['hi']:.2f}]"
              f"{'':6s} {c[cond]['flag_rate']:.2f}  [{seed_adopts}]")


if __name__ == "__main__":
    main()
