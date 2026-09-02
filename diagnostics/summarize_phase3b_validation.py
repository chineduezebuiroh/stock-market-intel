#!/usr/bin/env python3
"""Summarize completed local Phase 3B validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMBOS = ("stocks_b_wmq_all", "stocks_a_mqy_all")


def summarize(root: Path) -> dict[str, object]:
    decisions: dict[str, str] = {}
    for combo in COMBOS:
        path = root / combo / "coverage_summary.json"
        if not path.exists():
            decisions[combo] = "D — NOT SUPPORTABLE FROM CURRENT IMMUTABLE HISTORY"
            continue
        coverage = json.loads(path.read_text())
        decisions[combo] = coverage["readiness_decision"]

    ready = {combo: value.startswith("A —") for combo, value in decisions.items()}
    if all(ready.values()):
        overall = "both ready"
    elif ready["stocks_b_wmq_all"]:
        overall = "W/M/Q ready, M/Q/Y partial/not ready"
    elif ready["stocks_a_mqy_all"]:
        overall = "M/Q/Y ready, W/M/Q partial/not ready"
    else:
        overall = "neither ready"
    return {"combo_decisions": decisions, "overall_decision": overall}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[OK] wrote {args.output}: {result['overall_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
