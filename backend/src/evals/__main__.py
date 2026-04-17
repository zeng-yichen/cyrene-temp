"""CLI entry point for running evals.

Usage:
    python -m backend.src.evals --agent stelle --cases backend/src/evals/cases/stelle/
    python -m backend.src.evals --agent cyrene
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run Amphoreus agent evals")
    parser.add_argument("--agent", type=str, help="Filter by agent name")
    parser.add_argument("--cases", type=str, default=None, help="Path to cases directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    cases_dir = args.cases
    if not cases_dir:
        base = Path(__file__).parent / "cases"
        if args.agent:
            cases_dir = str(base / args.agent)
        else:
            cases_dir = str(base)

    if not Path(cases_dir).exists():
        print(f"Cases directory not found: {cases_dir}")
        sys.exit(1)

    from backend.src.evals.harness.runner import run_eval
    run = run_eval(cases_dir, agent_filter=args.agent)

    print(f"\nResults: {run.passed_cases}/{run.total_cases} passed ({run.pass_rate:.0%})")
    for result in run.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.case.id}")
        for failure in result.failures:
            print(f"    - {failure}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(run.model_dump(mode="json"), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
