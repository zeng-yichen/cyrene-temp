#!/usr/bin/env python3
"""
ICP Engager Report Generator — Phase 1 (LinkedIn CSV input).

Reads a LinkedIn analytics export CSV, classifies each engager against the
client's ICP definition, updates the per-client engager history store, and
writes a two-chart PNG: a distribution pie chart and an ICP-by-segment bar chart.

Usage
-----
    python scripts/generate_engager_report.py \\
        --company example-client \\
        --csv ~/Downloads/linkedin_reactions_export.csv \\
        --output ./output/hume_engager_report.png \\
        --window-days 30

Optional flags
--------------
    --post-id <ID>      Tag every engager with this LinkedIn post ID / URN.
                        Can be specified multiple times.
    --no-history        Skip updating the engager history store (read-only run).
    --verbose           Enable DEBUG-level logging.

CSV format
----------
LinkedIn exports reactions and comments as separate CSVs. Both are supported.
Expected columns (case-insensitive, any order):

  Reactions:  Name, Headline, Company, Reaction Type, Date
  Comments:   Commenter Name, Commenter Headline, Comment, Date

Unknown columns are silently ignored.  Rows without a name are skipped.

ICP Segment definition (optional)
----------------------------------
To get segment-level breakdowns in the bar chart, add a ``segments`` list to
``memory/{company}/icp_definition.json``::

    {
      "description": "...",
      "anti_description": "...",
      "segments": [
        {"label": "AI GTM/Ecosystem",  "description": "GTM, ecosystem, partnerships, or devrel roles at AI companies"},
        {"label": "AI Inference",       "description": "Engineers working on LLM inference, model serving, or GPU infrastructure"},
        {"label": "Voice AI Builders",  "description": "Engineers or researchers actively building voice AI, TTS, ASR, or audio AI products"}
      ]
    }

Each segment needs a ``label`` (the display name) and a ``description`` (plain-English
definition the model reasons from). No keyword lists — the LLM reads the description
directly. Without ``segments``, all ICP matches fall into a single "ICP Match" bucket.

History store
-------------
Engager records are persisted to ``backend/data/{company}/engagers.json``.
The "Orbit" classification is derived from this store: engagers who appeared
in a previous run but don't hit the ICP threshold this time.
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure project root is importable ─────────────────────────────────────────
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_engager_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--company",
        required=True,
        metavar="SLUG",
        help="Client slug, e.g. example-client",
    )
    p.add_argument(
        "--csv",
        required=True,
        metavar="PATH",
        help="Path to the LinkedIn analytics export CSV",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the output PNG",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Only include engagers whose row date is within the last N days. "
            "0 = include all rows (default)."
        ),
    )
    p.add_argument(
        "--post-id",
        default="",
        dest="post_id",
        metavar="ID",
        help=(
            "Ordinal post UUID for the post this CSV covers "
            "(e.g. b72c38a0-46d8-427d-9679-83ec0b518860). "
            "When provided, writes the CSV-derived ICP signal into RuanMei, "
            "closing the loop with the learning system."
        ),
    )
    p.add_argument(
        "--linkedin-url",
        default="",
        dest="linkedin_url",
        metavar="URL",
        help="LinkedIn post URL — stored on the observation alongside the ICP signal.",
    )
    p.add_argument(
        "--no-history",
        action="store_true",
        help="Skip updating the engager history store (dry-run / read-only).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("generate_engager_report")

    # ── Validate inputs ──────────────────────────────────────────────────────
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        log.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    memory_dir = _project_root / "memory" / args.company
    if not memory_dir.exists():
        log.error("No memory directory found for company %r — does the client exist?", args.company)
        sys.exit(1)

    icp_path = memory_dir / "icp_definition.json"
    if not icp_path.exists():
        log.warning(
            "No ICP definition at %s — all engagers will be classified as non_icp. "
            "Create icp_definition.json to enable ICP scoring.",
            icp_path,
        )

    # ── Run pipeline ─────────────────────────────────────────────────────────
    from backend.src.utils.engager_report import (
        parse_csv,
        load_history,
        history_names as get_history_names,
        classify_engagers,
        update_history,
        bridge_to_ruan_mei,
        generate_charts,
        save_report,
    )
    import matplotlib.pyplot as plt

    log.info("Parsing CSV: %s", csv_path)
    rows = parse_csv(csv_path, window_days=args.window_days)
    log.info("  → %d engager rows loaded", len(rows))

    if not rows:
        log.warning("No rows found in CSV. Check the file format and --window-days setting.")
        sys.exit(0)

    log.info("Loading history for %s …", args.company)
    store = load_history(args.company)
    known_names = get_history_names(store)
    log.info("  → %d previously seen engagers", len(known_names))

    log.info("Classifying %d engagers via ICP scorer …", len(rows))
    classified = classify_engagers(rows, args.company, known_names)

    # ── Summary table ────────────────────────────────────────────────────────
    counts = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
    segment_counts: dict[str, int] = {}
    for ce in classified:
        counts[ce.classification] += 1
        if ce.classification == "icp_match" and ce.segment:
            segment_counts[ce.segment] = segment_counts.get(ce.segment, 0) + 1

    total = len(classified)
    signal_total = counts["icp_match"] + counts["non_icp"] + counts["orbit"]

    print()
    print(f"  ── Engager Classification Summary ({'last ' + str(args.window_days) + 'd' if args.window_days else 'all time'}) ──")
    print(f"  Total engagers   :  {total}")
    print(f"  ICP Match        :  {counts['icp_match']}  ({_pct(counts['icp_match'], total)})")
    print(f"  Non-ICP          :  {counts['non_icp']}  ({_pct(counts['non_icp'], total)})")
    print(f"  Internal         :  {counts['internal']}  ({_pct(counts['internal'], total)})")
    print(f"  Orbit            :  {counts['orbit']}  ({_pct(counts['orbit'], total)})")
    if segment_counts:
        print()
        print("  ICP Segments:")
        for seg, cnt in sorted(segment_counts.items(), key=lambda x: -x[1]):
            print(f"    {seg:<30s}  {cnt}")
    print()

    # ── History update ───────────────────────────────────────────────────────
    if not args.no_history:
        log.info("Updating engager history store …")
        update_history(args.company, classified, [args.post_id] if args.post_id else [])
    else:
        log.info("--no-history set: skipping history update.")

    # ── Learning system bridge ───────────────────────────────────────────────
    if args.post_id:
        log.info("Bridging ICP signal to learning system (post %s…) …", args.post_id[:12])
        icp_rate = counts["icp_match"] / signal_total if signal_total else 0.0
        bridged = bridge_to_ruan_mei(
            args.company,
            classified,
            args.post_id,
            linkedin_post_url=args.linkedin_url,
        )
        if bridged:
            print(f"  ✓ Learning system updated: icp_match_rate={icp_rate:.1%} written to RuanMei")
        else:
            print(f"  ✗ Learning system bridge failed — no scored observation found for post ID {args.post_id!r}")
            print(f"    Check that the post has been scored (status=scored) in RuanMei for {args.company}")
    else:
        print("  ℹ  Pass --post-id <ordinal_uuid> to write ICP signal into the learning system.")
    print()

    # ── Chart ────────────────────────────────────────────────────────────────
    log.info("Generating charts …")
    window_label = f"(last {args.window_days}d)" if args.window_days else ""
    fig = generate_charts(classified, args.company, title_suffix=window_label)

    out_path = save_report(fig, args.output)
    plt.close(fig)

    log.info("✓ Report saved → %s", out_path)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{100 * n / total:.1f}%"


if __name__ == "__main__":
    main()
