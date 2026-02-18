"""
Audit speaker-ID overlaps across SVD pathology folders.
i.e how many speaker IDs appear in multiple pathology folders?

This script scans folders using the convention:

    data/{pathology}/{speaker_id}

Only numeric `speaker_id` folder names are considered valid speaker IDs.

CLI usage:
    python audit_speaker_overlaps.py
    python audit_speaker_overlaps.py --data-root data/raw
    python audit_speaker_overlaps.py --show-limit 100
    python audit_speaker_overlaps.py --json-out reports/overlap_audit.json

Typical workflow:
1) Run once with defaults to print a quick overlap summary.
2) Re-run with `--json-out` to save a machine-readable report for downstream
   checks (e.g., choosing single-label vs multi-label training strategy).

Output includes:
- Number of pathologies scanned
- Global unique speaker ID count
- Number/list of overlapping speaker IDs
- Pairwise pathology overlap counts
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set


def get_pathology_dirs(data_root: Path) -> List[Path]:
    """Return pathology folders under data_root (directories only)."""
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
    )


def get_speaker_ids(pathology_dir: Path) -> Set[str]:
    """
    Collect speaker IDs from `data/{pathology}/{speaker_id}` folder names.

    Only numeric directory names are treated as speaker IDs.
    """
    speaker_ids: Set[str] = set()
    for child in pathology_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            speaker_ids.add(child.name)
    return speaker_ids


def build_index(data_root: Path) -> Dict[str, Set[str]]:
    """Map pathology -> set of speaker IDs."""
    index: Dict[str, Set[str]] = {}
    for pathology_dir in get_pathology_dirs(data_root):
        index[pathology_dir.name] = get_speaker_ids(pathology_dir)
    return index


def build_membership(index: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Map speaker_id -> sorted list of pathologies containing that speaker."""
    membership: Dict[str, List[str]] = defaultdict(list)
    for pathology, speaker_ids in index.items():
        for speaker_id in speaker_ids:
            membership[speaker_id].append(pathology)

    for speaker_id in membership:
        membership[speaker_id].sort(key=str.lower)

    return dict(sorted(membership.items(), key=lambda kv: int(kv[0])))


def pairwise_intersections(index: Dict[str, Set[str]]) -> List[dict]:
    """Compute pairwise overlap counts and sample IDs."""
    rows: List[dict] = []
    for a, b in combinations(sorted(index.keys(), key=str.lower), 2):
        overlap = sorted(index[a] & index[b], key=int)
        rows.append(
            {
                "pathology_a": a,
                "pathology_b": b,
                "overlap_count": len(overlap),
                "sample_speaker_ids": overlap[:20],
            }
        )
    return rows


def print_report(index: Dict[str, Set[str]], show_limit: int) -> dict:
    membership = build_membership(index)
    overlapping = {sid: paths for sid, paths in membership.items() if len(paths) > 1}

    print("=" * 72)
    print("SVD Speaker Overlap Audit (folder-based)")
    print("Rule: scan data/{pathology}/{speaker_id}, numeric folder names only")
    print("=" * 72)

    print(f"Pathologies scanned: {len(index)}")
    print(f"Unique speaker IDs (global): {len(membership)}")
    print(f"Speaker IDs appearing in >1 pathology: {len(overlapping)}")
    print()

    print("Speakers per pathology:")
    for pathology in sorted(index.keys(), key=str.lower):
        print(f"- {pathology}: {len(index[pathology])}")
    print()

    if overlapping:
        print(f"Overlapping speaker IDs (showing up to {show_limit}):")
        shown = 0
        for speaker_id, pathologies in sorted(
            overlapping.items(), key=lambda kv: int(kv[0])
        ):
            print(f"- {speaker_id}: {', '.join(pathologies)}")
            shown += 1
            if shown >= show_limit:
                remaining = len(overlapping) - shown
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break
        print()
    else:
        print("No overlaps found across pathology folders.\n")

    pairwise = pairwise_intersections(index)
    pairwise_nonzero = [row for row in pairwise if row["overlap_count"] > 0]

    if pairwise_nonzero:
        print("Pairwise overlaps (non-zero only):")
        for row in sorted(
            pairwise_nonzero, key=lambda r: r["overlap_count"], reverse=True
        ):
            print(
                f"- {row['pathology_a']} <-> {row['pathology_b']}: "
                f"{row['overlap_count']} (sample: {row['sample_speaker_ids']})"
            )
    else:
        print("No non-zero pairwise overlaps.")

    return {
        "pathologies_scanned": len(index),
        "unique_speaker_ids": len(membership),
        "overlapping_speaker_ids_count": len(overlapping),
        "speakers_per_pathology": {
            p: len(sids)
            for p, sids in sorted(index.items(), key=lambda kv: kv[0].lower())
        },
        "overlapping_speakers": overlapping,
        "pairwise_overlaps": pairwise,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit speaker ID overlaps across data/{pathology}/{speaker_id} folders."
        ),
        epilog=(
            "Examples:\n"
            "  python audit_speaker_overlaps.py\n"
            "  python audit_speaker_overlaps.py --data-root data/raw\n"
            "  python audit_speaker_overlaps.py --show-limit 100\n"
            "  python audit_speaker_overlaps.py --json-out reports/overlap_audit.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Path to dataset root (default: data/raw)",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=50,
        help="Max overlapping speaker IDs to print (default: 50)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full audit report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()

    if not data_root.exists() or not data_root.is_dir():
        raise SystemExit(f"Invalid data root: {data_root}")

    index = build_index(data_root)
    report = print_report(index=index, show_limit=max(args.show_limit, 1))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote JSON report: {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
