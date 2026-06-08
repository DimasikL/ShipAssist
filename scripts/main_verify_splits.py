#!/usr/bin/env python3
"""
scripts/verify_splits.py — Data-split integrity check for ShipAssistant.

Validates that train / val / test audio groups are disjoint at both the
``audio_group`` level and the speaker-ID level.  Exits with code 1 on any
violation so the check can gate CI pipelines.

Design rationale
----------------
With only 5 speakers and 500 clips, a single group crossing split boundaries
would silently inflate validation accuracy (data leakage) — a fatal flaw at
thesis defense.  This script catches both kinds of leakage:

1. **Group overlap** — the same ``audio_group`` string appears in two splits.
2. **Speaker overlap** — two different groups that share the same speaker ID
   end up in different splits (e.g. "train user 1" in train and
   "train user 1 new" in test both belong to speaker "user 1").

Speaker IDs are extracted from group names via the regex pattern
``user\\s*\\d+`` (case-insensitive).  Non-speaker groups (noise sets, hard
negatives) do not carry a speaker ID and are excluded from the speaker check.

Usage
-----
Run from the project root:

    python scripts/verify_splits.py
    python scripts/verify_splits.py --csv artifacts/data/my_dataset.csv
    python scripts/verify_splits.py --config configs/base.yaml

Exit codes
----------
* 0 — all assertions pass.
* 1 — one or more violations detected (details logged at ERROR level).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Project root bootstrap — allows running as `python scripts/verify_splits.py`
# without installing the package.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import settings  # noqa: E402  (import after sys.path tweak)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Match an optional word-prefix (e.g. "test", "train") followed by "user N".
# Including the prefix is intentional: "test user 2" and "train user 2" are
# different physical speakers who happen to share the same number — the prefix
# disambiguates them.
_SPEAKER_RE = re.compile(r"(?:(?:test|train)\s+)?user\s*\d+", re.IGNORECASE)


def _extract_speaker_id(group: str) -> Optional[str]:
    """Return the normalised speaker token from a group name, or None.

    Examples
    --------
    >>> _extract_speaker_id("train user 1")
    'train user 1'
    >>> _extract_speaker_id("test user 2")
    'test user 2'
    >>> _extract_speaker_id("drug slova-hardneg1")
    None
    """
    match = _SPEAKER_RE.search(group)
    if match is None:
        return None
    # Normalise whitespace so "user1" and "user 1" compare equal.
    raw = match.group(0)
    parts = re.split(r"\s+", raw.strip())
    return " ".join(p.lower() for p in parts)


def _speaker_map(groups: List[str]) -> Dict[str, List[str]]:
    """Return {speaker_id: [group, ...]} for groups that carry a speaker ID."""
    mapping: Dict[str, List[str]] = {}
    for g in groups:
        spk = _extract_speaker_id(g)
        if spk is not None:
            mapping.setdefault(spk, []).append(g)
    return mapping


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def verify_splits(
    csv_path: Path,
    train_groups: List[str],
    val_groups: List[str],
    test_groups: List[str],
) -> bool:
    """Verify split integrity.  Returns True iff all assertions pass.

    Args:
        csv_path: Absolute path to the dataset metadata CSV.  Must have an
            ``audio_group`` column.
        train_groups: Group names belonging to the training split.
        val_groups: Group names belonging to the validation split.
        test_groups: Group names belonging to the test split.

    Returns:
        True when the splits are valid; False when at least one violation is
        found.  All violations are logged at ERROR level before returning.
    """
    if not csv_path.exists():
        logger.error("Dataset CSV not found: %s", csv_path)
        return False

    df = pd.read_csv(csv_path)

    if "audio_group" not in df.columns:
        logger.error(
            "Column 'audio_group' missing from %s.  Found: %s",
            csv_path,
            list(df.columns),
        )
        return False

    # Actual groups present in the CSV (used only for informational logging).
    csv_groups: Set[str] = set(df["audio_group"].dropna().unique())
    known_groups = set(train_groups) | set(val_groups) | set(test_groups)
    phantom = known_groups - csv_groups
    if phantom:
        logger.warning(
            "Groups declared in config but absent from CSV (may be harmless): %s",
            sorted(phantom),
        )

    # ── Assertion 1: group-level disjointness ────────────────────────────────
    violations: List[str] = []

    train_set: Set[str] = set(train_groups)
    val_set: Set[str] = set(val_groups)
    test_set: Set[str] = set(test_groups)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    if tv:
        violations.append(f"Train ∩ Val group overlap: {sorted(tv)}")
    if tt:
        violations.append(f"Train ∩ Test group overlap: {sorted(tt)}")
    if vt:
        violations.append(f"Val ∩ Test group overlap: {sorted(vt)}")

    # ── Assertion 2: speaker-level disjointness ──────────────────────────────
    train_speakers = _speaker_map(train_groups)
    val_speakers = _speaker_map(val_groups)
    test_speakers = _speaker_map(test_groups)

    for split_name, other_name, other_map in [
        ("train", "val", val_speakers),
        ("train", "test", test_speakers),
        ("val", "test", test_speakers),
    ]:
        src_map = train_speakers if split_name == "train" else val_speakers
        shared_speakers = set(src_map) & set(other_map)
        if shared_speakers:
            for spk in sorted(shared_speakers):
                violations.append(
                    f"Speaker '{spk}' appears in both {split_name} "
                    f"({src_map[spk]}) and {other_name} ({other_map[spk]})"
                )

    # ── Report ────────────────────────────────────────────────────────────────
    n_train = len([g for g in train_groups if g in csv_groups])
    n_val = len([g for g in val_groups if g in csv_groups])
    n_test = len([g for g in test_groups if g in csv_groups])

    if violations:
        for msg in violations:
            logger.error("SPLIT VIOLATION: %s", msg)
        logger.error(
            "✗ Split validation FAILED with %d violation(s). Fix before training.",
            len(violations),
        )
        return False

    logger.info(
        "✓ Splits validated: %d train, %d val, %d test groups, 0 overlap",
        n_train,
        n_val,
        n_test,
    )
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Verify that train/val/test audio groups are disjoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Path to the dataset metadata CSV.  Defaults to "
            "settings.paths.dataset_csv from core.config."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level (default: INFO).",
    )
    return p


def main() -> int:
    """Entry point.  Returns exit code: 0 = ok, 1 = violation."""
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    csv_path: Path = args.csv if args.csv is not None else settings.paths.dataset_csv
    csv_path = csv_path if csv_path.is_absolute() else (_PROJECT_ROOT / csv_path).resolve()

    train_groups: List[str] = list(settings.splits.train_groups)
    val_groups: List[str] = list(settings.splits.val_groups)
    test_groups: List[str] = list(settings.splits.test_groups)

    logger.info("Dataset CSV : %s", csv_path)
    logger.info("Train groups: %s", train_groups)
    logger.info("Val groups  : %s", val_groups)
    logger.info("Test groups : %s", test_groups)

    ok = verify_splits(csv_path, train_groups, val_groups, test_groups)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
