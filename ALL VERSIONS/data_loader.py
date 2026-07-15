"""
data_loader.py
==============
Loads and lightly validates candidate records from .jsonl (or .jsonl.gz).
Returns a list of raw dicts; no transformation happens here.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _iter_lines(path: Path) -> Iterator[str]:
    """Yield raw JSON lines regardless of whether the file is gzipped."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with open(path, "r", encoding="utf-8") as fh:
            yield from fh


def load_candidates(path: str | os.PathLike) -> list[dict]:
    """
    Load all candidate records from a JSONL or JSONL.GZ file.

    Parameters
    ----------
    path : str | PathLike
        Path to the candidate file.

    Returns
    -------
    list[dict]
        Parsed candidate records.  Empty lines are skipped silently.

    Raises
    ------
    FileNotFoundError
        If the given path does not exist.
    json.JSONDecodeError
        If a line is not valid JSON (logged + skipped rather than crash).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    candidates: list[dict] = []
    errors = 0
    for i, line in enumerate(_iter_lines(path), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping line %d — bad JSON: %s", i, exc)
            errors += 1

    logger.info(
        "Loaded %d candidates from %s (%d parse errors skipped)",
        len(candidates),
        path.name,
        errors,
    )
    return candidates


def load_job_description(path: str | os.PathLike) -> str:
    """
    Read a job description from a plain-text or markdown file.

    Parameters
    ----------
    path : str | PathLike

    Returns
    -------
    str
        Raw text of the job description.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JD file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    logger.info("Loaded JD from %s (%d chars)", path.name, len(text))
    return text