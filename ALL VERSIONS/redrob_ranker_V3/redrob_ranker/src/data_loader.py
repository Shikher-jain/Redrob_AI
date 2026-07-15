"""data_loader.py — load candidates from JSONL/JSONL.GZ or JSON, and JD from text."""

from __future__ import annotations
import gzip, json, logging, os
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _iter_lines(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with open(path, "r", encoding="utf-8") as fh:
            yield from fh


def load_candidates(path: str | os.PathLike) -> list[dict]:
    """
    Load candidates from:
    - .jsonl      (one JSON object per line)
    - .jsonl.gz   (gzip-compressed JSONL)
    - .json       (a JSON array — used by sample_candidates.json)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    # Handle JSON array format (sample file)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info("Loaded %d candidates from JSON array %s", len(data), path.name)
            return data
        raise ValueError(f"Expected JSON array in {path}")

    # JSONL / JSONL.GZ
    candidates, errors = [], 0
    for i, line in enumerate(_iter_lines(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("Skipping line %d: %s", i, e)
            errors += 1

    logger.info("Loaded %d candidates (%d errors skipped) from %s",
                len(candidates), errors, path.name)
    return candidates


def load_job_description(path: str | os.PathLike) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JD not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    logger.info("Loaded JD: %d chars from %s", len(text), path.name)
    return text
