"""utils.py — shared helpers used across all modules."""

from __future__ import annotations
import logging
import re
from datetime import date, datetime
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def norm(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(val)))


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def today() -> date:
    return datetime.utcnow().date()


def days_since(d: Optional[date]) -> int:
    if d is None:
        return 9999
    return max(0, (today() - d).days)


def contains_any(text: str, tokens: set) -> bool:
    t = norm(text)
    return any(tok in t for tok in tokens)


def candidate_text_blob(c: dict) -> str:
    """
    Flatten a candidate into one text string for embedding.

    Weighting strategy:
    - Career descriptions repeated 2× (richest signal of actual work done)
    - Headline + summary (self-described fit)
    - Skills list
    - Education field of study
    We deliberately exclude company names to avoid brand-matching bias.
    """
    parts: list[str] = []

    p = c.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(f"Current role: {p.get('current_title', '')}")

    for role in c.get("career_history", []):
        title = role.get("title", "")
        desc  = role.get("description", "")
        parts.append(f"{title}: {desc}")
        parts.append(desc)   # intentional repeat — upweights actual work done

    skill_names = [s.get("name", "") for s in c.get("skills", [])]
    parts.append("Skills: " + ", ".join(skill_names))

    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))

    for edu in c.get("education", []):
        parts.append(edu.get("field_of_study", ""))

    return " ".join(x for x in parts if x)
