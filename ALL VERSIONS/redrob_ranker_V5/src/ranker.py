"""
ranker.py — Combine semantic + structured features into a final score.

Formula (configurable in config.py):
  final = 0.30 * semantic_similarity
        + 0.25 * career_quality_score
        + 0.20 * skill_match_score
        + 0.15 * behavioral_engagement_score
        + 0.10 * location_availability_score

Honeypot penalty: multiplicative reduction if honeypot_score ≥ 0.5.
Hard disqualifiers: title from disqualified set OR 100% consulting career → score ≈ 0.
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np

from config import WEIGHTS, DISQUALIFIED_TITLE_TOKENS, PURE_CONSULTING
from feature_engineering import extract_all_features
from utils import clamp, contains_any, norm

logger = logging.getLogger(__name__)


def _is_hard_disqualified(c: dict) -> bool:
    """JD-stated outright disqualifiers."""
    title  = c.get("profile", {}).get("current_title", "")
    career = c.get("career_history", [])

    if contains_any(title, DISQUALIFIED_TITLE_TOKENS):
        return True

    if career:
        consult_m = sum(
            r.get("duration_months", 0) for r in career
            if any(f in norm(r.get("company", "")) for f in PURE_CONSULTING)
        )
        total_m = sum(r.get("duration_months", 0) for r in career)
        if total_m > 0 and consult_m / total_m >= 0.98:
            return True

    return False


def compute_scores(
    candidates: list[dict],
    semantic_sims: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Compute final score for all candidates.

    Returns list of dicts with keys:
        candidate_id, final_score, features, is_honeypot, disqualified, _raw_candidate
    """
    results: list[dict] = []

    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", f"UNKNOWN_{i}")

        if _is_hard_disqualified(c):
            results.append({
                "candidate_id":   cid,
                "final_score":    0.001,
                "features":       {},
                "is_honeypot":    False,
                "disqualified":   True,
                "_raw_candidate": c,
            })
            continue

        feats = extract_all_features(c)
        feats["semantic_similarity"] = float(semantic_sims[i])

        raw = (
            WEIGHTS["semantic_similarity"]          * feats["semantic_similarity"]
            + WEIGHTS["career_quality_score"]       * feats["career_quality_score"]
            + WEIGHTS["skill_match_score"]          * feats["skill_match_score"]
            + WEIGHTS["behavioral_engagement_score"]* feats["behavioral_engagement_score"]
            + WEIGHTS["location_availability_score"]* feats["location_availability_score"]
        )

        hp = feats["honeypot_score"]
        if hp >= 0.5:
            raw *= (1.0 - hp)

        results.append({
            "candidate_id":   cid,
            "final_score":    clamp(raw),
            "features":       feats,
            "is_honeypot":    hp >= 0.5,
            "disqualified":   False,
            "_raw_candidate": c,
        })

    logger.info(
        "Scored %d | honeypots: %d | disqualified: %d",
        len(results),
        sum(1 for r in results if r["is_honeypot"]),
        sum(1 for r in results if r["disqualified"]),
    )
    return results


def rank_and_select(scored: list[dict], top_n: int = 100) -> list[dict]:
    """Sort by score desc, tie-break by candidate_id asc, add rank field."""
    ranked = sorted(scored, key=lambda r: (-r["final_score"], r["candidate_id"]))
    top = ranked[:top_n]
    for i, rec in enumerate(top, 1):
        rec["rank"] = i
    return top
