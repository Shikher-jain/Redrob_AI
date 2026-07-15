"""output_writer.py — Write the submission CSV and a debug/analysis CSV."""

from __future__ import annotations
import csv, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUBMISSION_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def write_submission_csv(top_records: list[dict], output_path: str | Path) -> None:
    """
    Write the final ranked submission CSV.
    Exactly 100 rows, score non-increasing, reasoning included.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(SUBMISSION_HEADER)
        for rec in top_records:
            reasoning = rec.get("reasoning", "").replace("\n", " ").strip()
            writer.writerow([
                rec["candidate_id"],
                rec["rank"],
                f"{rec['final_score']:.6f}",
                reasoning,
            ])

    logger.info("Submission CSV written → %s (%d rows)", output_path.name, len(top_records))


def write_debug_csv(top_records: list[dict], output_path: str | Path) -> None:
    """
    Write a detailed debug CSV with all sub-scores for analysis.
    Not submitted — for your own understanding and iteration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_keys = [
        "semantic_similarity", "career_quality_score", "skill_match_score",
        "behavioral_engagement_score", "location_availability_score",
        "honeypot_score", "reranker_delta",
    ]

    header = (
        ["rank", "candidate_id", "final_score", "current_title",
         "yoe", "location", "country", "open_to_work", "notice_days"]
        + feature_keys
        + ["is_honeypot", "disqualified", "reasoning"]
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in top_records:
            c   = rec.get("_raw_candidate", {})
            p   = c.get("profile", {}) if c else {}
            sig = c.get("redrob_signals", {}) if c else {}
            feats = rec.get("features", {})

            row = [
                rec.get("rank", ""),
                rec["candidate_id"],
                f"{rec['final_score']:.6f}",
                p.get("current_title", ""),
                p.get("years_of_experience", ""),
                p.get("location", ""),
                p.get("country", ""),
                sig.get("open_to_work_flag", ""),
                sig.get("notice_period_days", ""),
            ]
            for k in feature_keys:
                v = feats.get(k, rec.get(k, ""))
                row.append(f"{v:.4f}" if isinstance(v, float) else v)

            row += [
                rec.get("is_honeypot", False),
                rec.get("disqualified", False),
                rec.get("reasoning", ""),
            ]
            writer.writerow(row)

    logger.info("Debug CSV written → %s", output_path.name)
