"""
reranker.py — Second-pass reranker on top-200 shortlist.

Why no LLM?
The spec says: CPU only, 5 min, no API calls. A local LLM on 100K candidates
is impossible in that budget. Instead we apply signal-intensive checks that
are too slow for 100K but fast on 200.

This catches the "plain-language Tier 5" candidates the JD specifically mentions:
those whose skills list lacks buzzwords but whose DESCRIPTIONS show real work.
"""

from __future__ import annotations
import logging
from config import ML_INDUSTRIES
from utils import clamp, norm, parse_date, days_since

logger = logging.getLogger(__name__)

DESCRIPTION_SIGNALS = {
    "embedding", "embeddings", "vector", "retrieval", "ranking",
    "similarity", "cosine", "faiss", "qdrant", "pinecone", "weaviate",
    "semantic search", "hybrid search", "bm25",
    "ndcg", "mrr", "a/b test", "recall@",
    "fine-tun", "lora", "qlora",
    "transformer", "bert", "roberta",
    "recommendation", "recommender",
    "search engine", "information retrieval",
    "deployed to production", "shipped to",
    "evaluation framework", "offline eval", "online eval",
    "low latency", "p99", "throughput",
}

RELEVANT_CERTS = {
    "aws certified machine learning",
    "google professional machine learning",
    "deeplearning.ai", "tensorflow developer",
    "hugging face", "vector database", "mlops", "databricks",
}


def _desc_signal_density(c: dict) -> float:
    blob = " ".join(norm(r.get("description", "")) for r in c.get("career_history", []))
    hits = sum(1 for tok in DESCRIPTION_SIGNALS if tok in blob)
    return clamp(hits / 10.0)


def _tenure_quality(c: dict) -> float:
    """Penalise title-chasers: many < 18 month tenures."""
    career = c.get("career_history", [])
    if not career:
        return 0.5
    short = sum(1 for r in career if r.get("duration_months", 99) < 18)
    return clamp(1.0 - short / len(career))


def _industry_trajectory(c: dict) -> float:
    """Is the career moving TOWARD AI/ML? Recent = higher weight."""
    career = c.get("career_history", [])
    if not career:
        return 0.5
    scores, weights = [], []
    for i, role in enumerate(career):
        ind   = norm(role.get("industry", ""))
        title = norm(role.get("title", ""))
        is_ml = any(m in ind or m in title for m in ML_INDUSTRIES)
        w = 1.0 / (i + 1)
        scores.append(float(is_ml) * w)
        weights.append(w)
    return clamp(sum(scores) / max(sum(weights), 1e-6))


def _cert_boost(c: dict) -> float:
    certs = c.get("certifications", [])
    hits  = sum(1 for cert in certs
                if any(tok in norm(cert.get("name", "")) for tok in RELEVANT_CERTS))
    return clamp(hits / 3.0)


def _github_signal(c: dict) -> float:
    raw = float(c.get("redrob_signals", {}).get("github_activity_score", -1))
    return clamp(raw / 100.0) if raw >= 0 else 0.2


def rerank(top_records: list[dict]) -> list[dict]:
    """
    Apply second-pass scoring to top-N records (in-place score update).
    Each record must have `_raw_candidate` attached.
    Delta is capped at ±0.05 to avoid overriding first-pass calibration.
    """
    logger.info("Reranking top %d candidates ...", len(top_records))

    for rec in top_records:
        c = rec.get("_raw_candidate", {})
        if not c:
            continue

        delta = 0.05 * (
            0.35 * _desc_signal_density(c)
            + 0.20 * _tenure_quality(c)
            + 0.20 * _industry_trajectory(c)
            + 0.15 * _cert_boost(c)
            + 0.10 * _github_signal(c)
            - 0.50   # centre at 0 so average candidate gets ~zero delta
        )
        rec["final_score"]     = clamp(rec["final_score"] + delta)
        rec["reranker_delta"]  = round(delta, 4)

    reranked = sorted(top_records, key=lambda r: (-r["final_score"], r["candidate_id"]))
    for i, rec in enumerate(reranked, 1):
        rec["rank"] = i

    logger.info("Reranking complete.")
    return reranked
