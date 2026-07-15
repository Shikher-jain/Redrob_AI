"""
main.py — End-to-end candidate ranking pipeline.

Usage:
    # Full 100K run:
    python main.py --candidates ../data/candidates.jsonl --out ../outputs/submission.csv

    # Sample run:
    python main.py --candidates ../data/sample_candidates.json --out ../outputs/sample_out.csv --top 50

    # Disable cache:
    python main.py --candidates ../data/candidates.jsonl --out ../outputs/submission.csv --no-cache
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader         import load_candidates, load_job_description
from embedding           import EmbeddingEngine, cosine_sims, build_candidate_texts
from ranker              import compute_scores, rank_and_select
from reranker            import rerank
from reasoning           import generate_reasoning
from output_writer       import write_submission_csv, write_debug_csv
from utils               import setup_logging

logger = logging.getLogger(__name__)

# ── Builtin JD (full Redrob hackathon text) ───────────────────────────────────
BUILTIN_JD = """
Senior AI Engineer founding team at Redrob AI, Series A talent intelligence platform.
Location: Pune or Noida India, hybrid, open to relocation from Tier 1 Indian cities.
Experience: 5 to 9 years, with 4 to 5 years in applied machine learning at product companies.

Role mandate: own the intelligence layer — ranking, retrieval, and matching systems
for recruiter search and candidate job search. First 90 days: audit existing BM25
rule-based scoring, ship version 2 ranking system with embeddings and hybrid retrieval,
set up evaluation infrastructure with NDCG MRR MAP offline A/B testing recruiter feedback.

Absolutely required: Production experience with embeddings-based retrieval systems
using sentence-transformers OpenAI embeddings BGE E5 or similar, handling embedding drift
index refresh retrieval quality regression in production. Production experience with vector
databases or hybrid search infrastructure including Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS. Strong Python production code quality. Evaluation frameworks
for ranking systems: NDCG MRR MAP offline-to-online correlation A/B test interpretation.

Nice to have: LLM fine-tuning with LoRA QLoRA PEFT. Learning to rank models XGBoost LightGBM.
Prior exposure to HR tech recruiting tech marketplace products. Distributed systems large scale
inference optimization. Open source contributions in AI ML space.

Disqualifiers: Pure research environments without production deployment. LangChain only AI
experience without pre-LLM-era ML production work. No production code written in last 18 months
due to moving into pure architecture role. Career exclusively at consulting firms TCS Infosys
Wipro Accenture Cognizant Capgemini HCL Tech Mahindra. Primary expertise in computer vision
speech or robotics without significant NLP information retrieval exposure.

Ideal candidate: 6 to 8 years total experience. Shipped at least one end-to-end ranking search
or recommendation system to real users at meaningful scale. Strong opinions about retrieval
hybrid versus dense, evaluation offline versus online, LLM integration fine-tune versus prompt.
Located in or willing to relocate to Noida or Pune. Active on Redrob platform recently.
Notice period under 30 days preferred, buyout available up to 30 days.
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Redrob Hackathon Candidate Ranker")
    p.add_argument("--candidates",   default="../data/sample_candidates.json")
    p.add_argument("--jd",           default=None,
                   help="Path to JD text file (builtin JD used if omitted)")
    p.add_argument("--out",          default="../outputs/submission.csv")
    p.add_argument("--top",          type=int, default=100)
    p.add_argument("--rerank-pool",  type=int, default=200)
    p.add_argument("--no-cache",     action="store_true")
    p.add_argument("--debug",        action="store_true",
                   help="Also write debug CSV with all sub-scores")
    p.add_argument("--model",        default=None,
                   help="Override embedding model (default: all-MiniLM-L6-v2)")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    t0   = time.time()

    logger.info("══════════════════════════════════════════════════════════")
    logger.info("  Redrob Hackathon — Candidate Ranker")
    logger.info("══════════════════════════════════════════════════════════")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    logger.info("STEP 1/7 | Loading data")
    candidates = load_candidates(args.candidates)
    jd_text    = load_job_description(args.jd) if args.jd else BUILTIN_JD
    if not args.jd:
        logger.info("Using builtin JD text (%d chars)", len(jd_text))
    logger.info("Candidates: %d", len(candidates))

    # ── 2. Embeddings ─────────────────────────────────────────────────────────
    logger.info("STEP 2/7 | Embedding (semantic representation)")
    from config import EMBEDDING_MODEL
    engine = EmbeddingEngine(args.model or EMBEDDING_MODEL)

    cand_texts = build_candidate_texts(candidates)
    all_texts  = [jd_text] + cand_texts   # joint corpus for LSA fitting

    use_cache  = not args.no_cache

    # For LSA: fit on the full corpus (JD + candidates)
    engine.fit_lsa(all_texts)

    jd_emb    = engine.encode([jd_text], tag="jd", use_cache=use_cache)[0]
    cand_embs = engine.encode(cand_texts, tag="candidates", use_cache=use_cache)

    sims = cosine_sims(jd_emb, cand_embs)
    logger.info("Semantic sim → min=%.3f  mean=%.3f  max=%.3f  backend=%s",
                sims.min(), sims.mean(), sims.max(), engine.backend)

    # ── 3. Feature Engineering + Scoring ─────────────────────────────────────
    logger.info("STEP 3/7 | Feature engineering & scoring")
    scored = compute_scores(candidates, sims)

    # ── 4. First-pass Ranking ─────────────────────────────────────────────────
    logger.info("STEP 4/7 | First-pass ranking → top %d", args.rerank_pool)
    pool_size = min(args.rerank_pool, len(scored))
    top_pool  = rank_and_select(scored, top_n=pool_size)

    # ── 5. Second-pass Reranking ──────────────────────────────────────────────
    logger.info("STEP 5/7 | Reranking (description signals, trajectory, certs)")
    reranked = rerank(top_pool)

    # ── 6. Final top-N ────────────────────────────────────────────────────────
    logger.info("STEP 6/7 | Selecting top %d & generating reasoning", args.top)
    final = reranked[:args.top]
    for rec in final:
        rec["reasoning"] = generate_reasoning(rec)

    # ── 7. Output ─────────────────────────────────────────────────────────────
    logger.info("STEP 7/7 | Writing output")
    out_path = Path(args.out)
    write_submission_csv(final, out_path)
    if args.debug:
        write_debug_csv(reranked, out_path.parent / (out_path.stem + "_debug.csv"))

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("Done in %.1fs | Output: %s", elapsed, out_path.resolve())
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("TOP 10:")
    for rec in final[:10]:
        c = rec.get("_raw_candidate", {})
        p = c.get("profile", {}) if c else {}
        logger.info("  #%3d %-14s %-35s %4.1fy %-22s %.4f",
                    rec["rank"], rec["candidate_id"],
                    p.get("current_title","")[:34],
                    p.get("years_of_experience", 0),
                    p.get("location","")[:21],
                    rec["final_score"])

    hp = sum(1 for r in final if r.get("is_honeypot"))
    logger.info("Honeypots in top %d: %d (limit 10%%)", args.top, hp)


if __name__ == "__main__":
    main()
