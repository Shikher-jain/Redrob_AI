"""
main.py — End-to-end candidate ranking pipeline.

Usage:
    python src/main.py --candidates data/candidates.jsonl --out outputs/submission.csv
    python src/main.py --candidates data/sample_candidates.json --out outputs/sample_out.csv --top 50
    python src/main.py --candidates data/candidates.jsonl --out outputs/submission.csv --no-cache
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader          import load_candidates, load_job_description
from embedding            import EmbeddingEngine, cosine_sims, build_candidate_texts
from ranker               import compute_scores, rank_and_select
from reranker             import rerank
from reasoning            import generate_reasoning
from output_writer        import write_submission_csv, write_debug_csv
from utils                import setup_logging

logger = logging.getLogger(__name__)

BUILTIN_JD = """
Senior AI Engineer founding team Redrob AI Series A talent intelligence platform.
Location Pune Noida India hybrid open relocation Tier 1 Indian cities.
Experience 5 to 9 years with 4 to 5 years applied machine learning product companies.

Role: own intelligence layer ranking retrieval matching systems recruiter search candidate job search.
First 90 days: audit BM25 rule-based scoring, ship version 2 ranking embeddings hybrid retrieval,
set up evaluation NDCG MRR MAP offline A/B testing recruiter feedback loops.

Required: Production embeddings-based retrieval sentence-transformers OpenAI embeddings BGE E5,
handling embedding drift index refresh retrieval quality regression production.
Production vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS.
Strong Python production code quality.
Evaluation frameworks ranking systems NDCG MRR MAP offline online correlation A/B test.

Nice to have: LLM fine-tuning LoRA QLoRA PEFT. Learning to rank XGBoost LightGBM.
HR tech recruiting tech marketplace products. Distributed systems large scale inference.
Open source contributions AI ML.

Disqualifiers: pure research no production deployment. LangChain only without pre-LLM ML production.
No production code 18 months pure architecture. Exclusively consulting TCS Infosys Wipro Accenture
Cognizant Capgemini HCL Tech Mahindra. Computer vision speech robotics without NLP information retrieval.

Ideal: 6 to 8 years total shipped end to end ranking search recommendation system real users scale.
Hybrid retrieval dense sparse evaluation offline online LLM fine-tune prompt opinions.
Noida Pune willing relocate. Active Redrob recently. Notice under 30 days preferred.
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    p.add_argument("--candidates",  default="../data/sample_candidates.json")
    p.add_argument("--jd",          default=None)
    p.add_argument("--out",         default="../outputs/submission.csv")
    p.add_argument("--top",         type=int, default=100)
    p.add_argument("--rerank-pool", type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=64,
                   help="Batch size for sentence-transformers encoding (lower = less RAM)")
    p.add_argument("--no-cache",    action="store_true")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--model",       default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    t0   = time.time()

    # Ensure output dir exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    logger.info("══════════════════════════════════════════════════════════")
    logger.info("  Redrob Hackathon — Candidate Ranker")
    logger.info("══════════════════════════════════════════════════════════")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("STEP 1/7 | Loading data")
    candidates = load_candidates(args.candidates)
    jd_text    = load_job_description(args.jd) if args.jd else BUILTIN_JD
    logger.info("Candidates: %d | JD: %d chars", len(candidates), len(jd_text))

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    logger.info("STEP 2/7 | Building embeddings (backend auto-selected)")
    from config import EMBEDDING_MODEL
    engine     = EmbeddingEngine(args.model or EMBEDDING_MODEL)
    use_cache  = not args.no_cache
    cand_texts = build_candidate_texts(candidates)

    # LSA needs to see the full corpus to fit its vocabulary
    if engine.backend == "lsa":
        engine.fit_lsa([jd_text] + cand_texts)

    jd_emb    = engine.encode([jd_text], tag="jd", use_cache=use_cache)[0]
    cand_embs = engine.encode(cand_texts, tag="candidates",
                              batch_size=args.batch_size, use_cache=use_cache)

    sims = cosine_sims(jd_emb, cand_embs)
    logger.info("Semantic sims → min=%.3f  mean=%.3f  max=%.3f  backend=%s",
                sims.min(), sims.mean(), sims.max(), engine.backend)

    # ── 3. Feature engineering + scoring ──────────────────────────────────────
    logger.info("STEP 3/7 | Feature engineering & scoring (%d candidates)", len(candidates))
    scored = compute_scores(candidates, sims)

    # ── 4. First-pass rank ────────────────────────────────────────────────────
    logger.info("STEP 4/7 | First-pass ranking → top %d", args.rerank_pool)
    pool  = rank_and_select(scored, top_n=min(args.rerank_pool, len(scored)))

    # ── 5. Rerank ─────────────────────────────────────────────────────────────
    logger.info("STEP 5/7 | Reranking (description signals, trajectory)")
    reranked = rerank(pool)

    # ── 6. Reasoning ──────────────────────────────────────────────────────────
    logger.info("STEP 6/7 | Generating reasoning for top %d", args.top)
    final = reranked[:args.top]
    for rec in final:
        rec["reasoning"] = generate_reasoning(rec)

    # ── 7. Write ──────────────────────────────────────────────────────────────
    logger.info("STEP 7/7 | Writing output")
    out_path = Path(args.out)
    write_submission_csv(final, out_path)
    if args.debug:
        write_debug_csv(reranked, out_path.parent / (out_path.stem + "_debug.csv"))

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("Done in %.1fs | %s", elapsed, out_path.resolve())
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
    logger.info("Honeypots in top %d: %d  (limit: 10%%)", args.top, hp)


if __name__ == "__main__":
    main()
