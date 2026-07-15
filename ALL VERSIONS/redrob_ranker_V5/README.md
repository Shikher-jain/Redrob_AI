# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Challenge:** Rank 100,000 candidates for a Senior AI Engineer role the way a great recruiter would — by semantic understanding, not keyword matching.

---

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
┌───────────────────┐
│   data_loader.py  │  ← loads JSONL/JSONL.GZ/JSON
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│               embedding.py                        │
│  sentence-transformers all-MiniLM-L6-v2           │
│  (fallback: TF-IDF + TruncatedSVD / LSA)          │
│  Output: L2-normalised (N, 384) float32 array     │
│  + disk cache keyed on data fingerprint            │
└────────┬──────────────────────────────────────────┘
         │   cosine similarity (N,)
         ▼
┌───────────────────────────────────────────────────┐
│          feature_engineering.py                   │
│                                                   │
│  career_quality_score        (YOE, product co.,  │
│                               title, prod. evid.) │
│  skill_match_score           (proficiency × trust)│
│  behavioral_engagement_score (23 Redrob signals)  │
│  location_availability_score (city, notice, OTW)  │
│  honeypot_score              (impossibility check)│
└────────┬──────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│                 ranker.py                         │
│  final = 0.30×semantic + 0.25×career              │
│        + 0.20×skill    + 0.15×behavioral          │
│        + 0.10×location                            │
│  honeypot penalty: ×(1 - honeypot_score)          │
│  hard disqualifiers: score → 0.001                │
│  → top-200 shortlist                              │
└────────┬──────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│                reranker.py                        │
│  Second-pass on top 200 (fast, no API calls)      │
│  description signal density (actual IR/ML work)   │
│  tenure quality (anti title-chaser)               │
│  industry trajectory (moving toward AI/ML?)       │
│  certification boost                              │
│  github activity signal                           │
│  → delta ±0.05 added to first-pass score         │
└────────┬──────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│               reasoning.py                        │
│  Generates specific, grounded 1-2 sentence        │
│  reasoning per candidate (rank-consistent,        │
│  honest about concerns, zero hallucination)       │
└────────┬──────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────┐
│            output_writer.py                       │
│  submission.csv  (100 rows, spec-compliant)       │
│  debug.csv       (all sub-scores, optional)       │
└───────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data
```
data/
├── candidates.jsonl        ← full 100K pool (or .jsonl.gz)
└── sample_candidates.json  ← 50-candidate sample (included)
```

### 3. Run the ranker

**Full 100K run (produces final submission):**
```bash
python src/main.py \
  --candidates data/candidates.jsonl \
  --out outputs/submission.csv
```

**Sample run (for testing):**
```bash
python src/main.py \
  --candidates data/sample_candidates.json \
  --out outputs/sample_out.csv \
  --top 50
```

**With custom JD file:**
```bash
python src/main.py \
  --candidates data/candidates.jsonl \
  --jd data/job_description.txt \
  --out outputs/submission.csv
```

**All options:**
```
--candidates    Path to JSONL/JSONL.GZ/JSON candidates file
--jd            Path to JD text file (builtin JD used if omitted)
--out           Output CSV path [default: outputs/submission.csv]
--top           Candidates in final output [default: 100]
--rerank-pool   Pool size for second-pass reranker [default: 200]
--no-cache      Disable embedding cache (forces re-encoding)
--debug         Also write debug CSV with all sub-scores
--model         Override embedding model name
```

### 4. Validate
```bash
python validate_submission.py outputs/submission.csv
```

---

## Scoring Formula

```
final_score = 0.30 × semantic_similarity
            + 0.25 × career_quality_score
            + 0.20 × skill_match_score
            + 0.15 × behavioral_engagement_score
            + 0.10 × location_availability_score
```

**Penalties applied after:**
- Honeypot penalty: `score × (1 - honeypot_score)` if `honeypot_score ≥ 0.5`
- Hard disqualifiers: `score → 0.001`

All weights are configurable in `src/config.py`.

---

## Why This Works

### Anti-Keyword-Stuffer Design
A skill listed as "expert" with 0 months used and 0 endorsements gets near-zero weight.
Assessment scores from the Redrob platform override self-reported proficiency.
This naturally filters out candidates who keyword-stuffed their skills section.

### JD-Specific Disqualifiers (from the actual JD)
- Title in clearly disqualified category → score ≈ 0
- Career 100% at consulting firms → score ≈ 0
- These are hard rules, not semantic guesses

### Honeypot Resistance
We check:
- Duration inconsistency (stated vs computed from dates, >24 month discrepancy)
- Future start dates
- Expert skill + zero evidence (0 months, 0 endorsements)
- Excessive expert skills (>12)

### Behavioral Signal Integration
From the JD: *"a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is not actually available."*
The `behavioral_engagement_score` encodes exactly this — `recruiter_response_rate` carries 25% of the behavioral weight.

### Plain-Language Tier 5 Detection (Reranker)
The second pass scans role **descriptions** (not the skills section) for IR/ML work evidence. A candidate who built a recommendation engine without using the word "RAG" in their skills section will still surface.

---

## Compute Performance

| Stage | Time (100K candidates, CPU) |
|-------|----------------------------|
| Data loading | ~8s |
| Embedding (all-MiniLM-L6-v2, cached) | ~3 min first run, ~5s cached |
| Embedding (LSA fallback, cached) | ~30s first run, ~2s cached |
| Feature engineering + scoring | ~45s |
| Reranking top 200 | <1s |
| Reasoning generation | <1s |
| **Total (cached embeddings)** | **< 2 minutes** |
| **Total (first run, LSA)** | **< 4 minutes** |

All within the 5-minute, 16GB RAM, CPU-only constraint.

---

## Project Structure

```
redrob_ranker/
├── src/
│   ├── config.py              # All weights and constants
│   ├── utils.py               # Shared helpers
│   ├── data_loader.py         # Load JSONL/JSON candidates + JD
│   ├── embedding.py           # Semantic embeddings (ST or LSA)
│   ├── feature_engineering.py # All structured features
│   ├── ranker.py              # Weighted scoring + first-pass rank
│   ├── reranker.py            # Second-pass precision reranker
│   ├── reasoning.py           # Grounded per-candidate reasoning
│   └── output_writer.py       # CSV writer (submission + debug)
├── data/
│   └── sample_candidates.json
├── outputs/
│   └── submission.csv
├── notebooks/
│   └── exploration.ipynb
├── validate_submission.py
├── requirements.txt
└── README.md
```

---

## AI Tools Declaration

Claude (Anthropic) was used for architecture discussion and code review.
No candidate data was fed to any LLM during ranking.
All scoring logic is implemented in local Python without any API calls.
