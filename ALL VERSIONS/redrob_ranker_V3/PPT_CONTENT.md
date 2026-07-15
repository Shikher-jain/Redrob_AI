# Redrob Hackathon — Presentation Deck
## Intelligent Candidate Discovery & Ranking
### Shikher Jain

---

## Slide 1 — Title

**Intelligent Candidate Discovery & Ranking**
*Redrob Hackathon — Data & AI Challenge*

Shikher Jain | AI/ML Engineer
[GitHub Repo] | [Sandbox Demo]

Visual suggestion: Redrob logo + a simple illustration of a recruiter looking at ranked candidates

---

## Slide 2 — The Problem

**Recruiters receive hundreds of profiles. Keyword filters miss the right person.**

Three specific failure modes:

1. **The Keyword Stuffer**
   A candidate lists "RAG, Pinecone, FAISS, LoRA, NDCG" as expert skills — with 0 months used and 0 endorsements.
   Keyword ranker puts them in top 10. They've never built anything.

2. **The Plain-Language Expert**
   A candidate writes: "I built a system that finds the most relevant job for each candidate from our 2M listing database."
   That's a retrieval system. Keyword ranker misses them entirely.

3. **The Perfect-On-Paper Ghost**
   9 years of experience, all the right skills — but last active 7 months ago, 3% recruiter response rate.
   For hiring purposes, this person is not available.

**The right system must catch all three.**

---

## Slide 3 — Why Keyword Matching Fails

| What keyword systems do | What they should do |
|------------------------|---------------------|
| Count "RAG" in skills section | Understand if the person built retrieval systems |
| Treat "expert" self-rating as truth | Weight by endorsements + duration used |
| Rank inactive profiles equally | Down-weight candidates who won't respond |
| Miss the plain-language candidate | Read descriptions for semantic meaning |
| Get fooled by honeypots | Detect impossible profile contradictions |

**Keyword matching is a skill-presence check. We need skill-credibility + context understanding.**

---

## Slide 4 — Our Solution: Hybrid Semantic Ranker

**Three layers working together:**

```
Semantic Understanding (30%)
  "What kind of work has this person actually done?"
  → Sentence-transformers embedding + cosine similarity to JD

Structured Signal Scoring (55%)
  "Do the hard facts check out?"
  → Career quality, skill credibility, behavioral signals, location

Second-Pass Reranker (±5% delta)
  "Are we catching the plain-language experts?"
  → Description-level IR/ML signal scan
```

**No API calls. No GPU. Runs in <5 minutes on CPU for 100K candidates.**

---

## Slide 5 — Architecture Diagram

```
candidates.jsonl (100,000)
         │
         ▼
   [ data_loader.py ]
         │
         ▼
   [ embedding.py ]
   sentence-transformers → (N, 384) embeddings
   fallback: TF-IDF + TruncatedSVD (LSA)
   + disk cache (re-run = 5 seconds, not 3 minutes)
         │
         │ cosine_similarity(JD, candidate) → (N,)
         ▼
   [ feature_engineering.py ]
   ┌────────────────────────────────────────────┐
   │ career_quality_score  (YOE, product co,   │
   │                        title, prod. evid.) │
   │ skill_match_score     (prof × trust)       │
   │ behavioral_score      (23 Redrob signals)  │
   │ location_avail_score  (city, notice, OTW)  │
   │ honeypot_score        (date checks)        │
   └────────────────────────────────────────────┘
         │
         ▼
   [ ranker.py ] → weighted score → top-200
         │
         ▼
   [ reranker.py ] → description scan → final top-100
         │
         ▼
   [ reasoning.py ] → grounded 1-2 sentence explanation
         │
         ▼
   submission.csv  ✓
```

---

## Slide 6 — Scoring Formula

```
final_score =
    0.30 × semantic_similarity         ← JD + candidate text cosine sim
  + 0.25 × career_quality_score        ← YOE, product co., title, prod. evidence
  + 0.20 × skill_match_score           ← proficiency × credibility
  + 0.15 × behavioral_engagement_score ← 23 Redrob platform signals
  + 0.10 × location_availability_score ← city, notice period, open-to-work
```

**Then penalties:**
- `honeypot_score ≥ 0.5` → multiply by `(1 - honeypot_score)` (smooth reduction)
- Hard disqualified title or 100% consulting career → score = 0.001

**All weights configurable in `config.py` — single file to tune the whole system.**

---

## Slide 7 — Feature Engineering Details

### Career Quality Score (0.25 weight)
| Sub-component | Weight | What it checks |
|---------------|--------|----------------|
| Years of experience | 30% | Penalty outside 5–9yr band; peak at 7yr |
| Product vs consulting | 25% | Fraction of career at non-consulting firms |
| Title alignment | 25% | Research-only / disqualified titles → near-zero |
| Production evidence | 20% | Keywords in descriptions: "deployed", "production", "scale", "latency" |

### Skill Match Score (0.20 weight)
- Trust = `0.5 × min(endorsements/15, 1) + 0.5 × min(duration_months/30, 1)`
- Effective = `proficiency × trust`
- Platform assessment scores override self-reported proficiency (objective evidence)
- Core JD skills weighted 2× over preferred skills

### Behavioral Engagement Score (0.15 weight)
| Signal | Weight | From JD |
|--------|--------|---------|
| recruiter_response_rate | 25% | "5% response rate = not available" |
| saved_by_recruiters_30d | 15% | Market validation signal |
| github_activity_score | 15% | External coding evidence |
| profile_completeness | 10% | Seriousness signal |
| interview_completion_rate | 10% | Reliability signal |
| avg_response_time_hours | 10% | Speed of engagement |
| verification bundle | 10% | Email + phone + LinkedIn verified |

---

## Slide 8 — Honeypot Detection

The dataset contains ~80 honeypots with subtly impossible profiles.

**We detect them through 4 checks:**

1. **Date inconsistency**: `|computed_months - stated_months| > 24`
   → e.g., "8 years experience at a company founded 3 years ago"

2. **Future start date**: role starts after today

3. **Zero-evidence expert skill**: `proficiency=expert AND duration=0 AND endorsements=0`
   → Claims expertise with no usage and no endorsements

4. **Excessive expert skills**: more than 12 expert skills simultaneously

**Score combines checks multiplicatively — single strong flag = significant penalty.**

Result: `honeypot_score ≥ 0.5` triggers penalty multiplier `× (1 - hp_score)`.
No hard blocklist — the penalty naturally pushes honeypots below rank 100.

---

## Slide 9 — Second-Pass Reranker

**Problem it solves:** The plain-language Tier 5 candidate.

The JD says: *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."*

**How:**
- First pass ranks 200 candidates
- Reranker scans **role descriptions** (not skills section) for 25 IR/ML signal tokens:
  `embedding`, `vector`, `retrieval`, `ranking`, `cosine`, `FAISS`, `NDCG`, `A/B test`, `fine-tun`, `deployed to production`, etc.
- Additional signals: tenure quality (anti title-chaser), industry trajectory, certifications, GitHub
- Delta = `±0.05` — doesn't override first-pass calibration, just breaks ties and surfaces buried candidates

**Runtime: <1 second on top 200.**

---

## Slide 10 — Reasoning Quality

**The spec evaluates reasoning on 6 dimensions. We engineered for all 6:**

| Spec requirement | Our approach |
|-----------------|--------------|
| Specific facts | Pull actual values: yoe, title, company, notice days, location |
| JD connection | Reference JD requirements: "preferred 5–9 year band", "preferred city" |
| Honest concerns | Acknowledge notice period >60d, low engagement, non-preferred location |
| No hallucination | Only facts from the candidate's actual record — zero inference |
| Variation | Tiered templates (rank 1–15, 16–40, 41–70, 71–100) with candidate-specific fills |
| Rank consistency | Tier A reasoning is positive; Tier D reasoning states the gap honestly |

**Example (Rank 1):**
> "Recommendation Systems Engineer with 6.0 years in the JD's preferred 5–9 year band; currently at Swiggy for 14 months. Strong semantic alignment (0.45) with verified skills in Pinecone, Embeddings, scikit-learn; based in Hyderabad (preferred city); 60-day notice period."

---

## Slide 11 — Results (Sample Data)

Sample: 50 candidates from the provided sample_candidates.json

| Rank | Candidate | Title | Score | Why |
|------|-----------|-------|-------|-----|
| 1 | CAND_0000031 | Recommendation Systems Engineer | 0.645 | Directly relevant role, Hyderabad (preferred), verified vector DB skills |
| 2 | CAND_0000014 | Frontend Engineer | 0.482 | Hyderabad, FAISS + OpenSearch skills, 8.4yr in band |
| 3 | CAND_0000038 | Java Developer | 0.475 | Weaviate skill, Swiggy (product co.), 6.7yr |
| 4 | CAND_0000032 | .NET Developer | 0.425 | Gurgaon, embeddings + OpenCV, 8.1yr |
| 5 | CAND_0000001 | Backend Engineer | 0.423 | Milvus + NLP, 6.9yr, BUT overseas = penalty |

**Disqualified (score ≈ 0):** Marketing Managers, HR Managers, Accountants, Civil Engineers, Mechanical Engineers — 26 of 50 correctly filtered.

**Honeypots detected: 0** (clean dataset in sample).

---

## Slide 12 — Compute Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Load 100K candidates | ~8s | JSONL streaming |
| Embed 100K candidates (ST, first run) | ~3 min | Saved to cache |
| Embed 100K candidates (ST, cached) | ~5s | Numpy .npy load |
| Embed 100K candidates (LSA, first run) | ~30s | TF-IDF + SVD |
| Feature engineering (100K) | ~45s | Pure Python |
| Reranking top 200 | <1s | |
| **Total (first run)** | **< 5 min** | Within spec |
| **Total (subsequent runs)** | **< 2 min** | Cache hit |

**Memory: ~2-4 GB peak** (well within 16 GB limit)
**No GPU, no API calls, no network during ranking** ✓

---

## Slide 13 — Future Work

1. **True LLM Reranking (post-constraint)**
   Run a local 7B model (Mistral/Llama) on top-50 candidates with full profile + JD context.
   Currently blocked by 5-min CPU constraint — feasible if constraint relaxed.

2. **Learning-to-Rank**
   With recruiter feedback data, train XGBoost/LightGBM on (candidate, JD, relevance) triples.
   Current heuristic weights become initial features; model learns true weights from data.

3. **Embedding Drift Detection**
   In production: monitor cosine sim distribution over time; alert when new candidate pool shifts.

4. **Evaluation Framework**
   NDCG@10/50, MRR, offline-to-online correlation, A/B testing harness.
   This is what the JD describes as the "evaluation infrastructure" to build in weeks 9–12.

5. **Cross-JD Generalisation**
   Extend config.py weights to be JD-derived (parse requirements → auto-configure weights).

---

## Slide 14 — Conclusion

**What we built:**

✓ Semantic ranker that understands what the JD means, not just what it says
✓ Skill credibility check — endorsements + duration + assessment override
✓ Behavioral signal integration — response rate, recency, market validation
✓ Honeypot resistance — date checks, zero-evidence expert detection
✓ Plain-language candidate surfacing — description-level IR/ML signal scan
✓ Grounded reasoning — specific facts, honest concerns, zero hallucination
✓ Runs in <5 min on CPU, no API calls, no GPU

**The system ranks candidates the way a good recruiter thinks:**
Not "who has the most AI keywords" but "who has actually built retrieval systems, ships to production, is reachable, and fits the 5-9 year sweet spot."

---

*Thank you.*
*Questions?*
