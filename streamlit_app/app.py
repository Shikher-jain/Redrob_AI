"""
app.py — Redrob Hackathon | Intelligent Candidate Ranker Dashboard
Team S2 | Shikher Jain

Run:
    streamlit run streamlit_app/app.py
"""

import sys, os, json, csv, io, time, hashlib, gzip
from pathlib import Path

# ── path setup so src/ modules are importable ────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker · Team S2",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Gradient header ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0533 0%, #3b0764 30%, #6d28d9 60%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(109,40,217,0.3);
}
.hero-title {
    font-size: 2.2rem; font-weight: 700; color: #ffffff;
    margin: 0 0 0.3rem 0; letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.0rem; color: #c4b5fd; margin: 0;
}
.hero-badges { margin-top: 0.8rem; display: flex; gap: 0.5rem; }
.badge {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; color: #e9d5ff;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #ddd6fe; border-radius: 12px;
    padding: 1.2rem 1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(109,40,217,0.08);
}
.metric-number {
    font-size: 2.0rem; font-weight: 700; color: #5b21b6;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem; color: #7c3aed; margin-top: 0.3rem;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;
}

/* ── Rank card ── */
.rank-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.rank-card:hover { box-shadow: 0 4px 16px rgba(109,40,217,0.12); }
.rank-badge-1  { background: linear-gradient(135deg,#fbbf24,#f59e0b); color:#fff; }
.rank-badge-2  { background: linear-gradient(135deg,#9ca3af,#6b7280); color:#fff; }
.rank-badge-3  { background: linear-gradient(135deg,#cd7f32,#b45309); color:#fff; }
.rank-badge    { background: linear-gradient(135deg,#7c3aed,#4f46e5); color:#fff; }
.rbadge {
    display:inline-block; width:32px; height:32px; border-radius:50%;
    font-weight:700; font-size:0.85rem; line-height:32px; text-align:center;
}

/* ── Score bar ── */
.score-bar-bg {
    background: #f3f4f6; border-radius: 4px; height: 6px; margin-top: 4px;
}
.score-bar-fill {
    height: 6px; border-radius: 4px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
}

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 500; font-size: 0.88rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
}
section[data-testid="stSidebar"] * { color: #e9d5ff !important; }
section[data-testid="stSidebar"] .stSlider > div { color: #c4b5fd !important; }
section[data-testid="stSidebar"] hr { border-color: #4338ca; }

/* ── Status pills ── */
.pill-green { background:#dcfce7; color:#166534; border-radius:12px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.pill-red   { background:#fee2e2; color:#991b1b; border-radius:12px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.pill-gray  { background:#f3f4f6; color:#374151; border-radius:12px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

/* ── Section headers ── */
.section-header {
    font-size:1.15rem; font-weight:700; color:#1e1b4b;
    border-left: 4px solid #7c3aed; padding-left: 0.75rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* ── Info box ── */
.info-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    font-size: 0.87rem; color: #166534;
}
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    font-size: 0.87rem; color: #92400e;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_candidates_cached(path: str) -> list:
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    candidates = []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass
    return candidates


def score_label(score: float) -> str:
    if score >= 0.75: return "🟢 Strong Fit"
    if score >= 0.55: return "🟡 Good Fit"
    if score >= 0.35: return "🟠 Partial Fit"
    return "🔴 Weak Fit"


def render_score_bar(score: float) -> str:
    pct = int(score * 100)
    return f"""
    <div class="score-bar-bg">
      <div class="score-bar-fill" style="width:{pct}%"></div>
    </div>"""


def rank_badge_html(rank: int) -> str:
    cls = {1:"rank-badge-1", 2:"rank-badge-2", 3:"rank-badge-3"}.get(rank, "rank-badge")
    return f'<span class="rbadge {cls}">#{rank}</span>'


def yoe_color(yoe: float) -> str:
    if 5 <= yoe <= 9: return "#166534"
    if 3 <= yoe < 5 or 9 < yoe <= 12: return "#92400e"
    return "#991b1b"


# ══════════════════════════════════════════════════════════════════════════════
# ── Pipeline runner ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(candidates: list, jd_text: str, weights: dict, top_n: int, use_cache: bool):
    """Run the full ranking pipeline and return list of result dicts."""
    import numpy as np
    from embedding import EmbeddingEngine, cosine_sims, build_candidate_texts
    from ranker import compute_scores, rank_and_select
    from reranker import rerank
    from reasoning import generate_reasoning

    progress = st.progress(0, text="⚙️ Initialising embedding engine...")

    engine = EmbeddingEngine()
    cand_texts = build_candidate_texts(candidates)

    progress.progress(10, text="📐 Fitting semantic model...")
    if engine.backend == "lsa":
        engine.fit_lsa([jd_text] + cand_texts)

    progress.progress(20, text=f"🔢 Encoding JD ({engine.backend.upper()})...")
    jd_emb = engine.encode([jd_text], tag="jd_app", use_cache=use_cache)[0]

    progress.progress(30, text=f"🔢 Encoding {len(candidates):,} candidates...")
    cand_embs = engine.encode(cand_texts, tag="candidates_app",
                               batch_size=64, use_cache=use_cache)

    progress.progress(60, text="📊 Computing similarity scores...")
    sims = cosine_sims(jd_emb, cand_embs)

    progress.progress(70, text="⚖️ Feature engineering & weighted scoring...")

    # Apply custom weights from sidebar
    import config as cfg_mod
    orig_weights = dict(cfg_mod.WEIGHTS)
    cfg_mod.WEIGHTS.update(weights)

    scored = compute_scores(candidates, sims)
    cfg_mod.WEIGHTS.update(orig_weights)  # restore

    progress.progress(85, text="🏆 Ranking & reranking...")
    pool = rank_and_select(scored, top_n=min(top_n * 2, len(scored)))
    reranked = rerank(pool)

    progress.progress(95, text="✍️ Generating reasoning...")
    final = reranked[:top_n]
    for rec in final:
        rec["reasoning"] = generate_reasoning(rec)

    progress.progress(100, text="✅ Done!")
    time.sleep(0.4)
    progress.empty()
    return final, sims, engine.backend


# ══════════════════════════════════════════════════════════════════════════════
# ── Session state init ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results   = None
    st.session_state.sims      = None
    st.session_state.backend   = None
    st.session_state.n_cands   = 0
    st.session_state.elapsed   = 0
    st.session_state.candidates = []
    st.session_state.run_count  = 0


# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎯 Redrob Ranker")
    st.markdown("**Team S2** · Shikher Jain")
    st.markdown("---")

    # ── Data source ──────────────────────────────────────────────────────────
    st.markdown("#### 📂 Data Source")
    data_mode = st.radio("Candidates", ["Upload file", "Use project data"], index=1)

    uploaded_file = None
    candidate_path = None

    if data_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload candidates (.json / .jsonl / .jsonl.gz)",
            type=["json", "jsonl", "gz"],
        )
    else:
        default_paths = [
            ROOT / "data" / "sample_candidates.json",
            ROOT / "data" / "candidates.jsonl",
            Path("/mnt/user-data/uploads/sample_candidates.json"),
        ]
        found = [p for p in default_paths if p.exists()]
        if found:
            options = {p.name: p for p in found}
            chosen = st.selectbox("Select file", list(options.keys()))
            candidate_path = options[chosen]
            st.caption(f"📍 {candidate_path}")
        else:
            st.warning("No data found. Upload a file instead.")
            data_mode = "Upload file"

    st.markdown("---")

    # ── Scoring weights ───────────────────────────────────────────────────────
    st.markdown("#### ⚖️ Scoring Weights")
    st.caption("Adjust to tune the ranker. Must sum to 1.0.")

    w_sem  = st.slider("Semantic Similarity",         0.0, 0.6, 0.30, 0.05)
    w_car  = st.slider("Career Quality",               0.0, 0.5, 0.25, 0.05)
    w_ski  = st.slider("Skill Match",                  0.0, 0.5, 0.20, 0.05)
    w_beh  = st.slider("Behavioral Engagement",        0.0, 0.4, 0.15, 0.05)
    w_loc  = st.slider("Location / Availability",      0.0, 0.3, 0.10, 0.05)

    total_w = round(w_sem + w_car + w_ski + w_beh + w_loc, 2)
    if abs(total_w - 1.0) > 0.01:
        st.markdown(f'<div class="warn-box">⚠️ Weights sum to <b>{total_w}</b> — must be 1.0</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">✅ Weights sum to <b>{total_w}</b></div>',
                    unsafe_allow_html=True)

    weights = {
        "semantic_similarity":          w_sem,
        "career_quality_score":         w_car,
        "skill_match_score":            w_ski,
        "behavioral_engagement_score":  w_beh,
        "location_availability_score":  w_loc,
    }

    st.markdown("---")

    # ── Run params ────────────────────────────────────────────────────────────
    st.markdown("#### 🔧 Run Parameters")
    top_n    = st.selectbox("Top-N candidates", [10, 25, 50, 100], index=1)
    use_cache = st.toggle("Use embedding cache", value=True)

    st.markdown("---")
    run_btn = st.button(
        "🚀 Run Ranking Pipeline",
        use_container_width=True,
        disabled=(abs(total_w - 1.0) > 0.01),
        type="primary",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── HERO HEADER ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🎯 Intelligent Candidate Discovery & Ranking</div>
  <div class="hero-sub">Redrob Hackathon · India.Runs · Team S2 — Shikher Jain</div>
  <div class="hero-badges">
    <span class="badge">sentence-transformers</span>
    <span class="badge">semantic search</span>
    <span class="badge">5-signal scoring</span>
    <span class="badge">honeypot detection</span>
    <span class="badge">CPU-only · no API calls</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── JD INPUT SECTION ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

BUILTIN_JD = """Senior AI Engineer — Founding Team at Redrob AI (Series A talent intelligence platform).
Location: Pune / Noida, India (Hybrid). Experience: 5–9 years.

Role: Own the intelligence layer — ranking, retrieval, and matching systems.
First 90 days: audit BM25 rule-based scoring, ship v2 ranking with embeddings and hybrid retrieval,
set up evaluation infrastructure (NDCG, MRR, MAP, A/B testing, recruiter feedback loops).

Absolutely required:
- Production embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI embeddings).
  Must have handled embedding drift, index refresh, retrieval quality regression in production.
- Production vector databases / hybrid search: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, FAISS.
- Strong Python, production code quality.
- Evaluation frameworks: NDCG, MRR, MAP, offline-to-online correlation, A/B testing.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost / LightGBM),
HR-tech / marketplace product experience, distributed systems, open-source AI/ML contributions.

Disqualifiers:
- Pure research career, no production deployment.
- Career exclusively at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL).
- No production code in last 18 months (pure architecture role).
- LangChain-only AI experience without pre-LLM-era ML production work.
- Primary expertise in CV / speech / robotics without NLP / IR exposure.

Ideal: 6–8 years total, 4–5 in applied ML at product companies.
Shipped end-to-end ranking / search / recommendation system to real users at scale.
Active on Redrob platform. Notice period under 30 days preferred."""

with st.expander("📋 Job Description — Click to view / edit", expanded=False):
    jd_text = st.text_area(
        "Job Description",
        value=BUILTIN_JD,
        height=300,
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── LOAD CANDIDATES ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

candidates = []

if uploaded_file:
    with st.spinner("Loading uploaded candidates..."):
        content = uploaded_file.read()
        if uploaded_file.name.endswith(".json"):
            candidates = json.loads(content)
        elif uploaded_file.name.endswith(".gz"):
            import gzip
            with gzip.open(io.BytesIO(content), "rt") as f:
                candidates = [json.loads(l) for l in f if l.strip()]
        else:
            candidates = [json.loads(l) for l in content.decode().split("\n") if l.strip()]
elif candidate_path and candidate_path.exists():
    with st.spinner(f"Loading {candidate_path.name}..."):
        candidates = load_candidates_cached(str(candidate_path))

st.session_state.candidates = candidates


# ══════════════════════════════════════════════════════════════════════════════
# ── RUN PIPELINE ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if run_btn and candidates:
    if abs(total_w - 1.0) > 0.01:
        st.error(f"Weights must sum to 1.0 (currently {total_w}). Adjust in sidebar.")
    else:
        t0 = time.time()
        with st.spinner(""):
            try:
                results, sims, backend = run_pipeline(
                    candidates, jd_text, weights, top_n, use_cache
                )
                st.session_state.results  = results
                st.session_state.sims     = sims
                st.session_state.backend  = backend
                st.session_state.n_cands  = len(candidates)
                st.session_state.elapsed  = round(time.time() - t0, 1)
                st.session_state.run_count += 1
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

elif run_btn and not candidates:
    st.error("No candidates loaded. Please upload a file or ensure data is in the data/ folder.")


# ══════════════════════════════════════════════════════════════════════════════
# ── RESULTS ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

results   = st.session_state.results
sims      = st.session_state.sims
n_cands   = st.session_state.n_cands
elapsed   = st.session_state.elapsed
backend   = st.session_state.backend

if not results and not candidates:
    # ── Empty state ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-number">100K</div>
          <div class="metric-label">Max candidates</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-number">5</div>
          <div class="metric-label">Scoring signals</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-number">0</div>
          <div class="metric-label">API calls</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">How it works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1️⃣ Semantic Layer**\n\nSentence-transformers (all-MiniLM-L6-v2) encode both JD and candidate profiles into 384-dim vectors. Cosine similarity captures actual semantic fit — not keyword overlap.")
    with c2:
        st.info("**2️⃣ Structured Features**\n\nCareer quality, skill credibility (proficiency × endorsements × duration), behavioral engagement from 23 Redrob signals, and location/availability scoring.")
    with c3:
        st.info("**3️⃣ Weighted Ranking**\n\nCustomisable weighted combination → top-200 shortlist → second-pass reranker scanning role descriptions for IR/ML evidence → final top-N with grounded reasoning.")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="warn-box">👈 Load your candidate data from the sidebar, then click <b>Run Ranking Pipeline</b> to get started.</div>', unsafe_allow_html=True)
    st.stop()


elif candidates and not results:
    # ── Loaded, not run yet ───────────────────────────────────────────────────
    hp_count = 0
    titles = [c.get("profile",{}).get("current_title","") for c in candidates[:200]]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-number">{len(candidates):,}</div>
          <div class="metric-label">Candidates Loaded</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        yoe_vals = [c.get("profile",{}).get("years_of_experience",0) for c in candidates]
        avg_yoe = round(sum(yoe_vals)/len(yoe_vals), 1) if yoe_vals else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-number">{avg_yoe}</div>
          <div class="metric-label">Avg Years Experience</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        countries = [c.get("profile",{}).get("country","") for c in candidates]
        india_pct = round(countries.count("India")/len(countries)*100) if countries else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-number">{india_pct}%</div>
          <div class="metric-label">India-based</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-number">{top_n}</div>
          <div class="metric-label">Top-N to Rank</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">✅ Candidates loaded successfully. Click <b>🚀 Run Ranking Pipeline</b> in the sidebar to start scoring.</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# ── DASHBOARD — results ready ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

hp_in_top = sum(1 for r in results if r.get("is_honeypot", False))
dq_count  = sum(1 for r in (st.session_state.results or []) if r.get("disqualified", False))

# ── Top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-number">{n_cands:,}</div>
      <div class="metric-label">Candidates Scored</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-number">{len(results)}</div>
      <div class="metric-label">Ranked Output</div></div>""", unsafe_allow_html=True)
with k3:
    top_score = results[0]["final_score"] if results else 0
    st.markdown(f"""<div class="metric-card">
      <div class="metric-number">{top_score:.3f}</div>
      <div class="metric-label">#1 Score</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-number" style="color:{'#166534' if hp_in_top==0 else '#991b1b'}">{hp_in_top}</div>
      <div class="metric-label">Honeypots in Top-{len(results)}</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-number">{elapsed}s</div>
      <div class="metric-label">Runtime · {backend.upper() if backend else ""}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Ranked Candidates",
    "📊 Analytics",
    "🔍 Candidate Detail",
    "📥 Export",
    "ℹ️ About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKED CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_q = st.text_input("🔎 Search by title, location, or candidate ID", placeholder="e.g. NLP, Bangalore, CAND_...")
    with col_f2:
        fit_filter = st.multiselect("Filter by fit", ["🟢 Strong Fit", "🟡 Good Fit", "🟠 Partial Fit", "🔴 Weak Fit"],
                                     default=["🟢 Strong Fit", "🟡 Good Fit", "🟠 Partial Fit", "🔴 Weak Fit"])
    with col_f3:
        show_n = st.select_slider("Show", options=[5, 10, 25, 50, 100], value=min(25, len(results)))

    filtered = results
    if search_q:
        q = search_q.lower()
        filtered = [r for r in filtered if
                    q in r["candidate_id"].lower() or
                    q in (r.get("_raw_candidate",{}).get("profile",{}).get("current_title","")).lower() or
                    q in (r.get("_raw_candidate",{}).get("profile",{}).get("location","")).lower()]
    if fit_filter:
        filtered = [r for r in filtered if score_label(r["final_score"]) in fit_filter]

    filtered = filtered[:show_n]

    st.markdown(f'<div class="section-header">Top Candidates — showing {len(filtered)}</div>', unsafe_allow_html=True)

    for rec in filtered:
        c = rec.get("_raw_candidate", {})
        p = c.get("profile", {}) if c else {}
        sig = c.get("redrob_signals", {}) if c else {}
        feats = rec.get("features", {})

        rank  = rec["rank"]
        score = rec["final_score"]
        title = p.get("current_title", "Unknown")
        yoe   = p.get("years_of_experience", 0)
        loc   = p.get("location", "—")
        notice = sig.get("notice_period_days", "?")
        otw   = "✅" if sig.get("open_to_work_flag") else "⬜"
        hp    = rec.get("is_honeypot", False)
        fit   = score_label(score)

        badge_html = rank_badge_html(rank)
        bar_html   = render_score_bar(score)
        hp_tag     = '<span class="pill-red">⚠️ Honeypot Risk</span>' if hp else ""

        sem   = feats.get("semantic_similarity", 0)
        cq    = feats.get("career_quality_score", 0)
        sm    = feats.get("skill_match_score", 0)
        be    = feats.get("behavioral_engagement_score", 0)
        la    = feats.get("location_availability_score", 0)

        reasoning = rec.get("reasoning", "")

        st.markdown(f"""
        <div class="rank-card">
          <div style="display:flex; align-items:flex-start; gap:1rem;">
            <div style="min-width:36px">{badge_html}</div>
            <div style="flex:1">
              <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
                <span style="font-weight:700; font-size:1.0rem; color:#1e1b4b">{title}</span>
                <span style="color:#6b7280; font-size:0.82rem">· {yoe:.1f} yrs · {loc}</span>
                <span class="pill-gray">{fit}</span>
                {hp_tag}
              </div>
              <div style="margin:0.3rem 0; font-size:0.8rem; color:#6b7280">
                {rec['candidate_id']} &nbsp;|&nbsp; Notice: {notice}d &nbsp;|&nbsp; OTW: {otw}
              </div>
              <div style="display:flex; gap:1.5rem; margin:0.4rem 0; flex-wrap:wrap">
                <span style="font-size:0.78rem; color:#5b21b6"><b>Semantic:</b> {sem:.2f}</span>
                <span style="font-size:0.78rem; color:#5b21b6"><b>Career:</b> {cq:.2f}</span>
                <span style="font-size:0.78rem; color:#5b21b6"><b>Skill:</b> {sm:.2f}</span>
                <span style="font-size:0.78rem; color:#5b21b6"><b>Behavioral:</b> {be:.2f}</span>
                <span style="font-size:0.78rem; color:#5b21b6"><b>Location:</b> {la:.2f}</span>
              </div>
              {bar_html}
              <div style="font-size:0.8rem; color:#374151; margin-top:0.5rem; line-height:1.5">
                💬 <i>{reasoning}</i>
              </div>
            </div>
            <div style="text-align:right; min-width:70px">
              <div style="font-size:1.5rem; font-weight:700; color:#5b21b6">{score:.3f}</div>
              <div style="font-size:0.72rem; color:#9ca3af">SCORE</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)

    scores_all = [r["final_score"] for r in results]
    scores_df  = pd.DataFrame({"score": scores_all, "rank": range(1, len(scores_all)+1)})

    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            scores_df, x="score", nbins=20,
            title="Score Distribution (Top Candidates)",
            color_discrete_sequence=["#7c3aed"],
            labels={"score": "Final Score", "count": "Count"},
        )
        fig_hist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        fig_hist.update_xaxes(gridcolor="#f3f4f6")
        fig_hist.update_yaxes(gridcolor="#f3f4f6")
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=scores_df["rank"], y=scores_df["score"],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=4, color="#5b21b6"),
            name="Score",
        ))
        fig_line.update_layout(
            title="Score by Rank",
            plot_bgcolor="white", paper_bgcolor="white",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Rank", yaxis_title="Score",
        )
        fig_line.update_xaxes(gridcolor="#f3f4f6")
        fig_line.update_yaxes(gridcolor="#f3f4f6")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="section-header">Feature Breakdown — Top 10</div>', unsafe_allow_html=True)

    feat_keys = ["semantic_similarity", "career_quality_score", "skill_match_score",
                 "behavioral_engagement_score", "location_availability_score"]
    feat_labels = ["Semantic", "Career", "Skill", "Behavioral", "Location"]

    top10 = results[:10]
    cids  = [r["candidate_id"] for r in top10]
    titles_t = [r.get("_raw_candidate",{}).get("profile",{}).get("current_title","")[:20] for r in top10]
    labels = [f"#{r['rank']} {t}" for r, t in zip(top10, titles_t)]

    fig_bar = go.Figure()
    colors_b = ["#7c3aed", "#5b21b6", "#4f46e5", "#2563eb", "#0891b2"]
    for fk, fl, col in zip(feat_keys, feat_labels, colors_b):
        vals = [r["features"].get(fk, 0) for r in top10]
        fig_bar.add_trace(go.Bar(name=fl, x=labels, y=vals, marker_color=col))

    fig_bar.update_layout(
        barmode="group", height=400,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=80),
        xaxis_tickangle=-35,
    )
    fig_bar.update_xaxes(gridcolor="#f3f4f6")
    fig_bar.update_yaxes(gridcolor="#f3f4f6", range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-header">Experience Band</div>', unsafe_allow_html=True)
        yoe_vals = []
        for r in results:
            yoe = r.get("_raw_candidate",{}).get("profile",{}).get("years_of_experience", 0)
            yoe_vals.append(yoe)

        def band(y):
            if y < 3: return "<3 yrs"
            if y < 5: return "3–5 yrs"
            if y <= 9: return "5–9 yrs ✓"
            if y <= 12: return "9–12 yrs"
            return ">12 yrs"

        band_counts = pd.Series([band(y) for y in yoe_vals]).value_counts()
        fig_pie = px.pie(
            values=band_counts.values,
            names=band_counts.index,
            color_discrete_sequence=["#7c3aed","#5b21b6","#4f46e5","#2563eb","#9ca3af"],
            hole=0.4,
        )
        fig_pie.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0),
                               paper_bgcolor="white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">Semantic Similarity Distribution</div>', unsafe_allow_html=True)
        if sims is not None:
            sim_sample = sims[:5000] if len(sims) > 5000 else sims
            fig_sim = px.histogram(
                x=sim_sample, nbins=30,
                labels={"x": "Cosine Similarity", "count": "Count"},
                color_discrete_sequence=["#2563eb"],
            )
            fig_sim.update_layout(
                height=280, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=20,r=20,t=20,b=20), showlegend=False,
            )
            fig_sim.add_vline(x=float(np.mean(sim_sample)), line_dash="dash",
                               line_color="#7c3aed",
                               annotation_text=f"mean={float(np.mean(sim_sample)):.3f}")
            fig_sim.update_xaxes(gridcolor="#f3f4f6")
            fig_sim.update_yaxes(gridcolor="#f3f4f6")
            st.plotly_chart(fig_sim, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CANDIDATE DETAIL
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Deep-dive a Specific Candidate</div>', unsafe_allow_html=True)

    candidate_options = {
        f"#{r['rank']} · {r['candidate_id']} · {r.get('_raw_candidate',{}).get('profile',{}).get('current_title','')[:30]}": r
        for r in results
    }
    selected_label = st.selectbox("Select candidate", list(candidate_options.keys()))
    rec = candidate_options[selected_label]

    c   = rec.get("_raw_candidate", {})
    p   = c.get("profile", {}) if c else {}
    sig = c.get("redrob_signals", {}) if c else {}
    feats  = rec.get("features", {})
    career = c.get("career_history", []) if c else []
    skills = c.get("skills", []) if c else []
    edu    = c.get("education", []) if c else []

    # Header row
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Rank", f"#{rec['rank']}")
    with d2:
        st.metric("Final Score", f"{rec['final_score']:.4f}")
    with d3:
        st.metric("Years Experience", p.get("years_of_experience", "—"))
    with d4:
        st.metric("Notice Period", f"{sig.get('notice_period_days','?')}d")

    cl, cr = st.columns([1.2, 1])

    with cl:
        st.markdown("**Profile**")
        st.markdown(f"**{p.get('current_title','—')}** at {p.get('current_company','—')}")
        st.caption(f"📍 {p.get('location','—')} · {p.get('country','—')}")
        if p.get("summary"):
            st.markdown(p["summary"][:400] + ("..." if len(p.get("summary","")) > 400 else ""))

        st.markdown("**Career History**")
        for role in career[:4]:
            with st.container():
                st.markdown(f"🏢 **{role.get('title','—')}** at {role.get('company','—')}")
                st.caption(f"{role.get('start_date','?')} → {role.get('end_date','Present')} ({role.get('duration_months',0)}m) · {role.get('industry','—')}")
                if role.get("description"):
                    st.caption(role["description"][:180] + "...")

        st.markdown("**Education**")
        for e in edu:
            st.markdown(f"🎓 {e.get('degree','—')} in {e.get('field_of_study','—')} · {e.get('institution','—')} ({e.get('end_year','—')})")

    with cr:
        st.markdown("**Feature Scores**")
        feat_display = {
            "Semantic Similarity":       feats.get("semantic_similarity", 0),
            "Career Quality":            feats.get("career_quality_score", 0),
            "Skill Match":               feats.get("skill_match_score", 0),
            "Behavioral Engagement":     feats.get("behavioral_engagement_score", 0),
            "Location / Availability":   feats.get("location_availability_score", 0),
            "Honeypot Score":            feats.get("honeypot_score", 0),
        }
        for feat, val in feat_display.items():
            bar_col = "#ef4444" if feat == "Honeypot Score" and val > 0.4 else "#7c3aed"
            pct = int(val * 100)
            st.markdown(f"""
            <div style="margin-bottom:0.5rem">
              <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:2px">
                <span style="color:#374151; font-weight:500">{feat}</span>
                <span style="color:#5b21b6; font-weight:700">{val:.3f}</span>
              </div>
              <div style="background:#f3f4f6; border-radius:4px; height:8px">
                <div style="width:{pct}%; height:8px; border-radius:4px; background:{bar_col}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Behavioral Signals**", unsafe_allow_html=True)
        bsig = {
            "Response Rate":     f"{sig.get('recruiter_response_rate',0):.0%}",
            "GitHub Activity":   sig.get("github_activity_score", "N/A"),
            "Saved 30d":         sig.get("saved_by_recruiters_30d", 0),
            "Profile Complete":  f"{sig.get('profile_completeness_score',0):.0f}%",
            "Interview Rate":    f"{sig.get('interview_completion_rate',0):.0%}",
            "Verified Email":    "✅" if sig.get("verified_email") else "❌",
            "LinkedIn":          "✅" if sig.get("linkedin_connected") else "❌",
            "Open to Work":      "✅" if sig.get("open_to_work_flag") else "❌",
        }
        for k, v in bsig.items():
            st.markdown(f"<span style='color:#6b7280; font-size:0.8rem'>{k}</span>&nbsp;&nbsp;<b style='color:#1e1b4b; font-size:0.85rem'>{v}</b>", unsafe_allow_html=True)

        st.markdown("<br>**Top Skills**", unsafe_allow_html=True)
        for sk in sorted(skills, key=lambda s: s.get("endorsements",0), reverse=True)[:8]:
            prof_color = {"expert":"#5b21b6","advanced":"#4f46e5","intermediate":"#6b7280","beginner":"#9ca3af"}.get(sk.get("proficiency",""),"#9ca3af")
            st.markdown(f"""<span style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:2px 8px; font-size:0.77rem; color:{prof_color}; margin:2px; display:inline-block'>
              <b>{sk.get('name','')}</b> · {sk.get('proficiency','')} · {sk.get('duration_months',0)}m · {sk.get('endorsements',0)} end.</span>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**AI Reasoning:** {rec.get('reasoning','')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("#### 📄 Submission CSV (hackathon format)")
        st.caption("Exactly 100 rows · candidate_id · rank · score · reasoning")

        if len(results) < 100:
            st.warning(f"Only {len(results)} candidates — need 100 for valid submission. Increase Top-N and re-run.")
        else:
            buf = io.StringIO()
            writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rec in results[:100]:
                writer.writerow([
                    rec["candidate_id"],
                    rec["rank"],
                    f"{rec['final_score']:.6f}",
                    rec.get("reasoning","").replace("\n"," "),
                ])
            st.download_button(
                "⬇️ Download submission.csv",
                data=buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

    with ex2:
        st.markdown("#### 🔍 Debug CSV (all sub-scores)")
        st.caption("All feature scores, honeypot flags, reranker delta")

        debug_rows = []
        for rec in results:
            c  = rec.get("_raw_candidate", {})
            p  = c.get("profile", {}) if c else {}
            feats = rec.get("features", {})
            debug_rows.append({
                "rank":        rec["rank"],
                "candidate_id": rec["candidate_id"],
                "final_score": round(rec["final_score"], 6),
                "current_title": p.get("current_title",""),
                "yoe":         p.get("years_of_experience",""),
                "location":    p.get("location",""),
                "semantic_similarity":          round(feats.get("semantic_similarity",0), 4),
                "career_quality_score":         round(feats.get("career_quality_score",0), 4),
                "skill_match_score":            round(feats.get("skill_match_score",0), 4),
                "behavioral_engagement_score":  round(feats.get("behavioral_engagement_score",0), 4),
                "location_availability_score":  round(feats.get("location_availability_score",0), 4),
                "honeypot_score":               round(feats.get("honeypot_score",0), 4),
                "is_honeypot":  rec.get("is_honeypot", False),
                "disqualified": rec.get("disqualified", False),
                "reranker_delta": round(rec.get("reranker_delta",0), 4),
                "reasoning":   rec.get("reasoning",""),
            })

        debug_df = pd.DataFrame(debug_rows)
        st.download_button(
            "⬇️ Download debug.csv",
            data=debug_df.to_csv(index=False),
            file_name="debug_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📊 Preview — Top 20 in table format")
    preview_df = pd.DataFrame([{
        "Rank":  r["rank"],
        "ID":    r["candidate_id"],
        "Score": round(r["final_score"], 4),
        "Title": r.get("_raw_candidate",{}).get("profile",{}).get("current_title","")[:35],
        "YOE":   r.get("_raw_candidate",{}).get("profile",{}).get("years_of_experience",""),
        "Location": r.get("_raw_candidate",{}).get("profile",{}).get("location","")[:20],
        "Fit":   score_label(r["final_score"]),
        "Reasoning": r.get("reasoning","")[:80] + "...",
    } for r in results[:20]])
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    <div class="section-header">About This System</div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
**Team S2 · Shikher Jain**
*Redrob Hackathon — India.Runs · Data & AI Challenge*

---

#### Scoring Formula
```
final_score =
    0.30 × semantic_similarity
  + 0.25 × career_quality_score
  + 0.20 × skill_match_score
  + 0.15 × behavioral_engagement_score
  + 0.10 × location_availability_score
```
Then:
- `honeypot_score ≥ 0.5` → multiply by `(1 − hp_score)`
- Hard disqualified title / 100% consulting → `score = 0.001`

Weights are fully configurable in the sidebar.

---

#### Pipeline Steps
1. Load & validate candidates from JSONL / JSON
2. Encode JD + candidates via sentence-transformers (all-MiniLM-L6-v2)
3. Compute cosine similarity for all candidates
4. Extract 5 structured feature scores per candidate
5. Weighted combination → first-pass rank → top-200
6. Second-pass reranker scans role descriptions for IR/ML evidence
7. Generate fact-grounded reasoning per candidate
8. Export submission.csv
        """)

    with a2:
        st.markdown("""
#### Tech Stack
| Component | Technology |
|-----------|-----------|
| Semantic layer | sentence-transformers all-MiniLM-L6-v2 |
| Fallback | TF-IDF + TruncatedSVD (LSA) |
| Similarity | scikit-learn cosine_similarity |
| Feature eng. | Pure Python, zero dependencies |
| Cache | numpy .npy fingerprint cache |
| UI | Streamlit + Plotly |
| Export | CSV (hackathon-spec compliant) |

#### Compute Constraints Met
- ✅ CPU only — no GPU
- ✅ No API calls during ranking
- ✅ Embedding cache → re-runs in <2 min
- ✅ Peak memory ~3–4 GB (within 16 GB limit)

#### Key Design Decisions
- Skills scored by `proficiency × trust` (endorsements + duration), not keyword presence
- Platform assessment scores override self-reported proficiency
- Recruiter response rate carries 25% of behavioral weight (JD explicitly calls this out)
- Honeypot penalty is a smooth multiplier, not a hard blocklist — avoids brittleness
- Reasoning is fact-grounded from actual JSON fields, zero LLM inference

---
*GitHub: github.com/Shikher-jain/Redrob_AI*
        """)
