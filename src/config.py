"""
config.py — single source of truth for all weights and constants.
Tune here; nothing else needs to change.
"""

# ── Final score weights (must sum to 1.0) ──────────────────────────────────
WEIGHTS = {
    "semantic_similarity":          0.30,
    "career_quality_score":         0.25,
    "skill_match_score":            0.20,
    "behavioral_engagement_score":  0.15,
    "location_availability_score":  0.10,
}

# ── JD experience band ──────────────────────────────────────────────────────
JD_YOE_MIN    = 5.0
JD_YOE_TARGET = 7.0
JD_YOE_MAX    = 9.0

# ── Salary band (INR LPA) for Senior AI Engineer ───────────────────────────
SALARY_MIN_LPA = 20.0
SALARY_MAX_LPA = 60.0

# ── Notice period preference ────────────────────────────────────────────────
NOTICE_PREFERRED_DAYS = 30

# ── Sentence-transformer model (CPU-friendly, 384-dim) ──────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Preferred India locations (directly from JD) ────────────────────────────
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bombay", "bangalore", "bengaluru",
}

# ── Consulting-only firms: JD explicitly disqualifies career-only consultants ─
PURE_CONSULTING = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree",
}

# ── Skills matching JD's "absolutely need" section ──────────────────────────
CORE_SKILLS = {
    "sentence-transformers", "sentence transformers", "embeddings",
    "vector search", "vector database", "vector db",
    "faiss", "qdrant", "pinecone", "weaviate", "milvus",
    "opensearch", "elasticsearch",
    "retrieval", "rag", "retrieval augmented generation",
    "information retrieval", "hybrid search",
    "ranking", "learning to rank", "ltr",
    "ndcg", "mrr", "map", "a/b testing", "ab testing",
    "transformers", "huggingface", "hugging face",
    "bert", "roberta", "sentence bert", "sbert",
    "nlp", "natural language processing",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "llm", "large language model",
    "python", "pytorch", "tensorflow",
    "evaluation framework", "offline evaluation", "online evaluation",
    "xgboost", "lightgbm",
    "recommendation", "recommender system",
}

# ── JD "nice-to-have" skills ────────────────────────────────────────────────
PREFERRED_SKILLS = {
    "distributed systems", "kafka", "spark",
    "open source", "open-source",
    "semantic search", "search",
}

# ── Outright disqualified title tokens (JD explicit) ────────────────────────
DISQUALIFIED_TITLE_TOKENS = {
    "marketing manager", "marketing",
    "operations manager",
    "hr manager", "human resources",
    "customer support", "support engineer",
    "graphic designer", "content writer", "seo",
    "mechanical engineer", "chemical engineer",
    "robotics engineer", "accountant",
    "civil engineer",
}

# ── Research-only penalty ────────────────────────────────────────────────────
RESEARCH_ONLY_TOKENS = {
    "research scientist", "research engineer",
    "phd researcher", "postdoctoral", "postdoc",
}

# ── Wrong domain (CV/speech without NLP) ────────────────────────────────────
WRONG_DOMAIN_TOKENS = {
    "computer vision engineer", "cv engineer",
    "speech engineer", "asr engineer",
}

# ── Production evidence keywords (scanned in descriptions) ──────────────────
PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "live", "users",
    "scale", "latency", "throughput", "api", "service",
    "pipeline", "real-time", "real time", "endpoint",
    "inference", "serving", "millions", "requests per second",
}

# ── ML industry identifiers ──────────────────────────────────────────────────
ML_INDUSTRIES = {
    "artificial intelligence", "machine learning", "ai", "ml",
    "nlp", "information retrieval", "data science",
    "tech", "technology", "software", "saas",
    "hr tech", "hrtech", "recruiting", "edtech", "fintech",
}
