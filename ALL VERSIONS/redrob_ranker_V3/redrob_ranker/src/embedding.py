"""
embedding.py — Semantic embeddings with local-first fallback.

Primary: sentence-transformers all-MiniLM-L6-v2 (if model is cached locally)
Fallback: TF-IDF + TruncatedSVD (LSA) — Latent Semantic Analysis

LSA is a proven, well-understood semantic method used in production IR systems
since the 1990s. It finds latent topics in text and represents documents in a
semantic space — not keyword matching.

For the hackathon on your machine: sentence-transformers will load fine (it's
just blocked in this build sandbox). The code tries ST first, falls back to LSA
if unavailable.

Both approaches:
- Return (N, D) float32 embeddings
- Are L2-normalised so cosine = dot product
- Use the same cache mechanism
"""

from __future__ import annotations
import hashlib, logging, os, pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL
from utils import candidate_text_blob

logger = logging.getLogger(__name__)
_CACHE_DIR = Path(__file__).parent.parent / "outputs" / ".embed_cache"


def _fingerprint(texts: list[str]) -> str:
    sample = "||".join(texts[:100])
    return hashlib.md5(sample.encode()).hexdigest()[:12]


def _cache_path(tag: str, n: int, h: str, ext: str = "npy") -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{tag}_{h}_{n}.{ext}"


# ── Sentence-Transformers (preferred) ────────────────────────────────────────

def _try_load_st(model_name: str):
    """Try to load sentence-transformers model. Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info("Loaded sentence-transformers model: %s (dim=%d)",
                    model_name, model.get_sentence_embedding_dimension())
        return ("st", model)
    except Exception as e:
        logger.warning("sentence-transformers unavailable (%s). Using LSA fallback.", str(e)[:80])
        return None


def _encode_st(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return emb.astype(np.float32)


# ── LSA Fallback ─────────────────────────────────────────────────────────────

_LSA_COMPONENTS = 200   # semantic dimensions; >100 captures rich semantics


def _fit_lsa(corpus: list[str]):
    """Fit TF-IDF + SVD on a corpus. Returns (vectorizer, svd) tuple."""
    logger.info("Fitting TF-IDF on %d documents ...", len(corpus))
    vec = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams for richer semantics
        min_df=2,                # ignore very rare terms
        max_df=0.95,             # ignore near-universal terms
        sublinear_tf=True,       # log(1+tf) dampens high-frequency terms
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    tfidf = vec.fit_transform(corpus)
    logger.info("TF-IDF matrix: %s, fitting SVD (k=%d) ...", tfidf.shape, _LSA_COMPONENTS)
    svd = TruncatedSVD(n_components=_LSA_COMPONENTS, random_state=42)
    svd.fit(tfidf)
    explained = svd.explained_variance_ratio_.sum()
    logger.info("SVD fit: %.1f%% variance explained", explained * 100)
    return vec, svd


def _encode_lsa(vec, svd, texts: list[str]) -> np.ndarray:
    tfidf = vec.transform(texts)
    emb   = svd.transform(tfidf).astype(np.float32)
    return normalize(emb, norm="l2")   # L2-normalise so cosine = dot


# ── Public API ────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Unified embedding engine.
    Uses sentence-transformers if available, LSA otherwise.
    Both produce L2-normalised float32 vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._backend = None
        self._st_model = None
        self._lsa_vec  = None
        self._lsa_svd  = None

        result = _try_load_st(model_name)
        if result:
            self._backend, self._st_model = result
        else:
            self._backend = "lsa"

    def fit_lsa(self, corpus: list[str]) -> None:
        """Must call before encode() when using LSA backend."""
        if self._backend == "lsa":
            h = _fingerprint(corpus)
            cache = _cache_path("lsa_model", len(corpus), h, "pkl")
            if cache.exists():
                logger.info("Loading cached LSA model ← %s", cache.name)
                with open(cache, "rb") as f:
                    self._lsa_vec, self._lsa_svd = pickle.load(f)
            else:
                self._lsa_vec, self._lsa_svd = _fit_lsa(corpus)
                with open(cache, "wb") as f:
                    pickle.dump((self._lsa_vec, self._lsa_svd), f)
                logger.info("Saved LSA model → %s", cache.name)

    def encode(
        self,
        texts: list[str],
        tag: str = "emb",
        batch_size: int = 256,
        use_cache: bool = True,
    ) -> np.ndarray:
        h     = _fingerprint(texts)
        cache = _cache_path(f"{self._backend}_{tag}", len(texts), h)

        if use_cache and cache.exists():
            logger.info("Cache hit → %s", cache.name)
            return np.load(str(cache))

        if self._backend == "st":
            emb = _encode_st(self._st_model, texts, batch_size)
        else:
            if self._lsa_vec is None:
                raise RuntimeError("Call fit_lsa(corpus) before encode() in LSA mode.")
            emb = _encode_lsa(self._lsa_vec, self._lsa_svd, texts)

        if use_cache:
            np.save(str(cache), emb)
        return emb

    @property
    def backend(self) -> str:
        return self._backend


def cosine_sims(jd_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity: JD vs all candidates. Returns shape (N,), clipped [0,1]."""
    sims = cosine_similarity(jd_emb.reshape(1, -1), cand_embs)[0]
    return np.clip(sims, 0.0, 1.0).astype(np.float32)


def build_candidate_texts(candidates: list[dict]) -> list[str]:
    return [candidate_text_blob(c) for c in candidates]
