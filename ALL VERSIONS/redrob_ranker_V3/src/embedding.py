"""
embedding.py — Semantic embeddings with robust batching and disk cache.

Primary: sentence-transformers all-MiniLM-L6-v2
Fallback: TF-IDF + TruncatedSVD (LSA) — offline, no downloads needed

Key fixes vs v1:
- Smaller batch size (64 instead of 256) to avoid Windows OOM kills
- max_seq_length=128 — candidates average 447 tokens but model truncates anyway;
  128 is the sweet spot for speed/quality on this task
- Chunked encoding with immediate numpy accumulation (no giant list in RAM)
- Progress bar per-chunk so you can see it's alive
- Cache saves after every chunk (crash-safe: partial re-runs skip done chunks)
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
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fingerprint(texts: list[str]) -> str:
    # Use first + last 50 + count as fingerprint (fast, stable)
    sample = "||".join(texts[:50] + texts[-50:] + [str(len(texts))])
    return hashlib.md5(sample.encode()).hexdigest()[:16]


def _cache_path(tag: str, h: str, ext: str = "npy") -> Path:
    return _CACHE_DIR / f"{tag}_{h}.{ext}"


# ── sentence-transformers backend ─────────────────────────────────────────────

def _try_load_st(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        # Truncate at 128 tokens: enough for job-relevance semantics,
        # avoids the long-tail blowup from candidates with verbose descriptions
        model.max_seq_length = 128
        dim = model.get_sentence_embedding_dimension()
        logger.info("sentence-transformers loaded: %s (dim=%d, max_seq=128)", model_name, dim)
        return ("st", model)
    except Exception as e:
        logger.warning("sentence-transformers failed: %s — using LSA fallback", str(e)[:100])
        return None


def _encode_st_chunked(
    model,
    texts: list[str],
    tag: str,
    batch_size: int = 64,
    chunk_size: int = 5000,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Encode in chunks of `chunk_size`, cache each chunk separately.
    If the process dies mid-run, the next run picks up from the last saved chunk.
    """
    h      = _fingerprint(texts)
    full_c = _cache_path(f"st_{tag}_full", h)

    if use_cache and full_c.exists():
        logger.info("Cache hit (full): %s", full_c.name)
        return np.load(str(full_c))

    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    chunks_emb: list[np.ndarray] = []

    for ci in range(n_chunks):
        chunk_c = _cache_path(f"st_{tag}_chunk{ci}", h)
        start   = ci * chunk_size
        end     = min(start + chunk_size, len(texts))
        chunk   = texts[start:end]

        if use_cache and chunk_c.exists():
            logger.info("  Chunk %d/%d: cache hit", ci + 1, n_chunks)
            chunks_emb.append(np.load(str(chunk_c)))
            continue

        logger.info("  Chunk %d/%d: encoding %d texts ...", ci + 1, n_chunks, len(chunk))
        emb = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        if use_cache:
            np.save(str(chunk_c), emb)
        chunks_emb.append(emb)

    full_emb = np.vstack(chunks_emb)

    if use_cache:
        np.save(str(full_c), full_emb)
        logger.info("Full embeddings saved: %s", full_c.name)

    return full_emb


# ── LSA fallback ──────────────────────────────────────────────────────────────

_LSA_DIM = 200


def _fit_lsa(corpus: list[str]):
    logger.info("Fitting TF-IDF on %d texts ...", len(corpus))
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        max_features=80_000,   # cap vocab to control RAM
    )
    mat = vec.fit_transform(corpus)
    logger.info("TF-IDF: %s | fitting SVD k=%d ...", mat.shape, _LSA_DIM)
    svd = TruncatedSVD(n_components=_LSA_DIM, random_state=42)
    svd.fit(mat)
    logger.info("SVD: %.1f%% variance explained", svd.explained_variance_ratio_.sum() * 100)
    return vec, svd


def _encode_lsa(vec, svd, texts: list[str]) -> np.ndarray:
    mat = vec.transform(texts)
    emb = svd.transform(mat).astype(np.float32)
    return normalize(emb, norm="l2")


# ── Public API ────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Unified embedding engine.
    Auto-selects sentence-transformers (better) or LSA (offline fallback).
    Both return L2-normalised float32 vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._backend   = None
        self._st_model  = None
        self._lsa_vec   = None
        self._lsa_svd   = None

        result = _try_load_st(model_name)
        if result:
            self._backend, self._st_model = result
        else:
            self._backend = "lsa"
            logger.info("Backend: LSA (TF-IDF + TruncatedSVD)")

    def fit_lsa(self, corpus: list[str]) -> None:
        """Only needed for LSA backend. No-op for sentence-transformers."""
        if self._backend != "lsa":
            return

        h     = _fingerprint(corpus)
        cache = _cache_path("lsa_model", h, "pkl")
        if cache.exists():
            logger.info("Loading cached LSA model: %s", cache.name)
            with open(cache, "rb") as f:
                self._lsa_vec, self._lsa_svd = pickle.load(f)
        else:
            self._lsa_vec, self._lsa_svd = _fit_lsa(corpus)
            with open(cache, "wb") as f:
                pickle.dump((self._lsa_vec, self._lsa_svd), f)
            logger.info("LSA model cached: %s", cache.name)

    def encode(
        self,
        texts: list[str],
        tag: str = "emb",
        batch_size: int = 64,
        use_cache: bool = True,
    ) -> np.ndarray:
        if self._backend == "st":
            return _encode_st_chunked(
                self._st_model, texts, tag,
                batch_size=batch_size,
                chunk_size=5000,
                use_cache=use_cache,
            )
        else:
            if self._lsa_vec is None:
                raise RuntimeError("Call fit_lsa(corpus) before encode() in LSA mode.")
            h     = _fingerprint(texts)
            cache = _cache_path(f"lsa_{tag}", h)
            if use_cache and cache.exists():
                logger.info("Cache hit (LSA %s): %s", tag, cache.name)
                return np.load(str(cache))
            emb = _encode_lsa(self._lsa_vec, self._lsa_svd, texts)
            if use_cache:
                np.save(str(cache), emb)
            return emb

    @property
    def backend(self) -> str:
        return self._backend


def cosine_sims(jd_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    sims = cosine_similarity(jd_emb.reshape(1, -1), cand_embs)[0]
    return np.clip(sims, 0.0, 1.0).astype(np.float32)


def build_candidate_texts(candidates: list[dict]) -> list[str]:
    return [candidate_text_blob(c) for c in candidates]
