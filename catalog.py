"""
Phase 2 - Deeplink catalog + retrieval.

Loads deeplinks.json exactly as provided (no assumed fields) and exposes
dual retrieval over each entry's retrieval text, which is built ONLY from
`description`, `message`, `qna_description` and `originalType` (see
CatalogEntry.search_text). The masked URIs (`deeplink`, `validation.deeplink`)
are opaque hashes and are never indexed or embedded.

  - BM25 (rank_bm25.BM25Okapi) - required, lexical signal.
  - Dense embeddings (sentence-transformers, default model
    "all-MiniLM-L6-v2") + cosine similarity - required, semantic signal.
    Embeddings for every catalog entry are computed ONCE, at
    DeeplinkCatalog construction (index-build/startup) time, and cached in
    memory (`self._entry_vectors`). The request path (`search()`) only
    encodes the single incoming query string; it never re-embeds the
    catalog and never downloads a model.

TF-IDF cosine similarity remains available ONLY as an explicit, opt-in
offline fallback for the dense signal (see `allow_tfidf_fallback` /
DenseRetrievalUnavailableError below) -- it is not a silent substitute for
dense retrieval and callers can always see which mode is active via
`DeeplinkCatalog.retrieval_mode`.

The two active signals (BM25 + whichever vector signal is active) are
combined with Reciprocal Rank Fusion (RRF), which is robust to the two
signals living on different scales and is a standard way to blend lexical +
vector retrieval.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

DUMMY_DEEPLINK_URI = "voiceassist://dummy_positive"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

logger = logging.getLogger(__name__)


class DenseRetrievalUnavailableError(RuntimeError):
    """
    Raised at catalog build (startup) time when the required dense embedding
    model cannot be loaded and no explicit fallback was requested. This is a
    controlled initialization/configuration error: the service should fail
    to start (or report unhealthy on GET /health) rather than silently
    degrade to a weaker retrieval mode.
    """


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass(frozen=True)
class CatalogEntry:
    """Mirrors deeplinks.json fields verbatim. No fields are invented."""

    id: str
    deeplink: str          # top-level action URI, e.g. voiceassist://masked/act/...
    description: str
    message: str
    originalType: Optional[str]
    control_type: Optional[int]
    qna_description: str
    validation: Optional[dict]  # {deeplink, key, resultType, condition, value} or None

    @property
    def search_text(self) -> str:
        """
        Retrieval text is built from description, message, qna_description,
        and originalType. The masked `deeplink` URI itself is NEVER part of
        this text -- it's an opaque hash and matching on it would be
        meaningless (see catalog's own _readme: match on description/
        message/qna_description, then copy the URI verbatim).
        """
        parts = [self.description, self.message, self.qna_description, self.originalType or ""]
        return " ".join(p for p in parts if p)

    @property
    def is_dummy(self) -> bool:
        return self.id == "DL-DUMMY" or self.deeplink == DUMMY_DEEPLINK_URI


class DenseEncoder:
    """
    Interface for the vector-retrieval signal. Production uses
    SentenceTransformerEncoder; tests inject a deterministic fake so unit
    tests don't require a network download or GPU/model load.
    """

    name: str = "abstract"

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEncoder(DenseEncoder):
    """
    Real dense embedding signal. Loads the model once at construction
    (startup); `encode()` at request time only embeds the query string(s)
    passed to it -- it never re-embeds the catalog and never re-downloads
    the model.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Imported lazily so environments that only ever use the fallback
        # path (or inject a fake encoder in tests) don't need the package
        # installed to import this module at all.
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


class TfidfFallbackEncoder(DenseEncoder):
    """
    Explicit, opt-in fallback used ONLY when the real dense encoder is
    unavailable and the caller has explicitly allowed degraded operation
    (see DeeplinkCatalog(allow_tfidf_fallback=True)). This is lexical-vector
    retrieval, not semantic embeddings, and DeeplinkCatalog.retrieval_mode
    always reports it as "tfidf_fallback" so callers/observability never
    mistake it for the dense path.
    """

    name = "tfidf_fallback"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, lowercase=False, token_pattern=None
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfFallbackEncoder.fit() must be called before encode()")
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray()


def _cosine_sim_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query_vec: (d,), matrix: (n, d) -> (n,) cosine similarities."""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m_norm @ q_norm


class DeeplinkCatalog:
    """Loads deeplinks.json and provides dual-retrieval ranked search."""

    def __init__(
        self,
        path: str | Path,
        dense_encoder: Optional[DenseEncoder] = None,
        allow_tfidf_fallback: bool = False,
        dense_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        dense_encoder: inject a DenseEncoder (e.g. in tests). If None,
            attempts to construct the real SentenceTransformerEncoder.
        allow_tfidf_fallback: if the dense encoder can't be constructed and
            this is False (default), raises DenseRetrievalUnavailableError --
            a controlled startup failure. If True, logs a warning and falls
            back to TfidfFallbackEncoder, and `retrieval_mode` reports
            "tfidf_fallback" so this is never silently mistaken for dense
            retrieval.
        """
        raw = json.loads(Path(path).read_text())
        self.readme: str = raw.get("_readme", "")
        self.count: int = raw.get("count", len(raw.get("deeplinks", [])))

        self.entries: List[CatalogEntry] = []
        self.dummy_entry: Optional[CatalogEntry] = None
        for e in raw["deeplinks"]:
            entry = CatalogEntry(
                id=e["id"],
                deeplink=e["deeplink"],
                description=e.get("description", ""),
                message=e.get("message", ""),
                originalType=e.get("originalType"),
                control_type=e.get("control_type"),
                qna_description=e.get("qna_description", ""),
                validation=e.get("validation"),
            )
            self.entries.append(entry)
            if entry.is_dummy:
                self.dummy_entry = entry

        if self.dummy_entry is None:
            raise ValueError("deeplinks.json has no DL-DUMMY / dummy_positive entry")

        # Searchable entries exclude the dummy placeholder itself.
        self._searchable = [e for e in self.entries if not e.is_dummy]
        corpus = [e.search_text for e in self._searchable]

        # --- Required signal 1: BM25 (rank_bm25) ---
        self._bm25 = BM25Okapi([_tokenize(doc) for doc in corpus])

        # --- Required signal 2: dense embeddings, precomputed at build time ---
        self.retrieval_mode: str
        if dense_encoder is not None:
            self._dense_encoder = dense_encoder
            self.retrieval_mode = dense_encoder.name
        else:
            try:
                self._dense_encoder = SentenceTransformerEncoder(dense_model_name)
                self.retrieval_mode = self._dense_encoder.name
            except Exception as exc:  # ImportError, OSError (model fetch), etc.
                if not allow_tfidf_fallback:
                    raise DenseRetrievalUnavailableError(
                        f"Dense embedding model '{dense_model_name}' unavailable "
                        f"({exc!r}) and allow_tfidf_fallback=False. Failing "
                        f"startup rather than silently degrading retrieval "
                        f"quality. Pass allow_tfidf_fallback=True to accept "
                        f"degraded (lexical-only-vector) retrieval instead."
                    ) from exc
                logger.warning(
                    "Dense embedding model unavailable (%r); falling back to "
                    "TF-IDF. retrieval_mode='tfidf_fallback' -- this is NOT "
                    "semantic retrieval.",
                    exc,
                )
                fallback = TfidfFallbackEncoder()
                self._dense_encoder = fallback
                self.retrieval_mode = fallback.name

        # A TF-IDF encoder (explicit fallback, or injected for offline
        # tests) must be fitted on exactly the corpus that gets indexed.
        if isinstance(self._dense_encoder, TfidfFallbackEncoder) and not self._dense_encoder.is_fitted:
            self._dense_encoder.fit(corpus)

        # Precompute catalog embeddings ONCE, here at build time. The
        # request path (search()) only encodes the query.
        self._entry_vectors = self._dense_encoder.encode(corpus)

    def _bm25_scores(self, query: str) -> List[float]:
        return list(self._bm25.get_scores(_tokenize(query)))

    def _vector_scores(self, query: str) -> List[float]:
        # Request-path work: encode only the query; the catalog matrix was
        # built once at startup and is reused here unchanged.
        qvec = self._dense_encoder.encode([query])[0]
        return list(_cosine_sim_matrix(qvec, self._entry_vectors))

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List["RankedMatch"]:
        """
        Dual retrieval (BM25 + dense embeddings, or BM25 + TF-IDF only in
        explicit fallback mode) fused with Reciprocal Rank Fusion. Returns
        top_k RankedMatch objects, best first, each carrying BOTH the fused
        rank-based signal and the raw underlying scores.

        IMPORTANT -- rank agreement alone is not sufficient evidence of a
        real match. If a query shares no vocabulary with the catalog at all
        (e.g. gibberish, or a domain the catalog doesn't cover), BM25 can
        return an all-zero score vector, and TF-IDF/embeddings can likewise
        return a near-zero or all-tied similarity vector. `sorted(...,
        reverse=True)` over an all-tied array is a stable sort, so it
        silently returns the FIRST catalog entry (currently DL-0001) as
        "rank 0" even though there's no real evidence for it. Ranks 0/0
        would then look like a confident match by rank agreement alone.
        Callers MUST also check the raw `bm25_score` / `vector_score` on the
        returned RankedMatch (see mapper.EvidenceThresholds and
        mapper.has_sufficient_evidence) before trusting a match.

        NOTE on polarity: even with real dense embeddings, catalog entries
        that differ mainly by enable/disable polarity (e.g. DL-0541
        "Disables data backup..." vs DL-0542 "Enables data backup...") can
        embed very close together. Callers that care about polarity
        (mapper.py) re-rank the returned candidates using the catalog's own
        `originalType` field (onURL/onClickURL vs offURL) rather than
        trusting fused order alone.
        """
        bm25_scores = self._bm25_scores(query)
        vector_scores = self._vector_scores(query)

        bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        vector_order = sorted(range(len(vector_scores)), key=lambda i: vector_scores[i], reverse=True)

        bm25_rank_of = {doc_i: r for r, doc_i in enumerate(bm25_order)}
        vector_rank_of = {doc_i: r for r, doc_i in enumerate(vector_order)}

        fused: Dict[int, float] = {}
        for doc_i in range(len(self._searchable)):
            r_bm25 = bm25_rank_of.get(doc_i, len(self._searchable))
            r_vec = vector_rank_of.get(doc_i, len(self._searchable))
            fused[doc_i] = 1.0 / (rrf_k + r_bm25 + 1) + 1.0 / (rrf_k + r_vec + 1)

        ranked_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
        return [
            RankedMatch(
                entry=self._searchable[i],
                fused_score=fused[i],
                bm25_rank=bm25_rank_of[i],
                vector_rank=vector_rank_of[i],
                bm25_score=float(bm25_scores[i]),
                vector_score=float(vector_scores[i]),
                retrieval_mode=self.retrieval_mode,
            )
            for i in ranked_ids
        ]


@dataclass
class RankedMatch:
    entry: CatalogEntry
    fused_score: float
    bm25_rank: int        # 0 = best
    vector_rank: int      # 0 = best (dense, or tfidf_fallback if degraded)
    bm25_score: float     # raw BM25 score, 0.0 = no lexical overlap at all
    vector_score: float   # raw cosine similarity, ~0.0 = no evidence
    retrieval_mode: str   # "sentence-transformers" or "tfidf_fallback"

    def agrees_in_top(self, n: int = 3) -> bool:
        return self.bm25_rank < n and self.vector_rank < n

    def has_score_evidence(self, min_bm25: float, min_vector: float) -> bool:
        """
        Rejects candidates whose raw scores show no real evidence, even if
        they happen to "agree" on rank (see search()'s docstring for why
        rank-only agreement is unsafe on out-of-vocabulary queries).
        """
        return (
            self.bm25_score > 0.0
            and self.vector_score > 0.0
            and self.bm25_score >= min_bm25
            and self.vector_score >= min_vector
        )
