"""
vector_service.py
=================
Production-grade Qdrant-backed vector store for the Legal AI system.

Architecture
------------
- Embedding  : Google Gemini Embedding API (gemini-embedding-001, 3072-dim)
- Vector DB  : Qdrant (local Docker or Qdrant Cloud)
- Collection : "legal_documents"  — HNSW + Cosine similarity
- Payload    : Structured legal metadata per chunk
- Operations : upsert, semantic search, filtered search, delete by case

Docker quick-start (local persistence):
    docker run -d -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict, List, Optional

# ── Environment ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass

# ── Sentence Transformers (embeddings) ───────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Qdrant ───────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "legal_documents"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Qwen3-Embedding-0.6B typically produces 1536-dim vectors (or up to 3584 depending on max pooling, we detect dynamically)
FALLBACK_VECTOR_DIM = 1536


def _strip(value: Optional[str]) -> Optional[str]:
    """Strip extraneous quotes / spaces from env-var strings."""
    return value.strip("'\" ") if value else None


# ─────────────────────────────────────────────────────────────────────────────
# LegalVectorStore — public API
# ─────────────────────────────────────────────────────────────────────────────


class LegalVectorStore:
    """
    Qdrant-backed vector store for Indian legal documents.

    Public methods
    --------------
    add_documents(chunks)           — upsert a list of document chunks
    search(query, limit)            — pure semantic search
    filtered_search(query, filters) — semantic search + metadata pre-filter
    delete_by_case(case_title)      — delete all vectors for a case
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        # Allow full URL override via env (useful for Qdrant Cloud)
        env_url = _strip(os.environ.get("QDRANT_URL"))
        env_api_key = _strip(os.environ.get("QDRANT_API_KEY"))
        env_host = _strip(os.environ.get("QDRANT_HOST")) or qdrant_host
        env_port_str = os.environ.get("QDRANT_PORT")
        env_port = int(env_port_str) if env_port_str else qdrant_port

        effective_url = env_url or qdrant_url
        effective_api_key = env_api_key or qdrant_api_key

        self.collection_name = collection_name
        self._vector_dim: Optional[int] = None  # resolved on first embed

        # ── Qdrant client ─────────────────────────────────────────────────
        try:
            if effective_url:
                self._qdrant = QdrantClient(
                    url=effective_url,
                    api_key=effective_api_key,
                    timeout=5, # Short timeout for auto-fallback
                )
                logger.info("[QDRANT] Connected via URL: %s", effective_url)
            else:
                # Try Docker/Host connection
                self._qdrant = QdrantClient(
                    host=env_host,
                    port=env_port,
                    api_key=effective_api_key,
                    timeout=2, # Very short timeout to detect if Docker is missing
                )
                # Check if connection actually works
                self._qdrant.get_collections()
                logger.info("[QDRANT] Connected to Docker at %s:%s", env_host, env_port)
        except Exception:
            # FALLBACK: Use local directory storage if Docker is not running
            local_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "qdrant_local")
            )
            os.makedirs(local_path, exist_ok=True)
            self._qdrant = QdrantClient(path=local_path)
            logger.warning("[QDRANT] Docker not found. Falling back to LOCAL STORAGE at: %s", local_path)

        # ── Sentence Transformer Embedding client ─────────────────────────
        try:
            # Optionally use HF_TOKEN from environment if set
            hf_token = _strip(os.environ.get("HF_TOKEN"))
            
            logger.info(f"[QDRANT] Loading embedding model: {DEFAULT_EMBED_MODEL}...")
            # Initialize the local SentenceTransformer model
            self._embed_model = SentenceTransformer(
                DEFAULT_EMBED_MODEL, 
                token=hf_token if hf_token else None
            )
            logger.info("[QDRANT] Embedding model loaded successfully.")
        except Exception as e:
            self._embed_model = None
            logger.error(f"[QDRANT] Failed to load embedding model: {e}")

        # Ensure collection exists (lazy creation with correct params)
        self._ensure_collection()

    # ── Embedding ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        """Return a float vector for *text*, or [] on failure."""
        if not self._embed_model:
            return []
        try:
            # Add instruction prefix if required by some models (Qwen generally handles raw text fine, 
            # but we pass it directly as per standard sentence-transformers usage)
            vector = self._embed_model.encode(text).tolist()
            
            # Cache dimension from first successful embed
            if self._vector_dim is None and vector:
                self._vector_dim = len(vector)
                logger.info("[QDRANT] Detected embedding dim = %d", self._vector_dim)
            return vector
        except Exception as exc:
            logger.error("[QDRANT] Embedding error: %s", exc)
            return []

    # ── Collection lifecycle ──────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        try:
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if self.collection_name not in existing:
                # We must know the vector dimension before creating.
                # Use a probe embed if dimension not yet cached.
                dim = self._vector_dim or self._probe_dim()
                self._qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                            full_scan_threshold=10_000,
                        ),
                    ),
                )
                logger.info(
                    "[QDRANT] Created collection '%s' (dim=%d, cosine/HNSW)",
                    self.collection_name,
                    dim,
                )
            else:
                logger.info("[QDRANT] Collection '%s' already exists.", self.collection_name)
        except Exception as exc:
            logger.error("[QDRANT] Could not ensure collection: %s", exc)

    def _probe_dim(self) -> int:
        """Embed a short probe string to discover the model dimension."""
        vec = self._embed("probe")
        return len(vec) if vec else FALLBACK_VECTOR_DIM

    # ── Public API ────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Upsert a list of document chunks into Qdrant.

        Each *chunk* dict must contain:
            text          (str)  — the raw text content
            chunk_id      (str)  — unique ID (will be used as Qdrant point ID)

        And should (optionally) contain any of these metadata keys:
            case_title, court, year, act, section, jurisdiction, judge

        Returns the number of points successfully upserted.
        """
        points: List[PointStruct] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue

            vector = self._embed(text)
            if not vector:
                logger.warning("[QDRANT] Skipping chunk (empty vector): %s", chunk.get("chunk_id"))
                continue

            # Build payload — all structured metadata for filtered search
            payload: Dict[str, Any] = {
                "text": text,
                "case_title":   chunk.get("case_title") or chunk.get("case_name") or "",
                "court":        chunk.get("court") or "",
                "year":         int(chunk["year"]) if chunk.get("year") else None,
                "act":          chunk.get("act") or "",
                "section":      chunk.get("section") or "",
                "jurisdiction": chunk.get("jurisdiction") or "",
                "judge":        chunk.get("judge") or "",
                "chunk_id":     chunk.get("chunk_id") or str(uuid.uuid4()),
                # Reranker hook: preserve source URL / relevance signals
                "source_url":   chunk.get("source_url") or None,
                "keyword_tags": chunk.get("keyword_tags") or [],
            }

            # Derive a deterministic UUID from chunk_id for Qdrant point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, payload["chunk_id"]))

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if not points:
            logger.warning("[QDRANT] add_documents: no valid points to upsert.")
            return 0

        try:
            self._qdrant.upsert(collection_name=self.collection_name, points=points)
            logger.info("[QDRANT] Upserted %d point(s).", len(points))
            return len(points)
        except Exception as exc:
            logger.error("[QDRANT] Upsert failed: %s", exc)
            return 0

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Pure semantic vector search — no metadata filtering.

        Returns a list of result dicts (payload + score).
        """
        return self._vector_search(query, limit=limit, qdrant_filter=None)

    def filtered_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional Qdrant metadata pre-filtering.

        Supported filter keys
        ---------------------
        court        (str)  — exact match,  e.g. "Supreme Court"
        act          (str)  — exact match,  e.g. "NDPS Act"
        jurisdiction (str)  — exact match
        section      (str)  — exact match
        year_gte     (int)  — year >= value, e.g. 2018
        year_lte     (int)  — year <= value
        judge        (str)  — exact match

        Missing / None filter values are silently ignored.
        Falls back to keyword search if vector embed fails.
        """
        qdrant_filter = self._build_filter(filters or {})
        results = self._vector_search(query, limit=limit, qdrant_filter=qdrant_filter)

        # Keyword fallback if vector search returned nothing
        if not results:
            logger.info("[QDRANT] Vector search empty — falling back to keyword scroll.")
            results = self._keyword_fallback(query, filters or {}, limit=limit)

        return results

    def delete_by_case(self, case_title: str) -> int:
        """
        Delete all vector points whose payload.case_title matches *case_title*.

        Returns number of points deleted (approximate from Qdrant response).
        """
        try:
            result = self._qdrant.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="case_title",
                            match=MatchValue(value=case_title),
                        )
                    ]
                ),
            )
            logger.info("[QDRANT] delete_by_case '%s': %s", case_title, result)
            return 1  # Qdrant returns operation_id, not count; signal success
        except Exception as exc:
            logger.error("[QDRANT] delete_by_case failed: %s", exc)
            return 0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        limit: int,
        qdrant_filter: Optional[Filter],
    ) -> List[Dict[str, Any]]:
        query_vector = self._embed(query)
        if not query_vector:
            return []

        try:
            # query_points is the modern API (Qdrant 1.10+) 
            # replacing the older .search()
            hits = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.error("[QDRANT] Search error: %s", exc)
            return []

        return [self._hit_to_result(h) for h in hits]

    def _hit_to_result(self, hit: Any) -> Dict[str, Any]:
        """Convert a Qdrant ScoredPoint to a standardised result dict."""
        payload = hit.payload or {}
        # For query_points the score might be hit.score or hit.values
        score = getattr(hit, "score", 0.0)
        relevance_pct = max(0, min(100, int(score * 100)))

        return {
            # Downstream-compatible keys (used by retriever_service + orchestrator)
            "act":          payload.get("act") or "Legal Document",
            "section":      payload.get("section") or payload.get("year") or "N/A",
            "title":        payload.get("case_title") or "Unknown Case",
            "content":      payload.get("text") or "",
            "summary":      f"Retrieved from Qdrant (Relevance: {relevance_pct}%)",
            "source_url":   payload.get("source_url"),
            # Full metadata — available for reranking layer
            "metadata": {
                "case_title":   payload.get("case_title"),
                "court":        payload.get("court"),
                "year":         payload.get("year"),
                "act":          payload.get("act"),
                "section":      payload.get("section"),
                "jurisdiction": payload.get("jurisdiction"),
                "judge":        payload.get("judge"),
                "chunk_id":     payload.get("chunk_id"),
                "keyword_tags": payload.get("keyword_tags", []),
                "score":        score,
            },
        }

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> Optional[Filter]:
        """
        Translate a plain dict of filter criteria into a Qdrant Filter object.
        Returns None if no valid conditions are present.
        """
        must_conditions = []

        # Exact-match fields
        for field in ("court", "act", "section", "jurisdiction", "judge"):
            val = filters.get(field)
            if val:
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=str(val)))
                )

        # Numeric range on 'year'
        year_gte = filters.get("year_gte") or filters.get("year")
        year_lte = filters.get("year_lte")
        if year_gte or year_lte:
            must_conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=int(year_gte) if year_gte else None,
                        lte=int(year_lte) if year_lte else None,
                    ),
                )
            )

        return Filter(must=must_conditions) if must_conditions else None

    def _keyword_fallback(
        self, query: str, filters: Dict[str, Any], limit: int
    ) -> List[Dict[str, Any]]:
        """
        Scroll through Qdrant payloads and do simple substring matching.
        Used only when vector search returns nothing (no embedding or cold start).
        """
        qdrant_filter = self._build_filter(filters)
        try:
            records, _ = self._qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=limit * 10,  # over-fetch for keyword re-ranking
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.error("[QDRANT] Keyword fallback scroll error: %s", exc)
            return []

        keywords = query.lower().split()
        scored = []
        for rec in records:
            payload = rec.payload or {}
            text = (payload.get("text") or "").lower()
            score = sum(1 for kw in keywords if kw in text)
            if score:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [self._hit_to_result(type("Hit", (), {"payload": r.payload, "score": s / 10})())
                for s, r in scored[:limit]]


# ── Backwards-compatible shim ─────────────────────────────────────────────────
# Keeps any legacy code that still imports VectorService working without edits.

class VectorService(LegalVectorStore):
    """
    Backwards-compatibility alias for LegalVectorStore.
    Provides the original upsert_judgment_chunk / query_similar_judgments API
    on top of the new Qdrant backend.
    """

    def upsert_judgment_chunk(
        self, id: str, text: str, metadata: Dict[str, Any]
    ) -> None:
        """Legacy API — maps to add_documents."""
        chunk = {
            "chunk_id":   id,
            "text":       text,
            "case_title": metadata.get("case_name") or metadata.get("case_title") or "",
            "court":      metadata.get("court") or "Supreme Court of India",
            "year":       metadata.get("case_year") or metadata.get("year"),
            "act":        metadata.get("act") or "",
            "section":    metadata.get("section") or "",
            "jurisdiction": metadata.get("jurisdiction") or "India",
            "judge":      metadata.get("judge") or "",
            "source_url": metadata.get("source_url"),
        }
        self.add_documents([chunk])

    def query_similar_judgments(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Legacy API — maps to search."""
        return self.search(query, limit=limit)
