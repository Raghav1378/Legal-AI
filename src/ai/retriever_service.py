"""
retriever_service.py
====================
Hybrid retrieval layer: keyword search over the JSON dataset +
semantic / filtered search via the Qdrant LegalVectorStore.

Retrieval priority
------------------
1. Filtered vector search  (Qdrant — semantic + metadata filters)
2. Plain semantic search   (Qdrant — no filter, broader recall)
3. Keyword search          (local dataset.json — fast statute look-up)

All branches are combined and deduplicated before returning.
"""

from __future__ import annotations

import re
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .vector_service import LegalVectorStore


class RetrieverService:
    """
    Hybrid retriever: local JSON statute DB + Qdrant vector store.
    """

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")

        with open(dataset_path, "r", encoding="utf-8") as fh:
            self.dataset: List[Dict[str, Any]] = json.load(fh)

        # Shared Qdrant-backed vector store
        self.store = LegalVectorStore()

    # ── Primary entry-point ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to *query*.

        Parameters
        ----------
        query   : natural-language legal question
        limit   : maximum number of results desired
        filters : optional metadata filters passed to filtered_search
                  Supported keys: court, act, section, jurisdiction, judge,
                                  year_gte, year_lte, year

        Returns a combined, deduplicated list of enriched document dicts.
        """
        # ── 1. Filtered vector search (Qdrant + metadata) ─────────────────
        vector_results: List[Dict[str, Any]] = []
        try:
            if filters:
                vector_results = self.store.filtered_search(
                    query, filters=filters, limit=limit
                )
            else:
                vector_results = self.store.search(query, limit=max(2, limit // 2))
        except Exception as exc:
            print(f"[WARN] Vector search failed: {exc}")

        # ── 2. Local keyword search (statute law fast path) ────────────────
        local_results = self._search_local(query, limit=limit)

        # ── 3. Combine & deduplicate ───────────────────────────────────────
        combined = self._deduplicate(vector_results + local_results)
        return [self._enrich_doc(doc) for doc in combined[:limit + 3]]

    # ── Convenience wrappers (for direct reranker / orchestrator use) ─────

    def filtered_search(
        self,
        query: str,
        filters: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Expose filtered_search directly for use by orchestrator tools."""
        try:
            results = self.store.filtered_search(query, filters=filters, limit=limit)
        except Exception as exc:
            print(f"[WARN] filtered_search failed: {exc}")
            results = []
        return [self._enrich_doc(r) for r in results]

    def delete_case_vectors(self, case_title: str) -> int:
        """Remove all Qdrant vectors for a case by title."""
        return self.store.delete_by_case(case_title)

    def upsert_document(self, chunk: Dict[str, Any]) -> int:
        """Upsert a single chunk dict into Qdrant (proxy method)."""
        return self.store.add_documents([chunk])

    # ── Local search ──────────────────────────────────────────────────────

    def _search_local(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """BM25-style keyword scoring over the local JSON dataset."""
        keywords = query.lower().split()
        scored: List[Tuple[int, Dict[str, Any]]] = []

        for doc in self.dataset:
            fields = [
                doc.get("title"),
                doc.get("content"),
                doc.get("summary"),
                doc.get("section"),
                doc.get("category"),
                " ".join(doc.get("keywords", [])),
                " ".join(doc.get("elements_required", [])),
            ]
            search_text = " ".join(str(f) for f in fields if f).lower()
            score = sum(1 for kw in keywords if kw in search_text)
            if score:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    # ── Deduplication ─────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate docs by title+content fingerprint."""
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for doc in docs:
            key = (
                str(doc.get("title") or "")[:80]
                + str(doc.get("content") or "")[:80]
            )
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    # ── Enrichment ────────────────────────────────────────────────────────

    def _enrich_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(doc)
        act = doc.get("act")
        section = str(doc.get("section")) if doc.get("section") is not None else None

        if act and section:
            enriched["_act_name"] = act
            enriched["_section"] = self._normalize_section(section, act)
            return enriched

        text_to_parse = (
            f"{doc.get('title', '')} "
            f"{doc.get('content', '')} "
            f"{doc.get('summary', '')}"
        )
        act_name, raw_section = self._parse_provision_metadata(text_to_parse)
        enriched["_act_name"] = act_name
        enriched["_section"] = (
            self._normalize_section(raw_section, act_name) if raw_section else None
        )
        return enriched

    @staticmethod
    def _normalize_section(
        section: Optional[str], act: Optional[str]
    ) -> Optional[str]:
        if not section:
            return None
        if section.lower() == "unknown":
            return None

        clean = str(section).strip()
        is_const = act and ("constitution" in act.lower() or "art" in act.lower())
        prefix = "Article" if is_const else "Section"

        if clean.isdigit():
            return f"{prefix} {clean}"
        if not (clean.startswith("Section") or clean.startswith("Article")):
            return f"{prefix} {clean}"
        return clean

    @staticmethod
    def _parse_provision_metadata(
        text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Heuristic extraction of act name and section from free text."""
        # Pattern: "Section 302 of the Indian Penal Code, 1860"
        m = re.search(
            r"([Ss]ection|[Aa]rticle)\s+([\dA-Za-z]+(?:\([^)]*\))?)\s+of\s+"
            r"(?:the\s+)?([A-Z][A-Za-z\s,]+(?:\d{4})?)",
            text,
        )
        if m:
            return m.group(3).strip().rstrip(","), m.group(2).strip()

        # Pattern: "u/s 302 IPC"
        m = re.search(
            r"(?:u/s|under\s+[Ss]ection|Art\.)\s+([\dA-Za-z]+)\s+([A-Z]{2,})", text
        )
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # Bare section + known acronym
        m_sec = re.search(r"([Ss]ection|[Aa]rticle)\s+([\dA-Za-z]+)", text)
        m_act = re.search(r"\b(IPC|CrPC|CPC|IEA|NDPS|POCSO|IT Act|Arms Act)\b", text)
        if m_sec:
            return (m_act.group(1) if m_act else None), m_sec.group(2).strip()

        return None, None

    # ── Utility ───────────────────────────────────────────────────────────

    def get_all_content(self) -> str:
        """Return concatenated dataset text (used by HallucinationGuard)."""
        parts = [
            (doc.get("content") or "") + " " + (doc.get("summary") or "")
            for doc in self.dataset
        ]
        return " ".join(parts).lower()
