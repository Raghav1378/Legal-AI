"""
tool_registry.py
================
Registry of LLM-callable tools for the MCPOrchestrator.

Tools
-----
search_legal_database          — hybrid (keyword + vector) retrieval from Qdrant
filtered_search_legal_database — hybrid retrieval with metadata pre-filters
detect_conflicts               — conflict/contradiction detector
live_web_search                — live internet search via Tavily (legal domains only)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from .retriever_service import RetrieverService
from .conflict_detector import ConflictDetector

logger = logging.getLogger(__name__)

# ── Authoritative Indian Legal Domains ────────────────────────────────────────
LEGAL_DOMAINS = [
    "sci.gov.in",
    "main.sci.gov.in",
    "indiankanoon.org",
    "indiacode.nic.in",
    "livelaw.in",
    "barandbench.com",
    "prsindia.org",
    "legislative.gov.in",
    "mhc.tn.gov.in",
]


def _normalize_tavily_results(raw_results: Any) -> List[Dict[str, Any]]:
    """
    Convert Tavily search results into a uniform format compatible with
    the vector DB retrieval output so downstream consumers don't need
    to care about the source.
    """
    if not raw_results:
        return []
    # Tavily returns a dict with a 'results' key, each item has url/title/content/score
    items = raw_results if isinstance(raw_results, list) else raw_results.get("results", [])
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "title":   item.get("title", "Web Result"),
            "content": item.get("content", ""),
            "section": None,
            "act":     None,
            "court":   None,
            "year":    None,
            "score":   item.get("score", 0.7),
            "source":  "web",
            "url":     item.get("url", ""),
        })
    return normalized


class ToolRegistry:
    def __init__(self, retriever: RetrieverService) -> None:
        self.retriever = retriever
        self.tools: Dict[str, Dict[str, Any]] = {}

        # Tavily client — initialised lazily so app still starts without the key
        self._tavily_client: Optional[Any] = None
        self._init_tavily()

        self._register_tools()

    # ── Tavily Setup ───────────────────────────────────────────────────────────

    def _init_tavily(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            logger.warning("[ToolRegistry] TAVILY_API_KEY not set — live_web_search disabled.")
            return
        try:
            from tavily import TavilyClient
            self._tavily_client = TavilyClient(api_key=api_key)
            logger.info("[ToolRegistry] Tavily client initialised successfully.")
        except Exception as e:
            logger.error(f"[ToolRegistry] Failed to init Tavily: {e}")

    # ── Registration ───────────────────────────────────────────────────────────

    def _register_tools(self) -> None:

        # ── 1. General hybrid search ──────────────────────────────────────────
        self.tools["search_legal_database"] = {
            "name": "search_legal_database",
            "description": (
                "Search for relevant sections and cases from IPC, CrPC, Arms Act, "
                "NDPS Act, POCSO Act, IT Act, and other Indian statutes stored in "
                "the offline legal vector database. Use this FIRST for any query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language legal question or keywords.",
                    },
                },
                "required": ["query"],
            },
            "execute": lambda args: self.retriever.retrieve(args["query"]),
        }

        # ── 2. Filtered semantic search ───────────────────────────────────────
        self.tools["filtered_search_legal_database"] = {
            "name": "filtered_search_legal_database",
            "description": (
                "Search the legal vector database with optional metadata filters. "
                "Useful when you want results from a specific court, act, or year range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language legal question.",
                    },
                    "court": {
                        "type": "string",
                        "description": "Filter by court name, e.g. 'Supreme Court of India'.",
                    },
                    "act": {
                        "type": "string",
                        "description": "Filter by act name, e.g. 'IPC', 'NDPS Act'.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Filter by section number, e.g. '302'.",
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Filter by jurisdiction, e.g. 'India'.",
                    },
                    "judge": {
                        "type": "string",
                        "description": "Filter by judge name.",
                    },
                    "year_gte": {
                        "type": "integer",
                        "description": "Include judgments from this year onwards.",
                    },
                    "year_lte": {
                        "type": "integer",
                        "description": "Include judgments up to (and including) this year.",
                    },
                },
                "required": ["query"],
            },
            "execute": lambda args: self.retriever.filtered_search(
                query=args["query"],
                filters={
                    k: args[k]
                    for k in ("court", "act", "section", "jurisdiction", "judge", "year_gte", "year_lte")
                    if args.get(k) is not None
                },
            ),
        }

        # ── 3. Conflict detector ──────────────────────────────────────────────
        self.tools["detect_conflicts"] = {
            "name": "detect_conflicts",
            "description": "Analyze retrieved legal documents for contradictions or conflicts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {"type": "array", "items": {"type": "object"}},
                },
            },
            "execute": lambda args: ConflictDetector.detect(args["documents"]),
        }

        # ── 4. Live web search (Tavily) ───────────────────────────────────────
        self.tools["live_web_search"] = {
            "name": "live_web_search",
            "description": (
                "Search the live internet for the latest Indian legal news, recent "
                "Supreme Court or High Court judgments, new amendments, PILs, and "
                "regulatory updates that may NOT be in the offline database. "
                "Use this when:\n"
                "  • search_legal_database returns empty or very few results.\n"
                "  • The query mentions events after 2022.\n"
                "  • The user specifically asks about 'latest', 'recent', or 'new' developments.\n"
                "Results come exclusively from authoritative Indian legal sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Focused legal search query for the internet.",
                    },
                },
                "required": ["query"],
            },
            "execute": self._execute_web_search,
        }

    # ── Tool Executors ─────────────────────────────────────────────────────────

    def _execute_web_search(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._tavily_client:
            logger.warning("[live_web_search] Tavily client unavailable — skipping.")
            return []
        query = args.get("query", "")
        if not query:
            return []
        try:
            logger.info(f"[live_web_search] Querying Tavily: {query!r}")
            raw = self._tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=6,
                include_domains=LEGAL_DOMAINS,
            )
            results = _normalize_tavily_results(raw)
            logger.info(f"[live_web_search] Got {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"[live_web_search] Tavily error: {e}")
            return []

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Return tool schemas (without execute callables) for LLM tool-use."""
        return [
            {k: v for k, v in tool.items() if k != "execute"}
            for tool in self.tools.values()
        ]

    def execute_tool(self, name: str, args: Any) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry.")
        return self.tools[name]["execute"](args)
