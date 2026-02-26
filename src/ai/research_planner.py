"""
research_planner.py
===================
LLM-driven query planner that decomposes a complex legal query into 2-3
focused sub-questions for targeted retrieval.

Replaces the old hardcoded keyword-matching approach so ANY legal topic
(not just bail/murder) is handled intelligently.
"""
from __future__ import annotations

import os
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ResearchPlanner:

    @staticmethod
    def plan(query: str, groq_client: Optional[object] = None) -> List[str]:
        """
        Break a legal query into 2-3 targeted sub-questions using the LLM.
        Falls back to a simple single-item list if the LLM is unavailable.

        Parameters
        ----------
        query        : The original user legal query.
        groq_client  : An initialised `groq.Groq` client. If None, a direct
                       os.environ lookup is attempted.
        """
        # Try LLM-driven planning first ----------------------------------------
        try:
            client = groq_client or ResearchPlanner._get_groq_client()
            if client:
                return ResearchPlanner._llm_plan(query, client)
        except Exception as e:
            logger.warning(f"[ResearchPlanner] LLM planning failed, using fallback: {e}")

        # Fallback: return original query as single task -----------------------
        return [query]

    @staticmethod
    def _get_groq_client() -> Optional[object]:
        api_key = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
        if not api_key:
            return None
        try:
            from groq import Groq
            return Groq(api_key=api_key)
        except Exception:
            return None

    @staticmethod
    def _llm_plan(query: str, client: object) -> List[str]:
        model = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
        prompt = (
            "You are an expert Indian legal research assistant.\n"
            "Break the following legal query into exactly 2–3 focused sub-questions "
            "that will guide targeted database and web searches.\n\n"
            "Rules:\n"
            "- Each sub-question must be self-contained and searchable.\n"
            "- Cover statutory provisions AND relevant case law angles.\n"
            "- If the query is about a recent event, include one sub-question "
            "  explicitly seeking 'latest' or 'recent' updates.\n"
            "- Return ONLY a JSON array of strings. No prose, no markdown.\n\n"
            f"Query: {query}\n\n"
            "Example output:\n"
            '["What are the statutory provisions under Section X of Act Y?", '
            '"What landmark Supreme Court cases have interpreted Section X?", '
            '"What are the recent amendments or updates to Act Y after 2022?"]'
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = completion.choices[0].message.content or "[]"
        # Strip markdown fences if the model wraps it
        raw = raw.strip()
        if raw.startswith("```"):
            import re
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            raw = m.group(1).strip() if m else raw
        sub_questions: List[str] = json.loads(raw)
        if not isinstance(sub_questions, list) or not sub_questions:
            return [query]
        logger.info(f"[ResearchPlanner] Decomposed into {len(sub_questions)} sub-questions.")
        return sub_questions[:3]  # Guard against over-verbose models
