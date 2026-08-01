"""Persist chat history and results into the existing Postgres database.

This implementation uses raw SQL against the same DATABASE_URL used by
the backend. It writes into the Prisma-managed tables (`message`,
`citation`, `agentexecutionlog`) using simple INSERT/SELECT statements.

Note: this module purposely avoids running migrations; it assumes the
Prisma schema already created the needed tables.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv(dotenv_path=os.path.join("backend", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in environment (backend/.env)")


class MemoryService:
    def __init__(self) -> None:
        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True

    def get_history(self, chat_id: str) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, role, content, structuredResponse, confidenceScore, conflictsDetected, createdAt FROM message WHERE chatId = %s ORDER BY createdAt ASC;",
                (chat_id,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def add_message(self, chat_id: str, role: str, content: str) -> Dict[str, Any]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO message (chatId, role, content) VALUES (%s, %s, %s) RETURNING id, chatId, role, content, createdAt;",
                (chat_id, role, content),
            )
            return dict(cur.fetchone())

    def add_result(self, chat_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        # result is expected to be the structuredResponse dict
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # insert assistant message
            structured_json = json.dumps(result)
            confidence = result.get("confidence_score")
            conflicts = result.get("conflicts_detected", False)

            cur.execute(
                "INSERT INTO message (chatId, role, content, structuredResponse, confidenceScore, conflictsDetected) VALUES (%s, %s, %s, %s::jsonb, %s, %s) RETURNING id;",
                (
                    chat_id,
                    "ASSISTANT",
                    result.get("issue_summary") or result.get("conclusion") or "",
                    structured_json,
                    float(confidence) if confidence is not None else None,
                    bool(conflicts),
                ),
            )
            message_row = cur.fetchone()
            message_id = message_row["id"]

            # insert citations
            for citation in result.get("citations", []) or []:
                cur.execute(
                    "INSERT INTO citation (messageId, title, court, year, source, url) VALUES (%s, %s, %s, %s, %s, %s);",
                    (
                        message_id,
                        citation.get("title") or citation.get("citation_reference") or "Unknown",
                        citation.get("court"),
                        citation.get("year") if isinstance(citation.get("year"), int) else None,
                        citation.get("source") or citation.get("citation_reference") or "",
                        citation.get("url") or citation.get("source_url") or None,
                    ),
                )

            # insert agent execution logs if present
            for log in result.get("agentLogs", []) or []:
                cur.execute(
                    "INSERT INTO agentexecutionlog (chatId, agentName, executionTimeMs, status, confidenceScore, conflictsDetected) VALUES (%s, %s, %s, %s, %s, %s);",
                    (
                        chat_id,
                        log.get("agentName"),
                        int(log.get("executionTimeMs") or 0),
                        "FAILED" if log.get("status") == "FAILED" else "SUCCESS",
                        float(confidence) if confidence is not None else None,
                        bool(conflicts),
                    ),
                )

            return {"messageId": message_id}

    def get_results(self, chat_id: str) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT m.id as message_id, m.structuredResponse, m.confidenceScore, m.conflictsDetected, c.id as citation_id, c.title, c.court, c.year, c.source, c.url FROM message m LEFT JOIN citation c ON c.messageId = m.id WHERE m.chatId = %s ORDER BY m.createdAt ASC;",
                (chat_id,),
            )
            rows = cur.fetchall()
            # Group by message
            messages: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                mid = r["message_id"]
                if mid not in messages:
                    messages[mid] = {
                        "structuredResponse": r.get("structuredResponse"),
                        "confidenceScore": r.get("confidenceScore"),
                        "conflictsDetected": r.get("conflictsDetected"),
                        "citations": [],
                    }
                if r.get("citation_id"):
                    messages[mid]["citations"].append(
                        {
                            "title": r.get("title"),
                            "court": r.get("court"),
                            "year": r.get("year"),
                            "source": r.get("source"),
                            "url": r.get("url"),
                        }
                    )

            return list(messages.values())

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
