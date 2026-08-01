"""pgvector-backed LegalVectorStore

Uses the same DATABASE_URL as the backend (loads backend/.env) and stores
embeddings in a `legal_chunks` table with a `vector(384)` column.
"""
from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import psycopg2

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LegalVectorStore:
    DIM = 384
    MODEL = "all-MiniLM-L6-v2"

    def __init__(self, database_url: Optional[str] = None) -> None:
        # Prefer explicit DATABASE_URL, but fall back to backend/.env then src/ai/.env
        load_dotenv(dotenv_path=os.path.join("backend", ".env"), override=False)
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL not found in environment (backend/.env or src/ai/.env)")

        # DB connection
        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = True
        self._ensure_schema()

        # Embedding model
        self.model = SentenceTransformer(self.MODEL)

    def _ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            # enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # create table
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS legal_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE,
                    text TEXT,
                    case_title TEXT,
                    act TEXT,
                    section TEXT,
                    year INT,
                    embedding vector({self.DIM}),
                    created_at TIMESTAMP DEFAULT now()
                );
                """
            )

    def _vec_to_sql(self, vec: List[float]) -> str:
        return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Upsert a list of documents into the pgvector table.

        documents: list of dicts with keys: chunk_id, text, case_title, act, section, year
        Returns number of upserted rows.
        """
        texts = [d["text"] for d in documents]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        count = 0
        with self.conn.cursor() as cur:
            for doc, emb in zip(documents, embeddings):
                emb_sql = self._vec_to_sql(list(map(float, emb)))
                cur.execute(
                    """
                    INSERT INTO legal_chunks (chunk_id, text, case_title, act, section, year, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE
                      SET text = EXCLUDED.text,
                          case_title = EXCLUDED.case_title,
                          act = EXCLUDED.act,
                          section = EXCLUDED.section,
                          year = EXCLUDED.year,
                          embedding = EXCLUDED.embedding;
                    """,
                    (
                        doc.get("chunk_id"),
                        doc.get("text"),
                        doc.get("case_title"),
                        doc.get("act"),
                        doc.get("section"),
                        doc.get("year"),
                        emb_sql,
                    ),
                )
                count += 1
        return count

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        q_sql = self._vec_to_sql(list(map(float, q_emb)))

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text, case_title, act, section, year,
                       (embedding <-> %s::vector) AS distance
                FROM legal_chunks
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (q_sql, top_k),
            )
            rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "chunk_id": r[0],
                "text": r[1],
                "case_title": r[2],
                "act": r[3],
                "section": r[4],
                "year": r[5],
                "distance": float(r[6]) if r[6] is not None else None,
            })
        return results

    def filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        q_sql = self._vec_to_sql(list(map(float, q_emb)))

        where_clauses = []
        params = [q_sql]
        if filters:
            for k, v in filters.items():
                where_clauses.append(f"{k} = %s")
                params.append(v)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(top_k)

        sql = f"""
        SELECT chunk_id, text, case_title, act, section, year,
               (embedding <-> %s::vector) AS distance
        FROM legal_chunks
        {where_sql}
        ORDER BY distance ASC
        LIMIT %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "chunk_id": r[0],
                "text": r[1],
                "case_title": r[2],
                "act": r[3],
                "section": r[4],
                "year": r[5],
                "distance": float(r[6]) if r[6] is not None else None,
            })
        return results

    def delete_by_case(self, case_title: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM legal_chunks WHERE case_title = %s;", (case_title,))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
