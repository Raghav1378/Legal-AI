"""
kaggle_ingestor.py
==================
Downloads Indian Supreme Court judgments from Kaggle and ingests them
into the Qdrant-backed LegalVectorStore.

Dataset : fanaticauthorship/sc-judgments-india-1950-2024
Chunking: 1 000-char windows with 200-char overlap (standard RAG strategy)
"""

from __future__ import annotations

import os
import re
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass

import fitz  # PyMuPDF
from tqdm import tqdm

from .vector_service import LegalVectorStore

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Sliding-window chunker."""
    chunks: List[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _stable_chunk_id(case_name: str, year: int, offset: int) -> str:
    """Deterministic, collision-resistant chunk ID."""
    raw = f"{case_name}::{year}::{offset}"
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def _infer_act(text: str) -> str:
    """Heuristic act detection from judgment text."""
    acts = [
        ("NDPS Act", r"\bNDPS\b"),
        ("IPC", r"\bI\.?P\.?C\.?\b"),
        ("CrPC", r"\bCr\.?P\.?C\.?\b"),
        ("IT Act", r"\bIT Act\b|\bInformation Technology Act\b"),
        ("POCSO Act", r"\bPOCSO\b"),
        ("Arms Act", r"\bArms Act\b"),
        ("Constitution of India", r"\bConstitution\b|\bArticle \d+\b"),
    ]
    for act_name, pattern in acts:
        if re.search(pattern, text):
            return act_name
    return ""


def _infer_section(text: str) -> str:
    """Extract a prominent section reference from judgment text."""
    m = re.search(r"[Ss]ection\s+(\d+[A-Za-z]?)", text)
    return m.group(1) if m else ""


def _infer_judge(text: str) -> str:
    """Very rough judge-name extraction from first 500 chars (bench header)."""
    m = re.search(r"BEFORE\s*:?\s*([A-Z][A-Z\s\.,]+(?:J\.|JJ\.))", text[:500])
    return m.group(1).strip() if m else ""


# ── Main ingestor ─────────────────────────────────────────────────────────────


class KaggleLegalIngestor:
    """
    Downloads Supreme Court judgment PDFs from Kaggle and ingests them
    into the Qdrant LegalVectorStore with full structured metadata.
    """

    DATASET_SLUG = "fanaticauthorship/sc-judgments-india-1950-2024"

    def __init__(self) -> None:
        # Explicitly move .env variables into KAGGLE_ environment for the client
        user = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if user: os.environ["KAGGLE_USERNAME"] = user.strip()
        if key:  os.environ["KAGGLE_KEY"] = key.strip()

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("[KAGGLE] Authenticated as %s", os.environ.get("KAGGLE_USERNAME"))
        except Exception as exc:
            logger.error("[KAGGLE] Authentication failed: %s", exc)
            print("\n[ERROR] Kaggle Authentication Failed.")
            print(f"DEBUG: Current KAGGLE_USERNAME='{os.environ.get('KAGGLE_USERNAME')}'")
            print("Check your .env file at src/ai/.env\n")
            raise

        self.store = LegalVectorStore()  # uses Qdrant under the hood

    # ── Public ────────────────────────────────────────────────────────────

    def ingest_year(self, year: int, max_files: int = 10) -> None:
        """Download and ingest up to *max_files* judgments for *year*."""
        download_path = os.path.abspath(f"./temp_judgments/{year}")
        os.makedirs(download_path, exist_ok=True)

        print(f"[KAGGLE] Fetching judgment list for {year} …")
        try:
            all_files = self.api.dataset_list_files(self.DATASET_SLUG).files
            print(f"[KAGGLE] Total files in dataset: {len(all_files)}")
            
            # The folder structure in this dataset might be different. 
            # Let's try multiple common patterns.
            patterns = [
                f"supreme_court_judgments/{year}/",
                f"judgments/{year}/",
                f"{year}/"
            ]
            
            year_files = []
            for pattern in patterns:
                year_files = [f for f in all_files if pattern in f.name]
                if year_files:
                    print(f"[KAGGLE] Found {len(year_files)} files using pattern: '{pattern}'")
                    break
            
            if not year_files:
                print(f"[KAGGLE] No files found for year {year} using any known pattern.")
                print(f"[DEBUG] First 5 files in dataset: {[f.name for f in all_files[:5]]}")
                return

            year_files = year_files[:max_files]
        except Exception as e:
            print(f"[ERR] Failed to list files: {e}")
            return

        for kfile in tqdm(year_files, desc=f"Ingesting {year}"):
            base_name = os.path.basename(kfile.name)
            local_pdf  = os.path.join(download_path, base_name)

            # Download (Kaggle may wrap in zip; handle both)
            self.api.dataset_download_file(
                self.DATASET_SLUG, kfile.name, path=download_path
            )
            # Try plain PDF path; fall back to zip variant
            if not os.path.exists(local_pdf):
                local_pdf = local_pdf + ".zip"

            text = self._extract_text(local_pdf)
            if not text:
                continue

            case_name = base_name.replace("_", " ").replace(".PDF", "").replace(".pdf", "")
            self._chunk_and_ingest(text, case_name=case_name, year=year)

    def ingest_local_pdf(
        self,
        pdf_path: str,
        case_title: str,
        year: int,
        court: str = "Supreme Court of India",
        jurisdiction: str = "India",
    ) -> int:
        """
        Ingest a single local PDF file.  Useful for testing without Kaggle.
        Returns number of chunks upserted.
        """
        text = self._extract_text(pdf_path)
        if not text:
            return 0
        return self._chunk_and_ingest(
            text,
            case_name=case_title,
            year=year,
            court=court,
            jurisdiction=jurisdiction,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _extract_text(self, path: str) -> str:
        try:
            doc = fitz.open(path)
            return "\n".join(page.get_text() for page in doc)
        except Exception as exc:
            print(f"[ERR] Text extraction failed for {path}: {exc}")
            return ""

    def _chunk_and_ingest(
        self,
        text: str,
        case_name: str,
        year: int,
        court: str = "Supreme Court of India",
        jurisdiction: str = "India",
    ) -> int:
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        
        # Infer act / section / judge from full text (once per document)
        act     = _infer_act(text)
        section = _infer_section(text)
        judge   = _infer_judge(text)

        documents: List[Dict[str, Any]] = []
        for offset, chunk_text in enumerate(chunks):
            documents.append({
                "chunk_id":    _stable_chunk_id(case_name, year, offset),
                "text":        chunk_text,
                "case_title":  case_name,
                "court":       court,
                "year":        year,
                "act":         act,
                "section":     section,
                "jurisdiction": jurisdiction,
                "judge":       judge,
                "source_url":  None,
                "keyword_tags": [act, court, str(year)],
            })

        count = self.store.add_documents(documents)
        print(f"[INGEST] '{case_name}' ({year}) → {count} chunks upserted to Qdrant.")
        return count


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        ingestor = KaggleLegalIngestor()
        ingestor.ingest_year(2023, max_files=5)
    except Exception:
        # Error message is already printed in __init__
        pass
