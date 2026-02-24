import os
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

class VectorService:
    def __init__(self):
        # Setup Google Embeddings (New google-genai SDK)
        raw_key = os.environ.get("GOOGLE_API_KEY")
        self.google_api_key = raw_key.strip("'\" ") if raw_key else None
        if self.google_api_key:
            self.genai_client = genai.Client(api_key=self.google_api_key)
            self.embed_model = "text-embedding-004"
        
        # Setup Pinecone
        self.pc_api_key = os.environ.get("PINECONE_API_KEY")
        self.index_name = "legal-ai-judgments"
        self.pc = Pinecone(api_key=self.pc_api_key) if self.pc_api_key else None
        self.index = None
        
    def _ensure_index_exists(self):
        if not self.pc:
            return
        
        # Check if index exists by listing all indexes
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"[VECTOR] Creating new index: {self.index_name}")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768, # Dimension for Google text-embedding-004
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(2)
            except Exception as e:
                print(f"[VECTOR] Index creation error (might already exist): {e}")
        
        self.index = self.pc.Index(self.index_name)

    def upsert_judgment_chunk(self, id: str, text: str, metadata: Dict[str, Any]):
        """Upsert a chunk of text with its metadata into Pinecone."""
        self._ensure_index_exists()
        if not self.index or not self.google_api_key:
            return

        # New google-genai embedding call
        try:
            response = self.genai_client.models.embed(
                model=self.embed_model,
                contents=[text]
            )
        except AttributeError:
            response = self.genai_client.models.embed_content(
                model=self.embed_model,
                contents=[text]
            )
        
        # The result structure might differ based on SDK version
        # For google-genai >= 0.1.0, it's response.embeddings[0].values
        vector = response.embeddings[0].values
        
        self.index.upsert(vectors=[{
            "id": id,
            "values": vector,
            "metadata": {**metadata, "text": text}
        }])

    def query_similar_judgments(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search Pinecone for the most similar recent Supreme Court judgments."""
        self._ensure_index_exists()
        if not self.index or not self.google_api_key:
            return []

        # New google-genai embedding call
        try:
            response = self.genai_client.models.embed(
                model=self.embed_model,
                contents=[query]
            )
        except AttributeError:
            response = self.genai_client.models.embed_content(
                model=self.embed_model,
                contents=[query]
            )
        query_vector = response.embeddings[0].values
        
        # Perform Vector Search
        response = self.index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True
        )
        
        findings = []
        for match in response.matches:
            meta = match.metadata or {}
            findings.append({
                "act": "Supreme Court Judgment",
                "section": str(meta.get("case_year", "N/A")),
                "title": meta.get("case_name", "Unknown Case"),
                "content": meta.get("text", ""),
                "summary": f"Retrieved from Supreme Court Archive (Relevance: {int(match.score * 100)}%)",
                "source_url": meta.get("source_url")
            })
            
        return findings
