from pinecone import Pinecone
import os

class LegalVectorStore:
    def __init__(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        env = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.client = Pinecone(api_key=api_key)
        self.index = self.client.Index(os.environ.get("PINECONE_INDEX_NAME"))

    def add_documents(self, docs):
        vectors = []
        for doc in docs:
            # Embed text → 1536-dim vector (use Groq embeddings or OpenAI)
            embedding = self._embed(doc["text"])
            vectors.append((
                doc["chunk_id"],
                embedding,
                {"case_title": doc.get("case_title"), "act": doc.get("act"), "year": doc.get("year")}
            ))
        self.index.upsert(vectors=vectors)
        return len(vectors)

    def search(self, query, limit=5):
        embedding = self._embed(query)
        results = self.index.query(vector=embedding, top_k=limit, include_metadata=True)
        return [{"text": r["metadata"].get("text"), **r["metadata"]} for r in results["matches"]]

    def filtered_search(self, query, filters, limit=5):
        embedding = self._embed(query)
        filter_dict = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
        results = self.index.query(vector=embedding, top_k=limit, filter=filter_dict, include_metadata=True)
        return [{"text": r["metadata"].get("text"), **r["metadata"]} for r in results["matches"]]

    def _embed(self, text):
        # Use Groq's embedding OR free option like sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()