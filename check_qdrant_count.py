from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "src", "ai", ".env")
load_dotenv(dotenv_path=env_path)

client = QdrantClient(host="localhost", port=6333)
try:
    collection_info = client.get_collection("legal_documents")
    print(f"Collection: legal_documents")
    print(f"Status: {collection_info.status}")
    print(f"Points count: {collection_info.points_count}")
except Exception as e:
    print(f"Error checking Qdrant: {e}")
