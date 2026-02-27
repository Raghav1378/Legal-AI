from qdrant_client import QdrantClient
import os

local_path = os.path.abspath(os.path.join(os.getcwd(), "qdrant_local"))
client = QdrantClient(path=local_path)

try:
    collection_info = client.get_collection("legal_documents")
    print(f"Collection: legal_documents")
    print(f"Points count: {collection_info.points_count}")
except Exception as e:
    print(f"Error: {e}")
