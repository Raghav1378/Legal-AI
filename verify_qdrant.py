import os
import time
from src.ai.vector_service import LegalVectorStore

def verify_migration():
    print("[START] Starting Qdrant Migration Verification...")
    
    # 1. Initialize Store
    try:
        store = LegalVectorStore()
        print("[SUCCESS] Qdrant Connection")
    except Exception as e:
        print(f"[ERROR] Qdrant Connection: {e}")
        return

    # 2. Test Ingestion
    test_chunk = {
        "chunk_id": "test_chunk_001",
        "text": "The Supreme Court of India ruled on Article 21 of the Constitution.",
        "case_title": "Test vs Union of India",
        "court": "Supreme Court of India",
        "year": 2024,
        "act": "Constitution of India",
        "section": "Article 21"
    }
    
    upserted = store.add_documents([test_chunk])
    if upserted > 0:
        print(f"[SUCCESS] Document Ingestion ({upserted} chunks)")
    else:
        print("[ERROR] Document Ingestion")

    # 3. Test Filtered Search
    print("[SEARCH] Testing Filtered Search (Year=2024)...")
    results = store.filtered_search(
        query="Article 21",
        filters={"year": 2024}
    )
    
    if results and any(r['metadata']['year'] == 2024 for r in results):
        print(f"[SUCCESS] Filtered Search (Found {len(results)} matches)")
    else:
        print("[ERROR] Filtered Search")

    # 4. Cleanup
    store.delete_by_case("Test vs Union of India")
    print("[CLEANUP] Test data removed.")

if __name__ == "__main__":
    verify_migration()
