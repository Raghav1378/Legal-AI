from src.ai.retriever_service import RetrieverService
import json

def test_legal_search():
    print("🚀 Running Production Data Test...")
    retriever = RetrieverService()
    
    # 1. Broad Semantic Search
    query = "fundamental rights and constitutional validity"
    print(f"\n🔍 Searching for: '{query}'")
    results = retriever.retrieve(query, limit=3)
    
    if not results:
        print("❌ No results found. Did the ingestion finish?")
        return

    print(f"✅ Found {len(results)} relevant documents.\n")
    
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Title:   {doc.get('title', 'N/A')}")
        print(f"Act/Sec: {doc.get('act', 'N/A')} - {doc.get('section', 'N/A')}")
        print(f"Source:  {doc.get('summary', 'Local Dataset')}")
        # Print a snippet of content
        content = doc.get('content', '')
        print(f"Snippet: {content[:150]}...")
        print("-" * 20)

    # 2. Test Metadata Filtering (The new Qdrant feature)
    print("\n📅 Testing Filtered Search (Year: 2023)...")
    filtered_results = retriever.retrieve(query, filters={"year": 2023}, limit=2)
    
    if filtered_results:
        print(f"✅ Successfully found {len(filtered_results)} cases from 2023.")
        for doc in filtered_results:
             # Check if it's actually from Qdrant metadata
             meta = doc.get('metadata', {})
             print(f"Confirmed Year: {meta.get('year') or 'N/A'}")
    else:
        print("ℹ️ No 2023 cases found yet in this specific search.")

if __name__ == "__main__":
    test_legal_search()
