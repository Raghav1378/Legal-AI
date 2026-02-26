import requests
import json

def test_api():
    url = "http://127.0.0.1:8000/query"
    payload = {
        "chat_id": "test-user-123",
        "query": "What is the penalty for cheating under IPC?"
    }
    
    print(f"📡 Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"📥 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n🚨 VERIFYING JSON FORMAT 🚨")
            print(json.dumps(data, indent=2))
            
            # Check for required fields
            required_fields = [
                "response_id", "issue_summary", "relevant_legal_provisions", 
                "applicable_sections", "case_references", "key_observations", 
                "legal_interpretation", "precedents", "conclusion", 
                "citations", "confidence_score", "generated_at", "jurisdiction"
            ]
            
            missing = [f for f in required_fields if f not in data]
            if not missing:
                print("\n✅ ALL REQUIRED FIELDS PRESENT!")
            else:
                print(f"\n❌ MISSING FIELDS: {missing}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_api()
