import requests
import json
import time

def test_api():
    url = "http://127.0.0.1:8000/query"
    payload = {
        "chat_id": "diagnostic-test",
        "query": "What are the essential elements of cheating under Section 415 of the IPC?"
    }
    
    print(f"📡 Sending request to {url}...")
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=180)
        elapsed = time.time() - start
        print(f"📥 Status Code: {response.status_code} (Took {elapsed:.2f}s)")
        
        with open("diag_output.json", "w", encoding="utf-8") as f:
            if response.status_code == 200:
                json.dump(response.json(), f, indent=2)
                print("✅ Full response saved to diag_output.json")
            else:
                f.write(response.text)
                print(f"❌ Error response saved to diag_output.json")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_api()
