import sys
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ai.ai_service import AIService
from api_models import QueryRequest, QueryResponse, LegalResponseSchema

app = FastAPI(
    title="Legal AI Research API",
    description="AI-Powered Indian Legal Research Platform — Multi-Agent + RAG + Live Web Search.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_service = AIService()

@app.get("/")
def read_root():
    return {"status": "online", "engine": "Hardened MCP Orchestrator"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Full response including agent execution logs and timing.
    Returns the complete 9-section structured legal response + metadata.
    """
    try:
        result = ai_service.process_legal_query(request.chat_id, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/structured", response_model=LegalResponseSchema)
async def process_query_structured(request: QueryRequest):
    """
    Returns ONLY the 9-section structured legal response (no agent logs).
    Ideal for frontend display.
    Sections: issue_summary, relevant_legal_provisions, applicable_sections,
    case_references, key_observations, legal_interpretation, precedents,
    conclusion, citations.
    """
    try:
        result = ai_service.process_legal_query(request.chat_id, request.query)
        return result["structuredResponse"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{chat_id}")
async def clear_history(chat_id: str):
    ai_service.clear_chat_history(chat_id)
    return {"status": "success", "message": f"Cleared history for {chat_id}"}

@app.get("/history/{chat_id}")
async def get_history(chat_id: str):
    results = ai_service.get_chat_results(chat_id)
    if not results:
        return {"chat_id": chat_id, "history": [], "count": 0}
    return {
        "chat_id": chat_id,
        "history": results,
        "count": len(results)
    }

if __name__ == "__main__":
    import socket, signal, subprocess, platform

    def _free_port(port: int):
        """Kill any process occupying the given port (cross-platform)."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return  # port is already free
            # Port is in use — find and kill the owner
            if platform.system() == "Windows":
                result = subprocess.run(
                    f"netstat -ano | findstr :{port}", shell=True, capture_output=True, text=True
                )
                for line in result.stdout.splitlines():
                    parts = line.strip().split()
                    if parts and parts[-1].isdigit():
                        pid = int(parts[-1])
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
                        print(f"[SERVER] Killed PID {pid} occupying port {port}")
                        break
            else:
                subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)
        except Exception as e:
            print(f"[SERVER] Could not free port {port}: {e}")

    _free_port(8000)
    print("[SERVER] Starting Legal AI Research API on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
