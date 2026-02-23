import sys
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add the 'src' directory to sys.path to allow imports of the 'ai' package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ai.ai_service import AIService
from api_models import QueryRequest, QueryResponse

app = FastAPI(
    title="Legal AI Research API",
    description="A hardened AI engine for legal research with MCP orchestration.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Service
ai_service = AIService()

@app.get("/")
def read_root():
    return {"status": "online", "engine": "Hardened MCP Orchestrator"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        start_time = time.time()
        result = ai_service.process_legal_query(request.chat_id, request.query)
        execution_time = time.time() - start_time
        
        return {
            "structuredResponse": result["structuredResponse"],
            "agentLogs": [
                {
                    "agentName": log["agentName"],
                    "executionTimeMs": log["executionTimeMs"],
                    "status": log["status"]
                } for log in result["agentLogs"]
            ],
            "totalExecutionTimeMs": result["totalExecutionTimeMs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{chat_id}")
async def clear_history(chat_id: str):
    ai_service.clear_chat_history(chat_id)
    return {"status": "success", "message": f"Cleared history for {chat_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
