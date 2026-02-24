# Legal Research AI: End-to-End System Explanation

This document provides a comprehensive walkthrough of how the **Legal Research AI** project works, from the user's query to the final, hardened response.

---

## 🏗 High-Level Architecture
The system is built as a **Multi-Agent Orchestration Engine** using a Retrieval-Augmented Generation (RAG) pattern. It is hosted as a FastAPI web server and leverages the Groq LLM API for high-speed inference.

### 🌟 Core Design Philosophies
1.  **Strict Grounding**: The AI must never "hallucinate" legal sections or cases not found in the verified `dataset.json`.
2.  **Schema Enforcement**: Responses must follow a precise JSON contract to be used by frontend applications.
3.  **Deterministic Confidence**: Confidence scores are calculated mathematically based on evidence, not guessed by the LLM.
4.  **Full Transparency**: Every step of the research process is logged and returned to the user.

---

## 🔄 The End-to-End Flow (The 6 Stages)

When a query is received (e.g., *"What is Section 438 of CrPC?"*), the `MCPOrchestrator` starts a linear sequence of agents:

### 1. Query Analysis Agent
- **Goal**: Understand the user's intent and legal context.
- **Process**: Identifies if the query is about a specific statute (IPC, CrPC), a landmark case, or a general legal topic.

### 2. Research Planning Agent
- **Goal**: Break the query into actionable sub-questions.
- **Process**: Uses the `ResearchPlanner` to identify key areas of investigation (e.g., "What is the punishment?" "What are the exceptions?").

### 3. Retrieval Agent (The MCP Loop)
- **Goal**: Fetch verified legal data from the database.
- **Process**: 
    - Executes a **Model Context Protocol (MCP)** loop. 
    - Calls the `search_legal_database` tool.
    - The `RetrieverService` searches `dataset.json` for keywords and extracts structured metadata (Act name, Section number).
    - It can run multiple loops to refine the search results.

### 4. Cross-Verification Agent
- **Goal**: Identify contradictions in the retrieved data.
- **Process**: Uses the `ConflictDetector` to look for keywords like "overruled" or "distinguished" that might indicate a change in law.

### 5. Response Formatter Agent
- **Goal**: Synthesize the research into a structured JSON response.
- **Process**: 
    - Prompts the LLM to format the collected findings into the `LegalResponse` schema.
    - If the LLM generates "dirty" or broken JSON, the `_validate_and_repair_schema` logic automatically "heals" it by filling missing keys and coercing types.
    - **Landmark Case Discipline**: If a landmark case is detected, it strictly enforces the inclusion of **Facts, Issue, Holding, Principle, and Impact**.

### 6. Hallucination Guard Agent
- **Goal**: The final "Police Officer" for truth.
- **Process**:
    - Compares every claim in the response against the text in the retrieved documents.
    - **Regex Check**: Ensures `precedents` are in correct `"Party v. Party"` format.
    - **Removal**: Deletes unsupported observations.
    - **Neutralization**: If a legal interpretation isn't grounded, it replaces it with a warning message.

---

## 📊 Confidence Scoring Model
The system doesn't "feel" confident; it **calculates** confidence using the following logic:

| Factor | Change to Score |
| :--- | :--- |
| **Base Score** | 85 |
| **Multiple Sources Found** | +5 |
| **Case Citation Present** | +5 |
| **Statute Citation Present** | +5 |
| **Contradictions Found** | -15 |
| **LLM "Hallucination" Removed** | -15 (per item) |
| **Precedent Format Contamination** | -10 (per item) |
| **Empty Database Retrieval** | -20 |
| **Schema Repair Required** | -10 (per item) |

---

## 💾 Session Management (History Storage Keeper)
The `MemoryService` acts as a central repository for:
1.  **Context Management**: Records the conversation history so you can ask follow-up questions.
2.  **Result Persistence**: Stores the **full JSON output** of every research task.
3.  **Instant Recall**: The `GET /history/{chat_id}` endpoint allows users to retrieve past answers instantly without re-paying for LLM tokens or waiting for the agent loop.

---

## 🛡 Security & Reliability
- **Git Protection**: A strict `.gitignore` ensures that API keys and environment secrets are never committed to version control.
- **Retry Mechanism**: All LLM calls and tool executions are wrapped in a `with_retry` decorator to handle temporary network issues or API rate limits.
- **API Isolation**: The logic is decoupled from the web layer, allowing it to be tested in isolation via `verify_orchestration.py`.

---

## 📖 Key Components List
- `main.py`: The FastAPI entry point.
- `ai_service.py`: The high-level coordinator of memory, retrieval, and orchestration.
- `mcp_orchestrator.py`: The brain that manages the agent sequence and JSON repair.
- `hallucination_guard.py`: The truth-verification logic.
- `dataset.json`: The "Source of Truth" for the RAG engine.
