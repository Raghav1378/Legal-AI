# 🏛️ Legal Research AI — Project Specification

## Overview

An agentic AI system for structured Indian legal research. Built for accuracy, reliability, and
source-backed reasoning — designed for advocates, researchers, and legal professionals.

---

## 4. Core Functional Modules

### 4.1 Legal Research Chat Engine

The primary interaction layer where users submit legal queries.

**Responsibilities:**
- Accept and interpret legal questions (IPC, CrPC, constitutional law, etc.)
- Maintain conversational context across multi-turn sessions
- Break complex queries into smaller, searchable research tasks
- Retrieve information from multiple source documents
- Validate and cross-check retrieved data against known corpus
- Generate structured, citation-backed responses

**Required Capabilities:**

| Capability | Implementation |
|---|---|
| Context-aware conversations | `MemoryService` per session |
| Follow-up question handling | Chat history injected into LLM context |
| Structured output generation | `LegalResponseSchema` (Pydantic) |
| Legal terminology understanding | Groq LLM + legal dataset retrieval |
| Response traceability | `agentLogs` with per-agent timing and status |

---

### 4.2 Multi-Agent Architecture

The system implements a multi-agent workflow where each agent performs a single specialized task.

```
┌─────────────────────────────────────────────────────┐
│                 MCPOrchestrator                      │
│                                                     │
│  QueryAnalysisAgent     → Intent & context parsing  │
│  ResearchPlanningAgent  → Query decomposition        │
│  RetrievalAgent         → MCP tool-calling loop      │
│    ├── search_legal_database()                       │
│    └── detect_conflicts()                            │
│  CrossVerificationAgent → Source conflict detection  │
│  HallucinationGuard     → Claim-corpus validation    │
│  ResponseFormatterAgent → Structured JSON generation │
│  ConfidenceScorer       → Deterministic [30–95]      │
└─────────────────────────────────────────────────────┘
```

**Agent Contracts:**
- Each agent operates independently on a specific task
- Produces structured intermediate output passed to the next stage
- Logs its own `executionTimeMs` and `status` (SUCCESS / FAILED)
- Never raises unhandled exceptions — failures degrade gracefully

**Resilience Rules:**
- Unknown tool calls → skipped, loop continues
- Duplicate tool calls → blocked by signature hash
- Empty retrieval → warning logged, confidence −20
- Malformed LLM JSON → schema repair, then safe fallback

---

### 4.3 Multi-Tool Research Layer

The system supports multiple research tools registered in `ToolRegistry`.

**Currently Registered Tools:**

| Tool | Purpose |
|---|---|
| `search_legal_database` | Keyword search across `dataset.json` + Semantic search across `chroma_db` |
| `detect_conflicts` | Scans docs for contradictory legal keywords |

**Retrieval Capabilities:**
- Searches Indian legal repositories (IPC, CrPC, CPA, NDPS, POCSO, Constitution)
- Extracts judgments, sections, and acts
- Identifies precedents and amendment references
- Ranks results by keyword relevance score
- Deduplicates retrieved documents
- Highlights source inconsistencies via `ConflictDetector`

**Adding a New Tool:**
```python
# In ToolRegistry._register_tools()
self.tools["my_new_tool"] = {
    "name": "my_new_tool",
    "description": "What this tool does.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    "execute": lambda args: MyNewService().search(args["query"])
}
```

---

## 7. ML Agentic Flow & Chat Intelligence

### Multi-Agent Workflow

```
User Query
    │
    ▼
QueryAnalysisAgent          → Understands intent & legal context
    │
    ▼
ResearchPlanningAgent       → Decomposes query into search strategies
    │
    ▼
RetrievalAgent              → Searches legal database (MCP Tool Loop)
    │
    ├──► search_legal_database("Section 438 CrPC")
    ├──► search_legal_database("anticipatory bail landmark cases")
    │
    ▼
CrossVerificationAgent      → detect_conflicts(retrieved_docs)
    │
    ▼
HallucinationGuard          → Validates claims against retrieved corpus
    │
    ▼
ResponseFormatterAgent      → Generates final structured JSON
    │
    ▼
ConfidenceScorer            → Deterministic scoring [30, 95] with penalties
```

### Reasoning Orchestration

The `MCPOrchestrator` drives a bounded tool-calling loop (max 5 turns) with:
- Duplicate call detection (signature hashing)
- Unknown tool rejection with safe fallback
- Automatic JSON repair and schema coercion

### Query Planning

`ResearchPlanner` decomposes queries into:
- Primary act/section keywords
- Relevant case type classification (bail, FIR, custody, etc.)
- Secondary search terms for cross-verification

### Confidence Scoring Rules

| Condition | Penalty |
|---|---|
| Conflict detected | −15 |
| JSON repair needed | −10 per repair |
| Hallucinated claim | −15 per claim |
| Metadata extraction fail | −5 per doc |
| Empty retrieval | −20 |
| Tool call corruption | −10 per incident |
| Multiple agreeing sources | +5 |

**Hard clamp:** `30 ≤ confidence_score ≤ 95`

### Conversational Memory

- Per-session message history via `MemoryService`
- Only verified conclusions (non-`"N/A"`) are written to memory
- Prevents fallback/repair responses from polluting future turns

---

## 8. Data Sources Integration

### Supported Source Types

| Source | Examples |
|---|---|
| Indian Penal Code | Sections 302, 376, 420 IPC |
| Code of Criminal Procedure | Sections 154, 161, 438 CrPC |
| Landmark Judgments | Supreme Court, High Courts |
| Consumer Protection | CPA 1986, 2019 |
| Special Legislation | NDPS, POCSO, Prevention of Corruption |
| Constitutional Articles | Articles 14, 19, 21 |

### Retrieval Pipeline

`RetrieverService` uses keyword-based relevance scoring across:
- `title`, `content`, `summary`, `section` fields
- All results enriched with structured metadata (`_act_name`, `_section`)

### Metadata Extraction (Three-tier Regex)

```
Priority 1: "Section 154 of the Code of Criminal Procedure, 1973"
Priority 2: "u/s 302 IPC" or "under Section 438 CrPC"
Priority 3: Bare "Section 438" + nearby acronym (IPC/CrPC/NDPS etc.)
Fallback:   _act_name = None, _section = None (never "Unknown")
```

### Expanding the Dataset

Add entries to `src/ai/dataset.json` in this structure:

```json
{
  "id": "unique_id",
  "type": "section | case | amendment",
  "act": "Indian Penal Code",
  "section": "302",
  "title": "Punishment for Murder",
  "content": "Full statutory text...",
  "summary": "Short description for retrieval scoring."
}
```

---

## 9. Structured Output Format

Every API response follows this strict contract:

```json
{
  "structuredResponse": {
    "issue_summary": "Legal issue analysis for: <query>",
    "relevant_legal_provisions": [
      { "act_name": "IPC", "section": "Section 302", "explanation": "..." }
    ],
    "applicable_sections": ["Section 302 IPC", "Section 304 IPC"],
    "case_references": [
      { "case_name": "State v. Accused", "court": "Supreme Court", "year": 2019, "citation_reference": "AIR 2019 SC 123" }
    ],
    "key_observations": [
      "The offence carries mandatory minimum sentence.",
      "Mitigating circumstances must be pleaded separately."
    ],
    "legal_interpretation": "Based on the retrieved provisions...",
    "precedents": ["Bachan Singh v. State of Punjab (1980)"],
    "conclusion": "The accused may be charged under...",
    "citations": [
      { "title": "...", "court": "...", "year": 2021, "source": "IPC", "url": null }
    ],
    "conflicts_detected": false,
    "confidence_score": 82
  },
  "agentLogs": [
    { "agentName": "RetrievalAgent", "executionTimeMs": 312, "status": "SUCCESS" },
    { "agentName": "ResponseFormatterAgent", "executionTimeMs": 1840, "status": "SUCCESS" }
  ],
  "totalExecutionTimeMs": 2152
}

### Contract Guarantees
- `structuredResponse`, `agentLogs`, and `totalExecutionTimeMs` are always present — even on error
- `confidence_score` is always in `[30, 95]`
- All list fields default to `[]` — never `null`
- `agentLogs[*].executionTimeMs` is always `>= 1`
- No extra top-level keys (snake_case fields stripped)

---

## Verification

```bash
# Run hardening test suite (8 tests)
$env:PYTHONPATH="src"
python -m ai.verify_orchestration
```

All 8 tests validate: metadata accuracy, hallucination removal, schema contract,
timing correctness, confidence clamping, empty retrieval grace, JSON repair, and multi-turn memory.
