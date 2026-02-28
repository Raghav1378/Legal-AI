# LegalResearchAI: End-to-End Workflow & Architecture

Hello, my name is **Raghav Ramani**. I am the Lead Architect and Developer of LegalResearchAI. 

I built this system to address a foundational problem in legal technology: AI hallucinations. Standard Large Language Models (LLMs) are incredibly capable, but in the legal domain, their tendency to confidently invent case law, misinterpret statutes, or hallucinate citations is a serious risk. Legal professionals cannot rely on probabilistic text generation; they need deterministic, deeply researched, and cross-verified intelligence.

LegalResearchAI is designed to mitigate these risks. I implemented a strict Model Context Protocol (MCP) based multi-agent orchestrator that forces the AI away from open-ended chat and into a rigid, structured research pipeline. 

Here is a glimpse of what the engine does when queried:

**Example Query:** *"What are the essential elements of cheating under Section 415 of the IPC?"*

**Abridged Engine Output:**
```json
{
  "issue_summary": "The query asks for the essential elements of the offense of 'cheating' as defined under Section 415 of the Indian Penal Code.",
  "applicable_sections": [
    {
      "section_number": "415",
      "section_title": "Cheating",
      "section_summary": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property... is said to 'cheat'."
    }
  ],
  "case_references": [
    {
      "case_title": "Shri Bhagwan Samardha Sreepada Vallabha Venkata Vishwanandha Maharaj v. State of A.P.",
      "court": "Supreme Court of India",
      "year": 1999,
      "holding_summary": "To hold a person guilty of cheating, the intention of the person must be dishonest or fraudulent at the time of making the promise."
    }
  ],
  "confidence_score": 0.92
}
```

Below, I outline exactly how the system achieves this, the architecture behind it, and how to run it yourself.

---

## 🏗️ Core Architecture & Components

I designed LegalResearchAI around several distinct, purpose-built components. For each piece of the architecture, here is an explanation of what it is, why I built it, and the specific problem it solves.

### 1. The MCP Orchestrator
- **What it is:** The central nervous system of the engine. It executes a strict 6-stage linear workflow: Query Analysis ➡️ Research Planning ➡️ Retrieval ➡️ Cross-Verification ➡️ Hallucination Guard ➡️ Response Formatter.
- **Why I built it:** General LLMs try to generate the final answer immediately based on their internal weights. I needed the system to show its work, gather evidence first, and then synthesize it. 
- **What problem it solves:** It solves the "chain of thought" drift problem. By forcing the model through rigid, independent stages, the Orchestrator ensures the system properly plans its research and retrieves concrete context before generating a highly structured response.

### 2. Tool Registry and Retrievers
- **What it is:** A suite of specialized tools (`search_legal_database`, `filtered_search_legal_database`, and `live_web_search`) capable of querying our offline Qdrant vector database (IPC, CrPC, IT Act, landmark judgments) and the live internet via Tavily.
- **Why I built it:** A static context window is insufficient for comprehensive legal research. I wanted the engine to actively "hunt" for information across both local, verified datasets and authoritative web domains (e.g., `sci.gov.in`, `indiankanoon.org`).
- **What problem it solves:** It overcomes the knowledge cutoff and context constraints of the base LLM, providing a rich, triangulated factual grounding from multiple authoritative legal sources.

### 3. Hallucination Guard
- **What it is:** A deterministic, post-generation validation layer. It intercepts the LLM’s final response and uses strict semantic matching protocols to verify that every citation, case name, and section mentioned actually exists in the provided retrieval context.
- **Why I built it:** Even with excellent Retrieval-Augmented Generation (RAG), LLMs can sometimes inject external, unverified knowledge or slightly modify case names. I needed a hard stop for these injections.
- **What problem it solves:** It aggressively cleans up and removes unverified claims before the user sees them, drastically minimizing the risk of a lawyer relying on a hallucinated precedent.

### 4. Deterministic Confidence Scorer
- **What it is:** An algorithmic scoring module that calculates a statistical confidence score (typically between 30 and 95) based on retrieval density, conflicting source detection, citation quality, and the number of hallucinations caught by the Guard.
- **Why I built it:** Subjective LLM-generated confidence scores (e.g., asking the AI "how confident are you?") evaluate to pure guesswork. I wanted a mathematical representation of the *evidence quality*.
- **What problem it solves:** It provides end-users with a transparent, actionable metric. If the score is low, the lawyer knows the retrieved data was sparse or contradictory and can adjust their research strategy accordingly.

### 5. Self-Healing JSON Engine
- **What it is:** An automated schema enforcer that intercepts malformed LLM outputs and reconstructs them into a strict, predefined JSON API contract, including a `REGENERATE` loop for severe structural failures (e.g., missing section objects).
- **Why I built it:** Upstream applications and front-ends cannot handle unpredictable markdown strings or broken JSON arrays.
- **What problem it solves:** It guarantees API stability. No matter what the underlying LLM attempts to output, the front-end will always receive a predictably structured and fully populated JSON object.

---

## 🔄 End-to-End Workflow Pipeline

Whenever a legal query is submitted to the engine, it undergoes the following step-by-step pipeline:

1. **User Input:** A query arrives into the system.
2. **Analysis & Agent Spawning:** The Orchestrator analyzes the intent and formulates a structured search strategy.
3. **Retrieval Tool Execution:**
   - The system queries the local Qdrant database (`search_legal_database`).
   - If offline results are insufficient or the query requires recent context, it automatically triggers a fallback to the live web (`live_web_search`) targeting authoritative domains.
4. **Context Consolidation:** All retrieved documents, metadata, and web links are stitched into a comprehensive legal context block.
5. **Generation:** The LLM receives strict JSON schema instructions and drafts the legal response based *strictly* on the provided context block, ignoring its internal unverified knowledge.
6. **Validation & Correction:** 
   - The JSON is repaired if structurally invalid.
   - The **Hallucination Guard** strips out unverified or unsupported citations.
   - The **Confidence Scorer** evaluates the evidence density and assigns a final mathematical score.
7. **Final Output Delivery:** The user receives a deeply researched, verified, and structured JSON object.

---

## 🚀 Getting Started & How to Run

To test the orchestration pipeline on your local machine, follow these steps:

### 1. Environment Setup

Ensure you have Python 3.9+ installed. Clone the repository, set up your virtual environment, and install dependencies:

```bash
git clone https://github.com/Raghav1378/Legal-AI.git
cd LegalResearchAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows: 
venv\Scripts\activate
# On Mac/Linux: 
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Required API Keys

Create a `.env` file inside `src/ai/` (or set the variables in your terminal) with the necessary API keys:

```ini
# Required: For high-speed LLM inference
GROQ_API_KEY=your_groq_api_key_here

# Recommended: For live legal web search retrieval
TAVILY_API_KEY=your_tavily_api_key_here 
```

### 3. Run the Example Command

I have included a local testing script (`test_orchestrator_local.py`) that simulates a full query trace without needing to start the FastAPI server. Run it from the root directory to see the engine in action:

```bash
# Make sure the src directory is in your PYTHONPATH
# On Windows (PowerShell):
$env:PYTHONPATH="src"
# On Mac/Linux:
export PYTHONPATH="src"

# Run the orchestration trace
python test_orchestrator_local.py
```

Watch the terminal as the Orchestrator logs the execution times of the various agents and cleanly outputs the final structured JSON response.
