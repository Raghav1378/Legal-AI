"""
Hardening Test Suite for Legal Research AI Engine.
Tests: metadata, hallucination guard, schema contract, agent logs, confidence clamping.
Run from project root with: $env:PYTHONPATH="src"; python -m ai.verify_orchestration
"""
import sys
import os
import json

# Allow running as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .ai_service import AIService
from .confidence_scorer import ConfidenceScorer
from .hallucination_guard import HallucinationGuard
from .retriever_service import RetrieverService

# ─────────────────────── helpers ───────────────────────

def _check(label: str, condition: bool):
    status = "✅" if condition else "❌"
    print(f"  {status} {label}")
    return condition

def _section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ─────────────────────── tests ─────────────────────────

def test_metadata_extraction():
    _section("Test 1: Metadata Extraction (no 'Unknown')")
    retriever = RetrieverService()
    docs = retriever.retrieve("Section 438 CrPC anticipatory bail")

    passed = True
    for doc in docs:
        act_name = doc.get('_act_name')
        section  = doc.get('_section')
        # Must not be the string "Unknown"
        ok = (act_name != "Unknown") and (section != "Unknown")
        _check(f"  Doc '{doc.get('id','')}': act_name={act_name!r}, section={section!r}", ok)
        passed = passed and ok

    _check("No 'Unknown' string returned for any doc", passed)


def test_precedent_purity():
    _section("Test 2.1: Precedent Purity (Pattern match)")
    docs = [{"content": "Gurbaksh Singh Sibbia v. State of Punjab.", "title": "Sibbia Case"}]
    guard = HallucinationGuard(docs)
    
    response = {
        "precedents": [
            "Gurbaksh Singh Sibbia v. State of Punjab",  # Valid
            "Bail is a right",                            # Invalid
            "State v. Unknown",                          # Valid
            "Some text without v dot"                    # Invalid
        ]
    }
    cleaned, h_count, p_count, warnings = guard.validate(response)
    
    _check("Valid case names retained", len(cleaned['precedents']) == 2)
    _check("Invalid names removed", p_count == 2)
    _check("Removed markers in warnings", any("Invalid precedent" in w for w in warnings))


def test_hallucination_guard_strict():
    _section("Test 2.2: Strict Hallucination Guard")
    docs = [{"content": "Anticipatory bail protects individual liberty.", "summary": ""}]
    guard = HallucinationGuard(docs)

    response = {
        "key_observations": [
            "Anticipatory bail protects individual liberty.",        # supported
            "FIR must be filed within 20 days of the incident.",    # not supported
        ],
        "legal_interpretation": "Criminal law is complex.",         # not supported (abstract)
    }
    cleaned, h_count, p_count, warnings = guard.validate(response)

    _check("Supported observation retained", len(cleaned['key_observations']) == 1)
    _check("Unsupported observation removed", h_count >= 2)
    _check("Abstract interpretation neutralized", "could not be independently verified" in cleaned['legal_interpretation'])


def test_schema_contract():
    _section("Test 3: Schema Contract & Log Uniqueness")
    service = AIService()
    result = service.process_legal_query("schema_test", "What is Section 438 CrPC?")

    top_keys = set(result.keys())
    allowed  = {"structuredResponse", "agentLogs", "totalExecutionTimeMs"}
    _check("Top-level keys are correct", top_keys == allowed)

    logs = result.get("agentLogs", [])
    agent_names = [log.get("agentName") for log in logs if "[WARNING]" not in (log.get("error_message") or "")]
    unique_names = sorted(list(set(agent_names)))
    
    expected_stages = sorted([
        "QueryAnalysisAgent", "ResearchPlanningAgent", "RetrievalAgent",
        "CrossVerificationAgent", "HallucinationGuardAgent", "ResponseFormatterAgent"
    ])
    
    _check(f"Exactly 6 stages logged: {unique_names}", len(unique_names) == 6)
    _check("Stage names match requirements", unique_names == expected_stages)
    _check("Each stage logs exactly once", len(agent_names) == 6)


def test_confidence_recalibration():
    _section("Test 5: Confidence Recalibration & Bonuses")

    # Lower baseline with penalties
    score1 = ConfidenceScorer.calculate(
        num_agreeing_sources=1,
        conflicts_detected=False,
        has_case_citation=False
    )
    
    # Higher score with case citation bonus
    score2 = ConfidenceScorer.calculate(
        num_agreeing_sources=1,
        conflicts_detected=False,
        has_case_citation=True
    )
    
    _check(f"Case citation adds bonus: {score2} > {score1}", score2 > score1)
    _check(f"Bonus amount is +5: {score2 - score1 == 5}", score2 - score1 == 5)

    # Precedent contamination penalty
    score3 = ConfidenceScorer.calculate(
        num_agreeing_sources=1,
        conflicts_detected=False,
        precedent_contaminations=1
    )
    _check(f"Precedent contamination applies penalty: {score3} < {score1}", score3 < score1)


def test_empty_retrieval_graceful():
    _section("Test 6: Empty Retrieval Graceful Handling")
    service = AIService()
    # An extremely specific query unlikely to match any dataset doc
    result = service.process_legal_query(
        "empty_test", "xyzzy gibberish legal provision nonexistent"
    )
    sr = result.get("structuredResponse", {})
    _check("structuredResponse returned (not None)", sr is not None)
    _check("conclusion field present", "conclusion" in sr)
    _check("confidence_score >= 30", sr.get("confidence_score", 0) >= 30)


def test_landmark_case_depth():
    _section("Test 9: Landmark Case depth enforcement")
    service = AIService()
    # Query for Sibbia which is in dataset.json
    result = service.process_legal_query("landmark_test", "Explain Gurbaksh Singh Sibbia v. State of Punjab")
    
    sr = result.get("structuredResponse", {})
    obs = " ".join(sr.get("key_observations", [])).lower()
    
    # Check for core elements in observations
    found_principle = any(word in obs for word in ["principle", "holding", "decision", "facts", "impact"])
    _check("Key observations contain depth (principle/holding/facts)", found_principle)
    _check("Confidence for landmark query >= 75", sr.get("confidence_score", 0) >= 75)


def verify_hardened_ai():
    print("\n" + "═"*55)
    print("  LEGAL AI ENGINE — GROUNDING & LOGGING VERIFICATION")
    print("═"*55)

    test_metadata_extraction()
    test_precedent_purity()
    test_hallucination_guard_strict()
    test_schema_contract()
    test_confidence_recalibration()
    test_empty_retrieval_graceful()
    test_landmark_case_depth()

    print("\n" + "═"*55)
    print("  VERIFICATION COMPLETE")
    print("═"*55 + "\n")


if __name__ == "__main__":
    verify_hardened_ai()
