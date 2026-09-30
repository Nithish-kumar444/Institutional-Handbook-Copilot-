"""
Automated Demo and Validation Suite for Institutional Handbook Copilot
Executes test cases specified in Section 16 of the specification:
1. "What is the minimum attendance requirement?"
2. "How many backlogs are allowed for promotion?"
3. "What is the cafeteria menu?" (Unsupported question)
"""

import os
import sys

# Ensure local imports work
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURR_DIR)

from generate_sample_docs import generate_sample_handbooks
from core import HandbookCopilot


def run_demo_test():
    print("=" * 70)
    print("INSTITUTIONAL HANDBOOK COPILOT — AUTOMATED DEMO TEST SUITE")
    print("=" * 70)

    # 1. Generate sample documents
    sample_dir = os.path.join(CURR_DIR, "sample_docs")
    pdf_2025, pdf_2026 = generate_sample_handbooks(sample_dir)
    print(f"\n[Step 1] Verified sample handbooks:")
    print(f"  • Older: {pdf_2025}")
    print(f"  • Newer: {pdf_2026}")

    # 2. Initialize Copilot and Index PDFs
    copilot = HandbookCopilot()
    print("\n[Step 2] Ingesting and indexing documents...")
    stats_2025 = copilot.load_pdf(pdf_2025)
    stats_2026 = copilot.load_pdf(pdf_2026)
    
    print(f"  • Ingested {stats_2025['filename']}: Detected Version='{stats_2025['detected_version']}', Chunks={stats_2025['chunk_count']}")
    print(f"  • Ingested {stats_2026['filename']}: Detected Version='{stats_2026['detected_version']}', Chunks={stats_2026['chunk_count']}")

    # 3. Test Cases
    test_cases = [
        {
            "id": "TEST_CASE_1",
            "query": "What is the minimum attendance requirement?",
            "expected_conflict": True,
            "expected_keywords": ["80%", "75%", "contradict"],
            "description": "Conflict detection on numerical percentage (Attendance 75% -> 80%)"
        },
        {
            "id": "TEST_CASE_2",
            "query": "How many backlogs are allowed for promotion?",
            "expected_conflict": True,
            "expected_keywords": ["2", "4", "backlog", "contradict"],
            "description": "Conflict detection on quantitative limit (Backlogs 4 -> 2)"
        },
        {
            "id": "TEST_CASE_3",
            "query": "What is the cafeteria menu?",
            "expected_conflict": False,
            "expected_keywords": ["could not find this information"],
            "description": "Grounded rejection of unsupported query"
        }
    ]

    all_passed = True

    for tc in test_cases:
        print("\n" + "-" * 70)
        print(f"RUNNING {tc['id']}: {tc['description']}")
        print(f"Query: \"{tc['query']}\"")
        print("-" * 70)

        res = copilot.ask(tc["query"], top_k=10)
        answer = res["answer"]
        has_conflict = res["has_conflict"]
        citations = res["citations"]

        print(f"\n[Application Output]:\n{answer}\n")
        print(f"[Conflict Detected]: {has_conflict}")
        print(f"[Citations Provided]:")
        for c in citations:
            print(f"  - {c.get('type')}: {c.get('document')} (v{c.get('version')}) Page {c.get('page')}")

        # Assertions
        passed = True
        if has_conflict != tc["expected_conflict"]:
            print(f"  ❌ Assertion Failed: Expected has_conflict={tc['expected_conflict']}, got {has_conflict}")
            passed = False

        answer_lower = answer.lower()
        for kw in tc["expected_keywords"]:
            if kw.lower() not in answer_lower:
                print(f"  ❌ Assertion Failed: Expected keyword '{kw}' not found in answer.")
                passed = False

        if passed:
            print(f"\n✅ {tc['id']} PASSED")
        else:
            print(f"\n❌ {tc['id']} FAILED")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY! (3/3)")
    else:
        print("SOME TESTS FAILED - REVIEW ABOVE OUTPUT")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_demo_test()
    sys.exit(0 if success else 1)
