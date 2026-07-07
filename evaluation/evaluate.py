#!/usr/bin/env python3
"""
RAG Evaluation Framework
========================
Computes retrieval and answer quality metrics for the RAG pipeline.

Metrics:
  - Precision@K  : fraction of top-K retrieved chunks that are relevant
  - Recall@K     : fraction of relevant chunks found in top-K results
  - MRR          : Mean Reciprocal Rank of the first relevant result
  - Groundedness : keyword overlap between generated answer and context
  - Relevance    : keyword overlap between generated answer and original query

Usage:
  python evaluate.py --source_table <collection_name> [--top_k 5] [--dataset test_dataset.json]

Output:
  evaluation_report.md
  evaluation_report.json
"""
import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

# Allow importing shared module from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.rag_core import retrieve_chunks, call_gemini, build_prompt


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "is", "a", "the", "in", "of", "to", "for", "and", "about",
    "what", "with", "this", "or", "an", "on", "at", "by", "from",
    "how", "why", "when", "where", "which", "who", "does", "do",
    "it", "its", "be", "are", "was", "were", "has", "have", "had"
}


def _tokenize(text: str) -> List[str]:
    """Lowercase + remove stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Fraction of top-K retrieved that are relevant."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids if r in set(relevant_ids))
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Fraction of relevant chunks found in top-K results."""
    if not relevant_ids:
        return 1.0   # no ground truth → trivially satisfied
    hits = sum(1 for r in retrieved_ids if r in set(relevant_ids))
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal of the rank of the first relevant result (0.0 if not found)."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Answer Quality (Heuristic)
# ---------------------------------------------------------------------------

def groundedness_score(answer: str, context: str) -> float:
    """Fraction of answer content words found in the context."""
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def relevance_score(answer: str, query: str) -> float:
    """Fraction of query keywords found in the answer."""
    query_tokens = set(_tokenize(query))
    answer_tokens = set(_tokenize(answer))
    if not query_tokens:
        return 0.0
    overlap = query_tokens & answer_tokens
    return round(len(overlap) / len(query_tokens), 4)


def answer_keyword_hit_rate(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of expected answer keywords present in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 4)


# ---------------------------------------------------------------------------
# Core Evaluation Loop
# ---------------------------------------------------------------------------

def evaluate_query(
    entry: Dict[str, Any],
    source_table: str,
    top_k: int,
    generate_answers: bool = True
) -> Dict[str, Any]:
    """Run evaluation for a single dataset entry."""
    query = entry["query"]
    expected_chunk_ids = entry.get("expected_chunk_ids", [])
    expected_keywords = entry.get("expected_answer_keywords", [])

    # 1. Retrieve
    try:
        chunks = retrieve_chunks(source_table, query, top_k)
    except Exception as e:
        return {
            "id": entry["id"],
            "query": query,
            "error": str(e),
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
        }

    retrieved_ids = [c["id"] for c in chunks]
    context = "\n\n---\n\n".join(c["content"] for c in chunks)

    # 2. Retrieval metrics (meaningful only if expected_chunk_ids is populated)
    p_at_k = precision_at_k(retrieved_ids, expected_chunk_ids)
    r_at_k = recall_at_k(retrieved_ids, expected_chunk_ids)
    rr      = reciprocal_rank(retrieved_ids, expected_chunk_ids)

    result = {
        "id": entry["id"],
        "query": query,
        "retrieved_count": len(chunks),
        "retrieved_ids": retrieved_ids,
        "has_ground_truth": bool(expected_chunk_ids),
        "precision_at_k": round(p_at_k, 4),
        "recall_at_k": round(r_at_k, 4),
        "reciprocal_rank": round(rr, 4),
    }

    # 3. Answer generation + quality metrics
    if generate_answers and context:
        prompt = build_prompt(context, query)
        answer = call_gemini(prompt)
        result["answer"] = answer
        result["groundedness"] = groundedness_score(answer, context)
        result["answer_relevance"] = relevance_score(answer, query)
        result["keyword_hit_rate"] = answer_keyword_hit_rate(answer, expected_keywords)
    else:
        result["answer"] = None
        result["groundedness"] = None
        result["answer_relevance"] = None
        result["keyword_hit_rate"] = answer_keyword_hit_rate("", expected_keywords)

    return result


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def compute_averages(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute macro-averages over all evaluated queries."""
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "mean_precision_at_k": avg("precision_at_k"),
        "mean_recall_at_k": avg("recall_at_k"),
        "mean_mrr": avg("reciprocal_rank"),
        "mean_groundedness": avg("groundedness"),
        "mean_answer_relevance": avg("answer_relevance"),
        "mean_keyword_hit_rate": avg("keyword_hit_rate"),
    }


def write_markdown_report(
    results: List[Dict],
    averages: Dict,
    source_table: str,
    top_k: int,
    output_path: str
) -> None:
    """Write a human-readable Markdown evaluation report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# RAG Evaluation Report",
        f"",
        f"**Generated:** {ts}  ",
        f"**Collection:** `{source_table}`  ",
        f"**Top-K:** {top_k}  ",
        f"**Queries evaluated:** {len(results)}",
        f"",
        f"---",
        f"",
        f"## Summary Metrics",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Mean Precision@{top_k} | {averages['mean_precision_at_k']} |",
        f"| Mean Recall@{top_k} | {averages['mean_recall_at_k']} |",
        f"| Mean MRR | {averages['mean_mrr']} |",
        f"| Mean Groundedness | {averages['mean_groundedness']} |",
        f"| Mean Answer Relevance | {averages['mean_answer_relevance']} |",
        f"| Mean Keyword Hit Rate | {averages['mean_keyword_hit_rate']} |",
        f"",
        f"> **Note on Precision/Recall/MRR:** These metrics are only meaningful when",
        f"> `expected_chunk_ids` in `test_dataset.json` are populated with actual",
        f"> chunk IDs from your ChromaDB collection.",
        f"",
        f"---",
        f"",
        f"## Per-Query Results",
        f""
    ]

    for r in results:
        lines.append(f"### Query {r['id']}: *{r['query']}*")
        lines.append(f"")
        if r.get("error"):
            lines.append(f"⚠️ **Error:** {r['error']}")
        else:
            lines.append(f"- **Precision@{top_k}:** {r['precision_at_k']}")
            lines.append(f"- **Recall@{top_k}:** {r['recall_at_k']}")
            lines.append(f"- **Reciprocal Rank:** {r['reciprocal_rank']}")
            if r.get("groundedness") is not None:
                lines.append(f"- **Groundedness:** {r['groundedness']}")
                lines.append(f"- **Answer Relevance:** {r['answer_relevance']}")
                lines.append(f"- **Keyword Hit Rate:** {r['keyword_hit_rate']}")
            if r.get("answer"):
                # Show first 200 chars of answer
                preview = r["answer"][:200].replace("\n", " ")
                lines.append(f"- **Answer Preview:** {preview}...")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Markdown report saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument("--source_table", required=True,
                        help="ChromaDB collection name (PostgreSQL table name, underscores ok)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of chunks to retrieve per query (default: 5)")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "test_dataset.json"),
                        help="Path to evaluation dataset JSON file")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip LLM answer generation (retrieval metrics only)")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to write evaluation reports (default: evaluation/)")
    args = parser.parse_args()

    # Load dataset
    print(f"📂 Loading dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"📋 {len(dataset)} queries loaded")

    generate_answers = not args.no_generate

    # Evaluate
    results = []
    for i, entry in enumerate(dataset, start=1):
        print(f"  [{i}/{len(dataset)}] Evaluating: {entry['query'][:60]}...")
        result = evaluate_query(entry, args.source_table, args.top_k, generate_answers)
        results.append(result)
        if result.get("error"):
            print(f"    ⚠️  Error: {result['error']}")
        else:
            print(f"    P@{args.top_k}={result['precision_at_k']:.2f}  "
                  f"R@{args.top_k}={result['recall_at_k']:.2f}  "
                  f"RR={result['reciprocal_rank']:.2f}"
                  + (f"  Grounded={result['groundedness']:.2f}" if result.get("groundedness") is not None else ""))

    # Averages
    averages = compute_averages(results)
    print(f"\n📊 Summary:")
    for k, v in averages.items():
        print(f"   {k}: {v}")

    # Write reports
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "evaluation_report.json")
    md_path   = os.path.join(args.output_dir, "evaluation_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"averages": averages, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON report saved: {json_path}")

    write_markdown_report(results, averages, args.source_table, args.top_k, md_path)
    print(f"\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
