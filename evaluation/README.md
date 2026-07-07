# Evaluation Framework

This folder contains the **RAG Evaluation Framework** — a standalone script that measures
retrieval quality and answer quality without requiring any running FastAPI server.

---

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Main evaluation script |
| `test_dataset.json` | 12 sample queries with expected keywords |
| `evaluation_report.md` | Generated Markdown report (after running) |
| `evaluation_report.json` | Generated JSON report (after running) |

---

## Metrics Computed

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K retrieved chunks that are relevant |
| **Recall@K** | Fraction of relevant chunks found within top-K |
| **MRR** | Mean Reciprocal Rank — where does the first relevant result appear? |

> **Note:** Precision, Recall, and MRR are only meaningful when you populate
> `expected_chunk_ids` in `test_dataset.json` with actual chunk IDs from your
> ChromaDB collection. By default those lists are empty, so P=0, R=1, MRR=0 is expected.

### Answer Quality Metrics (Heuristic)
| Metric | Description |
|--------|-------------|
| **Groundedness** | Fraction of answer words that appear in the retrieved context |
| **Answer Relevance** | Fraction of query keywords that appear in the answer |
| **Keyword Hit Rate** | Fraction of expected keywords found in the answer |

---

## Setup

Make sure you're in the project root with the virtual environment activated:

```bash
# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

## How to Run

### Basic run (retrieval + answer generation)
```bash
python evaluation/evaluate.py --source_table your_table_name
```

### Custom top-K
```bash
python evaluation/evaluate.py --source_table your_table_name --top_k 10
```

### Retrieval-only (skip Gemini answer generation)
```bash
python evaluation/evaluate.py --source_table your_table_name --no-generate
```

### Custom dataset file
```bash
python evaluation/evaluate.py --source_table your_table_name --dataset evaluation/my_queries.json
```

### Custom output directory
```bash
python evaluation/evaluate.py --source_table your_table_name --output-dir results/
```

---

## Dataset Format

The `test_dataset.json` file is a JSON array of entries:

```json
[
  {
    "id": 1,
    "query": "What is retrieval-augmented generation?",
    "expected_chunk_ids": ["chunk-id-from-chromadb-1", "chunk-id-from-chromadb-2"],
    "expected_answer_keywords": ["retrieval", "generation", "context", "documents"]
  }
]
```

To find actual chunk IDs, use Step 6 (localhost:8005) or Step 3 (localhost:8002) UI to
inspect your collection, then copy the IDs into `expected_chunk_ids`.

---

## Output Example

```
📂 Loading dataset: evaluation/test_dataset.json
📋 12 queries loaded
  [1/12] Evaluating: What is retrieval-augmented generation?...
    P@5=0.00  R@5=1.00  RR=0.00  Grounded=0.72
  [2/12] Evaluating: How does vector similarity search work?...
    ...

📊 Summary:
   mean_precision_at_k: 0.0
   mean_recall_at_k: 1.0
   mean_mrr: 0.0
   mean_groundedness: 0.68
   mean_answer_relevance: 0.75
   mean_keyword_hit_rate: 0.83

✅ JSON report saved: evaluation/evaluation_report.json
✅ Markdown report saved: evaluation/evaluation_report.md
🎉 Evaluation complete!
```
