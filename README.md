# RAG Application — Production-Grade Portfolio Project

A modular, production-ready Retrieval-Augmented Generation (RAG) simulator where each step
is an independent FastAPI app on its own port:

1. **Step 1: Chunking & Storing in PostgreSQL** — Split text/PDF files and save them.
2. **Step 2: Embedding Generation** — Calculate chunk embeddings in Python and store in PostgreSQL.
3. **Step 3: Vector DB Sync** — Sync Postgres embeddings into ChromaDB collection.
4. **Step 4: Simple RAG Generation** — Retrieve relevant chunks from ChromaDB and feed to Gemini LLM.
5. **Step 5: Query Embedding Visualization** — Convert input query to vector.
6. **Step 6: Retrieval Exploration** — Explore ChromaDB native vector similarity.
7. **Step 7: Reranking** — Re-evaluate results using Simulated Hybrid Scoring or True Cross-Encoder.
8. **Step 8: Complete RAG Pipeline** — Combined end-to-end retrieve + rerank + generate workflow.

---

## Configuration & Feature Flags

The project uses a global `.env` file at the root to store all credentials and feature toggles:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:2004/postgres

# Reranking strategy
USE_CROSS_ENCODER=false        # true = real CrossEncoder model (~80MB download on first use)

# Hybrid retrieval (BM25 + vector)
USE_HYBRID_SEARCH=false        # true = blend BM25 keyword score with cosine similarity

# LLM-powered query expansion
USE_QUERY_REWRITING=false      # true = Gemini rewrites query for better recall before retrieval
```

### Note on Databases
- **Postgres**: Credentials are loaded securely on the backend from `DATABASE_URL` in `.env`.
- **ChromaDB**: The Chroma database directory (`step3_vectordb/chroma_data/`) is ignored by Git.

---

## Running the Application

1. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your Gemini API key and Postgres credentials
   ```

2. **Start all FastAPI apps**:
   ```powershell
   .\start.ps1
   ```

3. **Optional — Enable Hybrid Search** (BM25 + vector):
   Set `USE_HYBRID_SEARCH=true` in `.env`, then restart the relevant apps.

4. **Optional — Enable Query Rewriting**:
   Set `USE_QUERY_REWRITING=true` in `.env`, then restart.

5. **Optional — Enable True Cross-Encoder Reranker**:
   Set `USE_CROSS_ENCODER=true` in `.env`.
   *Note: Downloads the MiniLM CrossEncoder model (~80MB) on first use.*

---

## Evaluation

Run the evaluation framework to measure retrieval and answer quality:

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run evaluation (generates evaluation_report.md and evaluation_report.json)
python evaluation/evaluate.py --source_table your_table_name

# Retrieval-only (no Gemini API calls)
python evaluation/evaluate.py --source_table your_table_name --no-generate

# Custom top-K
python evaluation/evaluate.py --source_table your_table_name --top_k 10
```

**Metrics computed:**
- **Precision@K** — Fraction of retrieved chunks that are relevant
- **Recall@K** — Fraction of relevant chunks retrieved
- **MRR** — Mean Reciprocal Rank
- **Groundedness** — Answer keyword overlap with context
- **Answer Relevance** — Answer keyword overlap with query
- **Keyword Hit Rate** — Expected keyword coverage in answer

See [`evaluation/README.md`](evaluation/README.md) for full details.

---

## Architecture

All steps share data through **PostgreSQL** (chunk text + metadata) and **ChromaDB**
(vector embeddings), and share code through **`shared/rag_core.py`**. There are no
cross-service HTTP calls — each step is fully independent.

```
shared/rag_core.py        <- Embedding, retrieval (vector + BM25), reranking, Gemini
backend/core/chunker.py   <- 7 chunking strategies (fixed, recursive, sentence,
                             paragraph, token, sliding_window, semantic)
evaluation/evaluate.py    <- Standalone evaluation script (no server needed)
```
