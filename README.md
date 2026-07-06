# RAG Application Learning Simulator

A modular simulator showing step-by-step implementation of a Retrieval-Augmented Generation (RAG) system:
1. **Step 1: Chunking & Storing in PostgreSQL**: Split text/PDF files and save them.
2. **Step 2: Embedding Generation**: Calculate chunk embeddings in Python and store in PostgreSQL.
3. **Step 3: Vector DB Sync**: Sync Postgres embeddings into ChromaDB collection.
4. **Step 4: Simple RAG Generation**: Retrieve relevant chunks from ChromaDB and feed to Gemini LLM.
5. **Step 5: Query Embedding Visualization**: Convert input query to vector.
6. **Step 6: Retrieval Exploration**: Explore ChromaDB native vector similarity.
7. **Step 7: Reranking**: Re-evaluate results using Simulated Hybrid Scoring or True Cross-Encoder.
8. **Step 8: Complete RAG Pipeline**: Combined end-to-end retrieve + rerank + generate workflow.

---

## 🛠️ Configuration & Security

The project uses a global `.env` file at the root to store all credentials securely:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:2004/postgres
USE_CROSS_ENCODER=false
```

### Note on Databases
- **Postgres**: Credentials are loaded securely on the backend from `DATABASE_URL` in `.env`. No credentials are sent from the client-side.
- **ChromaDB**: The Chroma database directory (`step3_vectordb/chroma_data/`) is ignored by Git. It is generated locally when you sync embeddings in Step 3.

---

## 🚀 Running the Application

1. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your Gemini API key and Postgres credentials:
   ```bash
   cp .env.example .env
   ```

2. **Start the FastAPI Backend**:
   Run the main launcher script:
   ```powershell
   .\start.ps1
   ```
   This will start the FastAPI backend server on `http://localhost:8000`.

3. **True Cross-Encoder Reranker**:
   By default, Step 7 uses a fast simulated hybrid reranking strategy. To enable a real Cross-Encoder model (`sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`), set:
   ```env
   USE_CROSS_ENCODER=true
   ```
   *Note: Enabling this will automatically download the MiniLM Cross-Encoder model (~80MB) on first use.*
