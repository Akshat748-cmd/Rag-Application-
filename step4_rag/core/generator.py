import os
import sys
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None

# ===== GEMINI API KEY CONFIG =====
# Apna Gemini API Key yahan dalein
GEMINI_API_KEY = "AIzaSyB5kLSjusUygSwxUB9vVheyh5VXogQUs1U"
# =================================

# Point ChromaDB to the folder created in Step 3
CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "step3_vectordb", 
    "chroma_data"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_MODEL_NAME)
    return _embed_model

def embed_query(query: str) -> list:
    return get_embed_model().encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()

def retrieve_similar_chunks(source_table: str, query: str, top_k: int = 5) -> List[dict]:
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(f"Chroma collection '{collection_name}' not found. Please Sync in Step 3 first.")

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted_results = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            # Handle L2 (squared distance can be > 1.0) and Cosine distance
            if distance > 1.001:
                score = 1.0 - (distance / 2.0)
            else:
                score = 1.0 - distance
            score = max(0.0, min(1.0, score))

            formatted_results.append({
                "id": results['ids'][0][i],
                "score": round(score, 4),
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
    
    return formatted_results

def generate_response(query: str, source_table: str, top_k: int = 5) -> dict:
    # 4) User input query (Passed as `query`)
    
    # 5) Embed the query & 6) Retrieve similar chunks
    try:
        chunks = retrieve_similar_chunks(source_table, query, top_k)
    except Exception as e:
        return {"success": False, "message": str(e)}
        
    if not chunks:
        return {"success": False, "message": "No relevant context found in Vector DB."}
        
    # 7) Re-rank the chunks (Optional, already ordered by Chroma similarity)
    context_texts = [c["content"] for c in chunks]
    context = "\n\n---\n\n".join(context_texts)
    
    prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If you cannot answer based on the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:"""

    # 8) Generate the final response
    # Use the hardcoded GEMINI_API_KEY if it's set, otherwise fallback to OS environment variable
    api_key = GEMINI_API_KEY if GEMINI_API_KEY != "YOUR_API_KEY_HERE" else os.environ.get("GEMINI_API_KEY")

    if api_key and HAS_GENAI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"[Error using Gemini API: {str(e)}]\n\nPrompt would have been:\n{prompt}"
    else:
        # Mock response if no API key or genai not installed
        msg = ""
        if not HAS_GENAI:
            msg += "[Note: google-generativeai is not installed. You can install it via 'pip install google-generativeai']\n"
        if not api_key:
            msg += "[Note: No API Key provided.]\n"
            
        answer = f"{msg}\n[Mock Response]\nBased on the retrieved context, I can see {len(chunks)} relevant chunks. \n\nTop chunk snippet: {context_texts[0][:100]}...\n\n(Please provide a Gemini API Key to get a real generated response!)"
        
    return {
        "success": True,
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks
    }
