"""
Step 8 — LLM Generation App Core
Retrieves chunks, re-ranks them, builds the prompt, and queries Gemini LLM.
"""
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import google.generativeai as genai
import re
import time

# API Key Config
GEMINI_API_KEY = "AIzaSyB5kLSjusUygSwxUB9vVheyh5VXogQUs1U"
_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None

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

def clean_and_tokenize(text: str) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"is", "a", "the", "in", "of", "to", "for", "and", "about", "what", "with", "this", "or", "an", "on", "at", "by", "from"}
    return [w for w in words if w not in stopwords]

def calculate_keyword_score(query: str, chunk_content: str) -> float:
    query_tokens = set(clean_and_tokenize(query))
    if not query_tokens:
        return 0.0
    chunk_tokens = set(clean_and_tokenize(chunk_content))
    matches = query_tokens.intersection(chunk_tokens)
    return len(matches) / len(query_tokens)

def generate_rag_response(source_table: str, query: str, top_k: int = 3) -> dict:
    t0 = time.perf_counter()
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "success": False,
            "message": f"Chroma collection '{collection_name}' not found. Sync in Step 3 first."
        }

    # 1. Retrieve pool of chunks
    retrieve_count = min(top_k * 2, collection.count())
    if retrieve_count == 0:
        return {"success": False, "message": "No chunks found in database."}

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieve_count
    )

    initial_chunks = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            if distance > 1.001:
                score = 1.0 - (distance / 2.0)
            else:
                score = 1.0 - distance
            score = max(0.0, min(1.0, score))

            initial_chunks.append({
                "id": results['ids'][0][i],
                "semantic_score": round(score, 4),
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })

    # 2. Re-rank (60% semantic + 40% keyword overlap)
    reranked_chunks = []
    for chunk in initial_chunks:
        keyword_score = calculate_keyword_score(query, chunk["content"])
        hybrid_score = (chunk["semantic_score"] * 0.6) + (keyword_score * 0.4)
        reranked_chunks.append({
            **chunk,
            "final_score": round(hybrid_score, 4)
        })

    reranked_chunks.sort(key=lambda x: x["final_score"], reverse=True)
    top_chunks = reranked_chunks[:top_k]

    # 3. Assemble Prompt
    context_texts = [c["content"] for c in top_chunks]
    context = "\n\n---\n\n".join(context_texts)
    
    prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If you cannot answer based on the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:"""

    # 4. LLM Generation
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"[Error using Gemini API: {str(e)}]\n\nPrompt was:\n{prompt}"

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "success": True,
        "query": query,
        "answer": answer,
        "prompt": prompt,
        "chunks": top_chunks,
        "elapsed_ms": elapsed_ms
    }
