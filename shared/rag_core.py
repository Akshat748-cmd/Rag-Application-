"""
Shared RAG Core Module
Provides common functions for embedding, retrieval, re-ranking, and response generation.
"""
import os
import re
import sys
import time
from typing import List, Optional
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

# Load configurations from root .env
load_dotenv(find_dotenv())

_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None
_cross_encoder_model = None

# ChromaDB connection config
CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "step3_vectordb", 
    "chroma_data"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def get_embed_model() -> SentenceTransformer:
    """Lazy load SentenceTransformer model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_MODEL_NAME)
    return _embed_model


def get_cross_encoder():
    """Lazy load CrossEncoder model."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder_model


def embed_query(query: str) -> list:
    """Generate 384-dimensional query embedding."""
    return get_embed_model().encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()


def retrieve_chunks(source_table: str, query: str, retrieve_count: int = 5) -> List[dict]:
    """Retrieve raw chunks from ChromaDB and return with semantic scores (cosine similarity)."""
    collection_name = source_table.replace("_", "-")[:63]
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(f"Chroma collection '{collection_name}' not found. Please sync in Step 3 first.")

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieve_count
    )

    formatted_results = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            # Consistently map cosine distance to similarity score:
            # For cosine distance range [0, 2], similarity is 1 - distance, clamped to [0, 1]
            score = 1.0 - distance
            score = max(0.0, min(1.0, score))

            formatted_results.append({
                "id": results['ids'][0][i],
                "score": round(score, 4),
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
    return formatted_results


def clean_and_tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, and split into words."""
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {
        "is", "a", "the", "in", "of", "to", "for", "and", "about", 
        "what", "with", "this", "or", "an", "on", "at", "by", "from"
    }
    return [w for w in words if w not in stopwords]


def calculate_keyword_score(query: str, chunk_content: str) -> float:
    """Calculate query token overlap ratio in chunk content."""
    query_tokens = set(clean_and_tokenize(query))
    if not query_tokens:
        return 0.0
    chunk_tokens = set(clean_and_tokenize(chunk_content))
    matches = query_tokens.intersection(chunk_tokens)
    return len(matches) / len(query_tokens)


def rerank_chunks(query: str, chunks: List[dict], top_k: int, use_cross_encoder: bool = False) -> List[dict]:
    """Re-ranks a list of chunks using simulated hybrid scoring or true cross-encoder."""
    if not chunks:
        return []

    if use_cross_encoder:
        try:
            model = get_cross_encoder()
            pairs = [[query, c["content"]] for c in chunks]
            scores = model.predict(pairs)
            
            reranked = []
            for idx, c in enumerate(chunks):
                # Sigmoid scaling for logits to range 0-1
                raw_score = float(scores[idx])
                scaled_score = 1.0 / (1.0 + np.exp(-raw_score))
                reranked.append({
                    **c,
                    "final_score": round(scaled_score, 4),
                    "rerank_type": "cross-encoder"
                })
        except Exception as e:
            # Fallback to hybrid scoring if loading cross-encoder fails
            print(f"[RAG-Core] CrossEncoder failed: {e}. Falling back to hybrid re-ranking.")
            use_cross_encoder = False

    if not use_cross_encoder:
        reranked = []
        for idx, c in enumerate(chunks):
            keyword_score = calculate_keyword_score(query, c["content"])
            # Use retrieved semantic score
            semantic = c.get("score", c.get("semantic_score", 0.0))
            hybrid_score = (semantic * 0.6) + (keyword_score * 0.4)
            reranked.append({
                **c,
                "keyword_score": round(keyword_score, 4),
                "final_score": round(hybrid_score, 4),
                "rerank_type": "hybrid"
            })

    # Sort descending by final score
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked[:top_k]


def build_prompt(context: str, query: str) -> str:
    """Forms the prompt used for Gemini LLM generation."""
    return f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. If you cannot answer based on the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:"""


def call_gemini(prompt: str, api_key: Optional[str] = None) -> str:
    """Safe call to Gemini API with clear error details or mock backup responses."""
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY_HERE" or api_key.startswith("AIzaSy") and len(api_key) < 15:
        # Invalid or missing key
        return (
            "[Note: Gemini API key is missing or not set in root .env.]\n"
            "[Mock Response]\n"
            "Here is what would have been generated based on the context provided in your request. "
            "Please configure GEMINI_API_KEY in the root .env to enable live generation."
        )

    if not HAS_GENAI:
        return (
            "[Note: google-generativeai package is not installed. Run 'pip install google-generativeai'.]\n"
            "[Mock Response]\n"
            "Please install the google-generativeai module to get a real LLM generation."
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error calling Gemini API: {str(e)}]\n\n[Prompt context used]:\n{prompt[:300]}..."
