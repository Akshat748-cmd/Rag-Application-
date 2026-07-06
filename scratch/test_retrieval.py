import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "step3_vectordb", 
    "chroma_data"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

query = "b.tech 6 sem syllabus"
query_embedding = embed_model.encode(query, show_progress_bar=False, convert_to_numpy=True).tolist()

collection = chroma_client.get_collection(name="btech-ds-full-syllabus-pdf")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

print(f"ChromaDB retrieval results for: '{query}'")
for i in range(len(results['ids'][0])):
    distance = results['distances'][0][i]
    score = 1.0 - distance
    chunk_id = results['ids'][0][i]
    content = results['documents'][0][i]
    metadata = results['metadatas'][0][i]
    print(f"\nRank {i+1}: ID: {chunk_id} | Distance: {distance:.4f} | Score: {score*100:.1f}%")
    print(f"  Metadata: {metadata}")
    print(f"  Content: {content[:150].replace(chr(10), ' ')}...")
