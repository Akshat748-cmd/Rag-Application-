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

# Retrieve top 50 chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=50
)

print(f"Searching for chunk_34 and chunk_33 in top 50 results:")
found_33 = False
found_34 = False

for i in range(len(results['ids'][0])):
    chunk_id = results['ids'][0][i]
    distance = results['distances'][0][i]
    if chunk_id == "chunk_33":
        print(f"-> chunk_33 (Semester VI Value Added) found at Rank {i+1} (Distance: {distance:.4f})")
        found_33 = True
    if chunk_id == "chunk_34":
        print(f"-> chunk_34 (Semester VI Core Subjects) found at Rank {i+1} (Distance: {distance:.4f})")
        found_34 = True

if not found_33:
    print("chunk_33 not in top 50")
if not found_34:
    print("chunk_34 not in top 50")
