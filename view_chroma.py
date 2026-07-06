import chromadb
import os
import json

# Connect to ChromaDB
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "step3_vectordb", "chroma_data")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Sab collections ki list nikalo
collections = client.list_collections()
if not collections:
    print("Abhi tak koi data sync nahi hua hai.")
    exit()

# Pehli collection uthao (jaise ml_notes)
first_collection = client.get_collection(collections[0].name)

# Data get karo (start ke 5 chunks dekhne ke liye)
data = first_collection.get(limit=5)

print(f"\n=========================================")
print(f"Collection Name: {collections[0].name}")
print(f"Total Chunks Stored: {first_collection.count()}")
print(f"=========================================\n")

# Data Print karo
for i in range(len(data['ids'])):
    print(f"Chunk ID: {data['ids'][i]}")
    print(f"Text: {data['documents'][i][:100]}... (truncated)")
    print(f"Metadata: {data['metadatas'][i]}")
    print("-" * 50)
