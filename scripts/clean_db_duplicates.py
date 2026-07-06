import psycopg2
import psycopg2.extras
import chromadb
import os
import json

# 1. Connect to PostgreSQL
db_uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"
conn = psycopg2.connect(db_uri)
cur = conn.cursor()

# Get all public tables that have a 'content' column
cur.execute("""
    SELECT t.tablename
    FROM pg_tables t
    WHERE t.schemaname = 'public'
      AND t.tablename != 'embeddings'
      AND (SELECT COUNT(*) FROM information_schema.columns c2
           WHERE c2.table_name = t.tablename AND c2.column_name = 'content') > 0;
""")
tables = [r[0] for r in cur.fetchall()]

print("Found tables to deduplicate:", tables)

for t in tables:
    print(f"\nDeduplicating table: {t}")
    # Count before
    cur.execute(f"SELECT COUNT(*) FROM {t};")
    count_before = cur.fetchone()[0]
    
    # Delete duplicate rows (keeping the one with the lowest id)
    cur.execute(f"""
        DELETE FROM {t} a
        USING {t} b
        WHERE a.id > b.id
          AND a.document_name = b.document_name
          AND a.chunk_index = b.chunk_index
          AND a.content = b.content;
    """)
    conn.commit()
    
    cur.execute(f"SELECT COUNT(*) FROM {t};")
    count_after = cur.fetchone()[0]
    print(f"Postgres chunks in {t}: {count_before} -> {count_after} (Removed {count_before - count_after} duplicates)")

# Deduplicate embeddings table
print("\nDeduplicating embeddings table...")
cur.execute("SELECT COUNT(*) FROM embeddings;")
emb_before = cur.fetchone()[0]

cur.execute("""
    DELETE FROM embeddings a
    USING embeddings b
    WHERE a.id > b.id
      AND a.source_table = b.source_table
      AND a.chunk_id = b.chunk_id;
""")
conn.commit()

cur.execute("SELECT COUNT(*) FROM embeddings;")
emb_after = cur.fetchone()[0]
print(f"Postgres embeddings: {emb_before} -> {emb_after} (Removed {emb_before - emb_after} duplicates)")

conn.close()

# 2. Recreate ChromaDB Collections and sync the clean embeddings
chroma_path = os.path.join("step3_vectordb", "chroma_data")
chroma_client = chromadb.PersistentClient(path=chroma_path)

conn = psycopg2.connect(db_uri)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

for t in tables:
    collection_name = t.replace("_", "-")[:63]
    print(f"\nResetting ChromaDB collection: {collection_name}")
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted old Chroma collection: {collection_name}")
    except Exception:
        pass
        
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Fetch deduplicated data from embeddings
    cur.execute("""
        SELECT chunk_id, document_name, chunk_index, content, embedding 
        FROM embeddings 
        WHERE source_table = %s
    """, (t,))
    rows = cur.fetchall()
    
    if rows:
        ids = [f"chunk_{r['chunk_id']}" for r in rows]
        embeddings = [json.loads(r["embedding"]) for r in rows]
        documents = [r["content"] for r in rows]
        metadatas = [{
            "chunk_id": r["chunk_id"],
            "document_name": r["document_name"],
            "chunk_index": r["chunk_index"]
        } for r in rows]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Synced {len(ids)} unique items to Chroma collection '{collection_name}'")

conn.close()
print("\nDeduplication completed successfully!")
