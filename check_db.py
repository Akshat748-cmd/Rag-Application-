import psycopg2
import sys
from psycopg2.extras import RealDictCursor

# Fix Windows terminal Unicode encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"

try:
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(uri)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Check if document_chunks table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'document_chunks'
        );
    """)
    exists = cur.fetchone()["exists"]
    
    if not exists:
        print("Table 'document_chunks' does not exist in the database.")
    else:
        cur.execute("SELECT COUNT(*) as total FROM document_chunks;")
        total = cur.fetchone()["total"]
        print(f"Table 'document_chunks' exists! Total rows: {total}")
        
        cur.execute("SELECT id, document_name, chunk_index, word_count, strategy, created_at, SUBSTRING(content, 1, 50) as preview FROM document_chunks ORDER BY id DESC LIMIT 10;")
        rows = cur.fetchall()
        
        print("\nLast 10 chunks inserted:")
        print(f"{'ID':<5} | {'Document':<20} | {'Index':<6} | {'Words':<6} | {'Strategy':<14} | {'Content Preview'}")
        print("-" * 110)
        for r in rows:
            print(f"{r['id']:<5} | {str(r['document_name']):<20} | {r['chunk_index']:<6} | {r['word_count']:<6} | {r['strategy']:<14} | {r['preview']}...")
            
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
