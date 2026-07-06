import psycopg2
import sys

connection_uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"

try:
    conn = psycopg2.connect(connection_uri)
    cursor = conn.cursor()
    
    # 1. Get all table names
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = [row[0] for row in cursor.fetchall()]
    print("Tables in database:", tables)
    
    for table in tables:
        if table.startswith("chunk_") or table == "embeddings":
            continue
        cursor.execute(f"SELECT COUNT(*) FROM \"{table}\"")
        cnt = cursor.fetchone()[0]
        print(f"Table '{table}' has {cnt} rows.")
        
        # Check if Semester 6 keywords exist
        cursor.execute(f"""
            SELECT id, chunk_index, content 
            FROM "{table}" 
            WHERE content ILIKE '%Semester VI%' OR content ILIKE '%Semester 6%' 
            LIMIT 5
        """)
        matches = cursor.fetchall()
        print(f"Matches for 'Semester VI' or 'Semester 6' in '{table}': {len(matches)}")
        for m in matches:
            print(f"  - ID: {m[0]}, Chunk Index: {m[1]}, Preview: {m[2][:150]}...")
            
    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)
