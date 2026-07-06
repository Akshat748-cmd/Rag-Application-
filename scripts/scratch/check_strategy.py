import psycopg2

connection_uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"

try:
    conn = psycopg2.connect(connection_uri)
    cursor = conn.cursor()
    
    # Check strategies in table
    cursor.execute("""
        SELECT DISTINCT strategy, COUNT(*) 
        FROM "btech_ds_full_syllabus_pdf" 
        GROUP BY strategy
    """)
    rows = cursor.fetchall()
    print("Strategies used for btech_ds_full_syllabus_pdf:")
    for r in rows:
        print(f"  - Strategy: {r[0]} | Count: {r[1]} chunks")
        
    # Check details of chunk index 32 and 33
    cursor.execute("""
        SELECT chunk_index, content, word_count 
        FROM "btech_ds_full_syllabus_pdf" 
        WHERE chunk_index IN (32, 33)
        ORDER BY chunk_index
    """)
    details = cursor.fetchall()
    print("\nChunk index 32 & 33 Details:")
    for d in details:
        print(f"  - Chunk {d[0]} (Word count: {d[2]}):")
        print(f"    Content: {repr(d[1])}\n")
        
    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)
