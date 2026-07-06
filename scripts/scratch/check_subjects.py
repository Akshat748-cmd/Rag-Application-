import psycopg2

connection_uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"

try:
    conn = psycopg2.connect(connection_uri)
    cursor = conn.cursor()
    
    table = "btech_ds_full_syllabus_pdf"
    
    subjects = ["Big Data Analytics", "Time Series", "IDS601", "IDS602", "IDS603"]
    for sub in subjects:
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM "{table}" 
            WHERE content ILIKE %s
        """, (f"%{sub}%",))
        cnt = cursor.fetchone()[0]
        print(f"Subject '{sub}': {cnt} chunks")
        
    # Let's print the first 50 characters of a few random chunks to see their content
    cursor.execute(f"SELECT chunk_index, content FROM \"{table}\" ORDER BY chunk_index LIMIT 10")
    rows = cursor.fetchall()
    print("\nFirst 10 chunks:")
    for r in rows:
        print(f"  - Chunk {r[0]}: {r[1][:80].replace(chr(10), ' ')}...")
        
    # Let's print chunks 30 to 40
    cursor.execute(f"SELECT chunk_index, content FROM \"{table}\" WHERE chunk_index BETWEEN 30 AND 40 ORDER BY chunk_index")
    rows = cursor.fetchall()
    print("\nChunks 30 to 40:")
    for r in rows:
        print(f"  - Chunk {r[0]}: {r[1][:80].replace(chr(10), ' ')}...")
        
    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)
