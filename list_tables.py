import psycopg2

uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"
conn = psycopg2.connect(uri)
cur = conn.cursor()

cur.execute("""
    SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_tables 
    WHERE schemaname = 'public' 
    ORDER BY tablename;
""")

rows = cur.fetchall()
print(f"\n=== PostgreSQL mein kul {len(rows)} table(s) hain ===\n")
for r in rows:
    print(f"  >> {r[0]}  (size: {r[1]})")

conn.close()
