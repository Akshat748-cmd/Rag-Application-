import psycopg2

conn = psycopg2.connect('postgresql://postgres:Akshat%402004@localhost:2004/postgres')
cur = conn.cursor()

# List all tables
cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
tables = cur.fetchall()
print(f"\n=== Total {len(tables)} tables in PostgreSQL ===")
for t in tables:
    cur.execute(f"SELECT COUNT(1) FROM {t[0]}")
    cnt = cur.fetchone()[0]
    print(f"  >> {t[0]}  ({cnt} rows)")

# Show sample from plant health table
tbl = 'predictive_plant_health_technology_for_home_gardens_study_guide'
print(f"\n=== Sample rows from: {tbl} ===")
cur.execute(f"SELECT chunk_index, word_count, strategy, LEFT(content, 100) FROM {tbl} ORDER BY chunk_index LIMIT 5")
rows = cur.fetchall()
for r in rows:
    print(f"  Chunk #{r[0]} | {r[1]} words | {r[2]}")
    print(f"    Text: {r[3]}...")
    print()

conn.close()
