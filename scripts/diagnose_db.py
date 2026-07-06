import psycopg2
from psycopg2.extras import execute_values

uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"

print("\n=== PostgreSQL Diagnostic ===\n")

try:
    conn = psycopg2.connect(uri)
    print("OK Connection: OK")
except Exception as e:
    print(f"FAIL Connection FAILED: {e}")
    exit(1)

cur = conn.cursor()

# 1. List all tables
cur.execute("SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
rows = cur.fetchall()
print(f"\n=== Tables in public schema ({len(rows)} tables) ===")
for r in rows:
    print(f"  >> {r[0]}  ({r[1]})")

# 2. Test create + insert + drop
print("\n=== Test Insert ===")
try:
    cur.execute("CREATE TABLE IF NOT EXISTS diag_test (id SERIAL PRIMARY KEY, content TEXT, word_count INT, chunk_index INT, strategy VARCHAR(50), document_name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())")
    conn.commit()
    print("OK CREATE TABLE")

    values = [("doc_test", 0, "This is test chunk one.", 5, "fixed"),
              ("doc_test", 1, "This is test chunk two.", 5, "fixed")]
    execute_values(cur, "INSERT INTO diag_test (document_name, chunk_index, content, word_count, strategy) VALUES %s", values)
    conn.commit()
    print("OK INSERT")

    cur.execute("SELECT COUNT(*) FROM diag_test")
    cnt = cur.fetchone()[0]
    print(f"OK Row count: {cnt}")

    cur.execute("DROP TABLE diag_test")
    conn.commit()
    print("OK CLEANUP")

except Exception as e:
    conn.rollback()
    print(f"FAIL Insert test FAILED: {e}")

# 3. Check chunk tables
print("\n=== Checking saved chunk tables ===")
try:
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname='public'
        AND tablename NOT IN ('address','customer','customer_addresses','customers')
        ORDER BY tablename
    """)
    chunk_tables = cur.fetchall()
    for (tbl,) in chunk_tables:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        cnt = cur.fetchone()[0]
        print(f"  >> {tbl}: {cnt} rows")
except Exception as e:
    print(f"FAIL Table check failed: {e}")

conn.close()
print("\n=== Diagnostic complete ===\n")
