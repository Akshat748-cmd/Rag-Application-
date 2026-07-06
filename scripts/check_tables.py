import urllib.request, json

# Check step2 tables API
try:
    req = urllib.request.urlopen('http://127.0.0.1:8001/api/tables', timeout=5)
    data = json.loads(req.read().decode())
    print("=== Step 2 API se tables ===")
    for t in data['tables']:
        print(f"  - {t['name']}")
    print(f"Total: {len(data['tables'])} tables")
except Exception as e:
    print(f"Step 2 not running: {e}")

# Also directly check DB which tables have 'content' column
import psycopg2
conn = psycopg2.connect('postgresql://postgres:Akshat%402004@localhost:2004/postgres')
cur = conn.cursor()
cur.execute("""
    SELECT t.tablename,
           (SELECT COUNT(*) FROM information_schema.columns c2
            WHERE c2.table_name = t.tablename AND c2.column_name = 'content') > 0 AS has_content
    FROM pg_tables t
    WHERE t.schemaname = 'public'
    ORDER BY t.tablename;
""")
rows = cur.fetchall()
print("\n=== DB mein sabhi tables aur unka 'content' column status ===")
for r in rows:
    status = "CONTENT COLUMN HAI (embed hogi)" if r[1] else "content column NAHI (nahi dikhegi)"
    print(f"  {r[0]}: {status}")
conn.close()
