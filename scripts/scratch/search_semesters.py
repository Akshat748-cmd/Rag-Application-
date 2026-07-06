import psycopg2

connection_uri = "postgresql://postgres:Akshat%402004@localhost:2004/postgres"

try:
    conn = psycopg2.connect(connection_uri)
    cursor = conn.cursor()
    
    table = "btech_ds_full_syllabus_pdf"
    
    # 1. Search for Roman numerals for Semesters
    for roman in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]:
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM "{table}" 
            WHERE content ILIKE '%Semester {roman}%' OR content ILIKE '%SEMESTER {roman}%'
        """)
        cnt = cursor.fetchone()[0]
        print(f"Semester {roman}: {cnt} chunks")

    # 2. Search for digit representation
    for digit in range(1, 9):
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM "{table}" 
            WHERE content ILIKE '%Semester {digit}%' OR content ILIKE '%SEMESTER {digit}%'
        """)
        cnt = cursor.fetchone()[0]
        print(f"Semester {digit}: {cnt} chunks")

    cursor.close()
    conn.close()

except Exception as e:
    print("Error:", e)
