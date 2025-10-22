"""Check DB tables"""
import sqlite3

conn = sqlite3.connect('HotpotQA/metadata_v2.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("Tables in database:")
for table in tables:
    print(f"  - {table[0]}")
    
    # Get row count
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"    Rows: {count}")
    except:
        print(f"    (Cannot count - virtual table or error)")

conn.close()
