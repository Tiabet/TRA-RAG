import sqlite3

conn = sqlite3.connect('HotpotQA/metadata_v2.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

print("Tables in database:")
for table in tables:
    print(f"  - {table}")
    
    # Get schema for each table
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    print(f"    Columns:")
    for col in columns:
        print(f"      {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"    Row count: {count:,}\n")

conn.close()
