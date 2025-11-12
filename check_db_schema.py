import sqlite3

conn = sqlite3.connect('HotpotQA/metadata_v3.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("="*80)
print("Database Schema Check")
print("="*80)

# Get all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]

print("\nTables in database:")
for table in tables:
    print(f"  - {table}")

# Check each table schema
for table in tables:
    if not table.startswith('sqlite_'):
        print(f"\n{'='*80}")
        print(f"Table: {table}")
        print('='*80)
        
        cur.execute(f"PRAGMA table_info({table})")
        columns = cur.fetchall()
        
        print("\nColumns:")
        for col in columns:
            print(f"  {col['name']:20s} {col['type']:15s} {'NOT NULL' if col['notnull'] else ''} {'PK' if col['pk'] else ''}")
        
        # Show sample data
        cur.execute(f"SELECT COUNT(*) as count FROM {table}")
        count = cur.fetchone()['count']
        print(f"\nTotal rows: {count}")
        
        if count > 0:
            cur.execute(f"SELECT * FROM {table} LIMIT 1")
            sample = cur.fetchone()
            print("\nSample row:")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")

conn.close()
