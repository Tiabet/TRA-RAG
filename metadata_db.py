"""
SQLite-based Metadata Database for Entity Retrieval
====================================================
Stores metadata with hierarchical structure and enables deep value search.
"""
import sqlite3
import json
import re
from typing import List, Dict, Optional, Tuple


class MetadataDB:
    def __init__(self, db_path='HotpotQA/metadata_v2.db'):
        """Initialize database connection"""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        """Create database schema with JSON support"""
        # Main metadata table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                title_normalized TEXT,
                type TEXT,
                subtype TEXT,
                metadata_json TEXT NOT NULL,
                searchable_text TEXT,  -- 모든 value를 flatten한 텍스트
                
                UNIQUE(title)
            )
        """)
        
        # Indexes for fast search
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_title_normalized 
            ON metadata(title_normalized)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_type_subtype 
            ON metadata(type, subtype)
        """)
        
        # Full-text search index
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS metadata_fts 
            USING fts5(
                title,
                searchable_text,
                content='metadata',
                content_rowid='id'
            )
        """)
        
        self.conn.commit()
        print("✓ Database tables created")
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return ' '.join(text.split())
    
    @staticmethod
    def extract_all_values(obj, parent_key='') -> List[str]:
        """
        Recursively extract all string values from nested JSON/dict.
        Returns list of all text values found.
        """
        values = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    values.extend(MetadataDB.extract_all_values(value, key))
                elif value is not None:
                    # Convert to string and add
                    values.append(str(value))
        
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    values.extend(MetadataDB.extract_all_values(item, parent_key))
                elif item is not None:
                    values.append(str(item))
        
        else:
            if obj is not None:
                values.append(str(obj))
        
        return values
    
    def build_searchable_text(self, metadata: Dict) -> str:
        """
        Build searchable text from all values in metadata.
        모든 value를 추출해서 하나의 텍스트로 만듦.
        """
        all_values = self.extract_all_values(metadata)
        # Join all values with spaces
        combined = ' '.join(all_values)
        # Normalize for search
        return self.normalize_text(combined)
    
    def insert_metadata_list(self, metadata_list: List[Dict]):
        """
        Insert list of metadata entries.
        
        Args:
            metadata_list: List of {"title": str, "metadata": dict}
        """
        print(f"Inserting {len(metadata_list)} metadata entries...")
        
        inserted = 0
        skipped = 0
        
        for entry in metadata_list:
            title = entry['title']
            metadata = entry['metadata']
            
            # Extract type/subtype
            entity_type = metadata.get('type')
            entity_subtype = metadata.get('subtype')
            
            # Build searchable text from ALL values
            searchable_text = self.build_searchable_text(metadata)
            
            try:
                self.cursor.execute("""
                    INSERT INTO metadata 
                    (title, title_normalized, type, subtype, metadata_json, searchable_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    title,
                    self.normalize_text(title),
                    entity_type,
                    entity_subtype,
                    json.dumps(metadata, ensure_ascii=False),
                    searchable_text
                ))
                inserted += 1
                
            except sqlite3.IntegrityError:
                # Duplicate title
                skipped += 1
                continue
        
        self.conn.commit()
        print(f"✓ Inserted: {inserted}, Skipped: {skipped}")
        
        # Update FTS index
        self.cursor.execute("""
            INSERT INTO metadata_fts(metadata_fts) VALUES('rebuild')
        """)
        self.conn.commit()
        print("✓ Full-text search index updated")
    
    def search_by_entity(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        entity_subtype: Optional[str] = None,
        search_title_only: bool = False
    ) -> List[Dict]:
        """
        Search metadata by entity name.
        
        Args:
            entity_name: Entity name to search for
            entity_type: Optional type filter
            entity_subtype: Optional subtype filter
            search_title_only: If True, search only in title (fast)
                              If False, search in all values (comprehensive)
        
        Returns:
            List of matching entries with title and metadata
        """
        normalized_entity = self.normalize_text(entity_name)
        
        if search_title_only:
            # Fast search: title only
            query = """
                SELECT id, title, type, subtype, metadata_json
                FROM metadata
                WHERE title_normalized LIKE ?
            """
            params = [f"%{normalized_entity}%"]
        else:
            # Comprehensive search: all values
            query = """
                SELECT id, title, type, subtype, metadata_json
                FROM metadata
                WHERE searchable_text LIKE ?
            """
            params = [f"%{normalized_entity}%"]
        
        # Add type/subtype filters if provided
        if entity_type and entity_subtype:
            query += " AND type = ? AND subtype = ?"
            params.extend([entity_type, entity_subtype])
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'title': row['title'],
                'metadata': json.loads(row['metadata_json'])
            })
        
        # Fallback: If type filtering returned 0 results, retry without type filter
        if len(results) == 0 and entity_type and entity_subtype:
            if search_title_only:
                query_fallback = """
                    SELECT id, title, type, subtype, metadata_json
                    FROM metadata
                    WHERE title_normalized LIKE ?
                """
            else:
                query_fallback = """
                    SELECT id, title, type, subtype, metadata_json
                    FROM metadata
                    WHERE searchable_text LIKE ?
                """
            
            self.cursor.execute(query_fallback, [f"%{normalized_entity}%"])
            rows = self.cursor.fetchall()
            
            for row in rows:
                results.append({
                    'title': row['title'],
                    'metadata': json.loads(row['metadata_json'])
                })
        
        return results
    
    def search_by_entity_fts(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        entity_subtype: Optional[str] = None
    ) -> List[Dict]:
        """
        Search using Full-Text Search (FTS5) for better performance.
        모든 value를 검색하되 FTS를 사용해서 빠르게.
        
        Type 필터링 완화: Type이 맞는 결과가 없으면 type 없이 재검색
        
        Returns:
            List of dicts with 'title', 'metadata', 'type', 'subtype', 'matched_fields'
        """
        # FTS5 query - escape special characters and use phrase search
        # Remove FTS special characters: " - ( ) [ ] { } ^ ~ * : ,
        import re
        fts_query = re.sub(r'["\-\(\)\[\]\{\}\^\~\*:,]', ' ', entity_name)
        fts_query = ' '.join(fts_query.split())  # Remove extra spaces
        
        # Use phrase search with quotes if multiple words
        if ' ' in fts_query:
            fts_query = f'"{fts_query}"'
        
        query = """
            SELECT m.id, m.title, m.type, m.subtype, m.metadata_json
            FROM metadata_fts f
            JOIN metadata m ON f.rowid = m.id
            WHERE metadata_fts MATCH ?
        """
        params = [fts_query]
        
        # Add type/subtype filters if provided
        if entity_type and entity_subtype:
            query += " AND m.type = ? AND m.subtype = ?"
            params.extend([entity_type, entity_subtype])
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        results = []
        search_terms = entity_name.lower().split()
        
        for row in rows:
            metadata_obj = json.loads(row['metadata_json'])
            
            # Find which fields matched the search term
            matched_fields = self._find_matched_fields(metadata_obj, search_terms)
            
            results.append({
                'title': row['title'],
                'type': row['type'],
                'subtype': row['subtype'],
                'metadata': metadata_obj,
                'matched_fields': matched_fields  # NEW: 매칭된 key-value 쌍
            })
        
        # Fallback: If type filtering returned 0 results, retry without type filter
        if len(results) == 0 and entity_type and entity_subtype:
            query_fallback = """
                SELECT m.id, m.title, m.type, m.subtype, m.metadata_json
                FROM metadata_fts f
                JOIN metadata m ON f.rowid = m.id
                WHERE metadata_fts MATCH ?
            """
            self.cursor.execute(query_fallback, [fts_query])
            rows = self.cursor.fetchall()
            
            for row in rows:
                metadata_obj = json.loads(row['metadata_json'])
                matched_fields = self._find_matched_fields(metadata_obj, search_terms)
                
                results.append({
                    'title': row['title'],
                    'type': row['type'],
                    'subtype': row['subtype'],
                    'metadata': metadata_obj,
                    'matched_fields': matched_fields
                })
        
        return results
    
    def _find_matched_fields(self, metadata: Dict, search_terms: List[str]) -> List[Dict]:
        """
        Find which metadata fields contain the search terms.
        Returns list of {key: value} pairs that matched.
        """
        matched = []
        
        def search_recursive(obj, path=""):
            """Recursively search through nested dict/list"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if value (converted to string) contains any search term
                    value_str = str(value).lower()
                    if any(term in value_str for term in search_terms):
                        # Limit value length for display
                        display_value = str(value)
                        if len(display_value) > 200:
                            display_value = display_value[:200] + "..."
                        
                        matched.append({
                            'key': current_path,
                            'value': display_value
                        })
                    
                    # Continue recursion for nested structures
                    if isinstance(value, (dict, list)):
                        search_recursive(value, current_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    search_recursive(item, current_path)
        
        search_recursive(metadata)
        return matched
    
    def get_by_title(self, title: str) -> Optional[Dict]:
        """Get metadata by exact title"""
        self.cursor.execute("""
            SELECT title, metadata_json
            FROM metadata
            WHERE title = ?
        """, (title,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'title': row['title'],
                'metadata': json.loads(row['metadata_json'])
            }
        return None
    
    def search_by_type(
        self,
        entity_type: str,
        entity_subtype: Optional[str] = None
    ) -> List[Dict]:
        """
        Search all passages matching the given type and subtype.
        Used for Stage 1-B type filtering.
        
        Args:
            entity_type: Entity type to filter by
            entity_subtype: Optional entity subtype to filter by
            
        Returns:
            List of passages with matching type/subtype
        """
        if entity_subtype:
            query = """
                SELECT title, type, subtype, metadata_json
                FROM metadata
                WHERE type = ? AND subtype = ?
            """
            params = [entity_type, entity_subtype]
        else:
            query = """
                SELECT title, type, subtype, metadata_json
                FROM metadata
                WHERE type = ?
            """
            params = [entity_type]
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'title': row['title'],
                'metadata': json.loads(row['metadata_json'])
            })
        
        return results
    
    def search_by_relation(self, target_entity: str) -> List[Dict]:
        """
        Search metadata that has relations to target_entity.
        relations 배열에서 특정 entity를 참조하는 passage 찾기.
        """
        self.cursor.execute("""
            SELECT title, metadata_json
            FROM metadata
            WHERE json_extract(metadata_json, '$.relations') IS NOT NULL
        """)
        
        results = []
        for row in self.cursor.fetchall():
            metadata = json.loads(row['metadata_json'])
            relations = metadata.get('relations', [])
            
            # Check if any relation matches target_entity
            for relation in relations:
                if isinstance(relation, dict):
                    entity = relation.get('entity', '')
                    if self.normalize_text(target_entity) in self.normalize_text(entity):
                        results.append({
                            'title': row['title'],
                            'metadata': metadata
                        })
                        break
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        self.cursor.execute("SELECT COUNT(*) as total FROM metadata")
        total = self.cursor.fetchone()['total']
        
        self.cursor.execute("""
            SELECT type, COUNT(*) as count
            FROM metadata
            WHERE type IS NOT NULL
            GROUP BY type
            ORDER BY count DESC
        """)
        type_counts = {row['type']: row['count'] for row in self.cursor.fetchall()}
        
        return {
            'total_entries': total,
            'type_distribution': type_counts
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage and testing
if __name__ == "__main__":
    import os
    
    print("="*60)
    print("SQLite Metadata Database - Initialization")
    print("="*60)
    
    # Remove old database for fresh start
    db_path = 'metadata.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✓ Removed old database")
    
    # Initialize database
    with MetadataDB(db_path) as db:
        # Create tables
        db.create_tables()
        
        # Load metadata from JSON
        print("\n" + "="*60)
        print("Loading metadata from JSON file...")
        print("="*60)
        
        metadata_path = 'HotpotQA/hotpotqa_sample_200_pure_metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        print(f"Loaded {len(metadata_list)} metadata entries")
        
        # Insert into database
        db.insert_metadata_list(metadata_list)
        
        # Show statistics
        print("\n" + "="*60)
        print("Database Statistics")
        print("="*60)
        stats = db.get_stats()
        print(f"Total entries: {stats['total_entries']}")
        print(f"\nType distribution:")
        for entity_type, count in sorted(stats['type_distribution'].items(), key=lambda x: -x[1])[:10]:
            print(f"  {entity_type}: {count}")
        
        # Test searches
        print("\n" + "="*60)
        print("Test 1: Title-only search (fast)")
        print("="*60)
        results = db.search_by_entity("Baltic Cup", search_title_only=True)
        print(f"Found {len(results)} matches for 'Baltic Cup' in titles:")
        for r in results[:5]:
            print(f"  - {r['title']}")
        
        print("\n" + "="*60)
        print("Test 2: All-values search (comprehensive)")
        print("="*60)
        results = db.search_by_entity("Baltic Cup", search_title_only=False)
        print(f"Found {len(results)} matches for 'Baltic Cup' in all values:")
        for r in results[:5]:
            print(f"  - {r['title']}")
        
        print("\n" + "="*60)
        print("Test 3: FTS search (fast + comprehensive)")
        print("="*60)
        results = db.search_by_entity_fts("Baltic Cup")
        print(f"Found {len(results)} matches using FTS:")
        for r in results[:5]:
            print(f"  - {r['title']}")
        
        print("\n" + "="*60)
        print("Test 4: Search in nested values")
        print("="*60)
        # Search for something that's NOT in title but in metadata
        results = db.search_by_entity("Estonia", search_title_only=False)
        print(f"Found {len(results)} matches for 'Estonia' in all values:")
        for r in results[:5]:
            print(f"  - {r['title']}")
        
        print("\n" + "="*60)
        print("Test 5: Type-filtered search")
        print("="*60)
        results = db.search_by_entity(
            "Baltic Cup",
            entity_type="Event",
            entity_subtype="SportsEvent",
            search_title_only=False
        )
        print(f"Found {len(results)} matches for 'Baltic Cup' (Event/SportsEvent):")
        for r in results[:5]:
            print(f"  - {r['title']}")
        
        print("\n" + "="*60)
        print("✓ Database initialization complete!")
        print("="*60)
