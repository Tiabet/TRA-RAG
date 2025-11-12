"""
SQLite-based Metadata Database V2 - Path-based Storage
=======================================================
Stores metadata with path-value pairs for better structured search.

Key differences from V1:
- searchable_paths: Each value is stored with its full path from root
- Format: "key1-key2-value, key3-key4-value, ..."
- Better for structured queries and field-specific searches
"""
import sqlite3
import json
import re
from typing import List, Dict, Optional, Tuple, Any


class MetadataDBV2:
    def __init__(self, db_path='HotpotQA/metadata_v3.db'):
        """Initialize database connection"""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        """Create database schema with path-based search"""
        # Main metadata table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                title_normalized TEXT,
                type TEXT,
                subtype TEXT,
                metadata_json TEXT NOT NULL,
                searchable_paths TEXT,  -- 경로-값 쌍들 (쉼표 구분)
                
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
        
        # Full-text search index on paths
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS metadata_fts 
            USING fts5(
                title,
                searchable_paths,
                content='metadata',
                content_rowid='id'
            )
        """)
        
        self.conn.commit()
        print("✓ Database tables created (V2 - Path-based)")
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return ' '.join(text.split())
    
    def extract_paths(self, obj: Any, current_path: str = "", parent_is_entity: bool = False) -> List[str]:
        """
        Recursively extract path-value pairs from nested structure.
        
        Strategy:
        1. Skip 'attributes' and 'relations' keys in path
        2. For entities (dicts with type+subtype), append type-subtype after the value
        3. For relations, extract relation type and target info
        4. Recursively process all nested structures
        
        Args:
            obj: Object to extract from (dict, list, or value)
            current_path: Current path string
            parent_is_entity: Whether parent dict is an entity (has type+subtype)
        
        Returns:
            List of path strings
        """
        paths = []
        
        if isinstance(obj, dict):
            # Check if this is an entity object (has type and subtype)
            is_entity = 'type' in obj and 'subtype' in obj
            
            # Special handling for relation objects
            if 'relation' in obj and 'target' in obj:
                relation_type = obj.get('relation', '')
                target = obj.get('target')
                
                if target and relation_type:
                    # Recursively extract target info
                    target_paths = self.extract_paths(target, "", False)
                    # Combine relation type with target info
                    for tp in target_paths:
                        paths.append(f"{relation_type}-{tp}")
                
                # Also process other fields in the relation (like year, years, etc.)
                for key, value in obj.items():
                    if key in ['relation', 'target']:
                        continue
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        paths.append(f"{relation_type}-{key}-{value}")
                
                return paths
            
            # Process all keys
            for key, value in obj.items():
                # Skip 'metadata' key but process its value (common wrapper)
                if key == 'metadata':
                    paths.extend(self.extract_paths(value, current_path, False))
                    continue
                
                # Skip 'attributes' key but process its value
                if key == 'attributes':
                    paths.extend(self.extract_paths(value, current_path, False))
                    continue
                
                # Skip 'relations' key but process its value
                if key == 'relations':
                    if isinstance(value, list):
                        for item in value:
                            paths.extend(self.extract_paths(item, current_path, False))
                    continue
                
                # For entity objects, handle type/subtype specially
                if is_entity and key in ['type', 'subtype']:
                    # These will be appended to values, not separate paths
                    continue
                
                # Build new path
                new_path = f"{current_path}-{key}" if current_path else key
                
                # Process value
                if value is None:
                    continue
                elif isinstance(value, (str, int, float, bool)):
                    # Simple value - create path
                    # For top-level entity fields (title at root level with type/subtype)
                    if not current_path and is_entity and key in ['title']:
                        # title-value-Type-Subtype
                        paths.append(f"{key}-{value}-{obj['type']}-{obj['subtype']}")
                    else:
                        # Regular simple value
                        paths.append(f"{new_path}-{value}")
                    
                elif isinstance(value, dict):
                    # Check if value is an entity
                    value_is_entity = 'type' in value and 'subtype' in value
                    
                    if value_is_entity:
                        # Extract entity info as a single path
                        entity_path = self._extract_entity_path(value, new_path)
                        if entity_path:
                            paths.append(entity_path)
                    else:
                        # Regular dict - recurse
                        paths.extend(self.extract_paths(value, new_path, False))
                        
                elif isinstance(value, list):
                    # Process list items
                    paths.extend(self.extract_paths(value, new_path, False))
        
        elif isinstance(obj, list):
            # Process each item in list
            for item in obj:
                if isinstance(item, dict):
                    # Check if item is an entity
                    item_is_entity = 'type' in item and 'subtype' in item
                    
                    if item_is_entity:
                        # Extract entity as single path
                        entity_path = self._extract_entity_path(item, current_path)
                        if entity_path:
                            paths.append(entity_path)
                    else:
                        # Regular dict - extract all key-value pairs as one path
                        item_parts = self._flatten_dict(item)
                        if item_parts:
                            if current_path:
                                paths.append(f"{current_path}-{'-'.join(item_parts)}")
                            else:
                                paths.append('-'.join(item_parts))
                                
                elif isinstance(item, (str, int, float, bool)):
                    # Simple value
                    if current_path:
                        paths.append(f"{current_path}-{item}")
        
        return paths
    
    def _extract_entity_path(self, entity: dict, prefix: str = "") -> str:
        """
        Extract entity object as a single path.
        Entity = dict with type and subtype.
        
        Format: prefix-key1-value1-key2-value2-...-Type-Subtype
        """
        parts = []
        if prefix:
            parts.append(prefix)
        
        entity_type = entity.get('type')
        entity_subtype = entity.get('subtype')
        
        # Extract all non-type/subtype fields
        for key, value in entity.items():
            if key in ['type', 'subtype', 'attributes', 'relations']:
                continue
            
            if isinstance(value, (str, int, float, bool)) and value is not None:
                parts.extend([key, str(value)])
            elif isinstance(value, dict):
                # Nested dict - flatten it
                nested = self._flatten_dict(value)
                if nested:
                    parts.append(key)
                    parts.extend(nested)
            elif isinstance(value, list):
                # List - process each item
                for item in value:
                    if isinstance(item, (str, int, float, bool)):
                        parts.extend([key, str(item)])
                    elif isinstance(item, dict):
                        nested = self._flatten_dict(item)
                        if nested:
                            parts.append(key)
                            parts.extend(nested)
        
        # Append type-subtype at the end
        if entity_type and entity_subtype:
            parts.extend([entity_type, entity_subtype])
        
        return '-'.join(parts) if parts else ""
    
    def _flatten_dict(self, obj: dict, include_entity_type: bool = False) -> List[str]:
        """
        Flatten dict into list of key-value pairs.
        Recursively processes nested structures.
        
        Args:
            obj: Dict to flatten
            include_entity_type: If True and obj is entity, include type-subtype at end
        """
        parts = []
        
        # Check if this is an entity
        is_entity = 'type' in obj and 'subtype' in obj
        entity_type = obj.get('type') if is_entity else None
        entity_subtype = obj.get('subtype') if is_entity else None
        
        for key, value in obj.items():
            if key in ['type', 'subtype', 'attributes', 'relations']:
                continue
            
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                parts.extend([key, str(value)])
            elif isinstance(value, dict):
                # Check if nested dict is an entity
                nested_is_entity = 'type' in value and 'subtype' in value
                
                if nested_is_entity:
                    # Entity object - flatten with type-subtype
                    nested = self._flatten_dict(value, include_entity_type=True)
                    if nested:
                        parts.append(key)
                        parts.extend(nested)
                else:
                    # Regular dict
                    nested = self._flatten_dict(value, include_entity_type=False)
                    if nested:
                        parts.append(key)
                        parts.extend(nested)
            elif isinstance(value, list):
                # List - process each item
                for item in value:
                    if isinstance(item, (str, int, float, bool)):
                        parts.extend([key, str(item)])
                    elif isinstance(item, dict):
                        # Check if item is entity
                        item_is_entity = 'type' in item and 'subtype' in item
                        nested = self._flatten_dict(item, include_entity_type=item_is_entity)
                        if nested:
                            parts.append(key)
                            parts.extend(nested)
        
        # Add type-subtype at the end if requested and available
        if include_entity_type and is_entity and entity_type and entity_subtype:
            parts.extend([entity_type, entity_subtype])
        
        return parts
    
    def build_searchable_paths(self, metadata: Dict) -> str:
        """
        Build searchable paths from metadata.
        
        Returns:
            Comma-separated path strings
            Example: "release_year-2013, director-name-Martin Scorsese-type-Person, ..."
        """
        paths = self.extract_paths(metadata)
        # Join with comma and space
        return ', '.join(paths)
    
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
            
            # Build searchable paths
            searchable_paths = self.build_searchable_paths(metadata)
            
            try:
                self.cursor.execute("""
                    INSERT INTO metadata 
                    (title, title_normalized, type, subtype, metadata_json, searchable_paths)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    title,
                    self.normalize_text(title),
                    entity_type,
                    entity_subtype,
                    json.dumps(metadata, ensure_ascii=False),
                    searchable_paths
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
                              If False, search in searchable_paths (comprehensive)
        
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
            # Comprehensive search: searchable_paths
            query = """
                SELECT id, title, type, subtype, metadata_json
                FROM metadata
                WHERE searchable_paths LIKE ?
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
                    WHERE searchable_paths LIKE ?
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
        
        Type 필터링 완화: Type이 맞는 결과가 없으면 type 없이 재검색
        
        Returns:
            List of dicts with 'title', 'metadata', 'type', 'subtype', 'matched_paths'
        """
        # FTS5 query - escape special characters and use phrase search
        fts_query = re.sub(r'["\-\(\)\[\]\{\}\^\~\*:,]', ' ', entity_name)
        fts_query = ' '.join(fts_query.split())  # Remove extra spaces
        
        # Use phrase search with quotes if multiple words
        if ' ' in fts_query:
            fts_query = f'"{fts_query}"'
        
        query = """
            SELECT m.id, m.title, m.type, m.subtype, m.metadata_json, m.searchable_paths
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
            
            # Find which paths matched the search term
            matched_paths = self._find_matched_paths(row['searchable_paths'], search_terms)
            
            results.append({
                'title': row['title'],
                'type': row['type'],
                'subtype': row['subtype'],
                'metadata': metadata_obj,
                'matched_paths': matched_paths  # NEW: 매칭된 path들
            })
        
        # Fallback: If type filtering returned 0 results, retry without type filter
        if len(results) == 0 and entity_type and entity_subtype:
            query_fallback = """
                SELECT m.id, m.title, m.type, m.subtype, m.metadata_json, m.searchable_paths
                FROM metadata_fts f
                JOIN metadata m ON f.rowid = m.id
                WHERE metadata_fts MATCH ?
            """
            self.cursor.execute(query_fallback, [fts_query])
            rows = self.cursor.fetchall()
            
            for row in rows:
                metadata_obj = json.loads(row['metadata_json'])
                matched_paths = self._find_matched_paths(row['searchable_paths'], search_terms)
                
                results.append({
                    'title': row['title'],
                    'type': row['type'],
                    'subtype': row['subtype'],
                    'metadata': metadata_obj,
                    'matched_paths': matched_paths
                })
        
        return results
    
    def _find_matched_paths(self, searchable_paths: str, search_terms: List[str]) -> List[str]:
        """
        Find which paths contain the search terms.
        
        Args:
            searchable_paths: Comma-separated path strings
            search_terms: List of search terms
        
        Returns:
            List of matched path strings
        """
        if not searchable_paths:
            return []
        
        paths = [p.strip() for p in searchable_paths.split(',')]
        matched = []
        
        for path in paths:
            path_lower = path.lower()
            if any(term in path_lower for term in search_terms):
                # Limit path length for display
                if len(path) > 200:
                    path = path[:200] + "..."
                matched.append(path)
        
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
    
    def search_by_path_pattern(self, path_pattern: str) -> List[Dict]:
        """
        Search by specific path pattern.
        
        Example:
            search_by_path_pattern("director-name")
            search_by_path_pattern("cast-role")
        
        Args:
            path_pattern: Path pattern to search for
            
        Returns:
            List of matching entries
        """
        query = """
            SELECT id, title, type, subtype, metadata_json, searchable_paths
            FROM metadata
            WHERE searchable_paths LIKE ?
        """
        
        self.cursor.execute(query, [f"%{path_pattern}%"])
        rows = self.cursor.fetchall()
        
        results = []
        for row in rows:
            metadata_obj = json.loads(row['metadata_json'])
            
            # Extract matching paths
            paths = [p.strip() for p in row['searchable_paths'].split(',')]
            matched_paths = [p for p in paths if path_pattern.lower() in p.lower()]
            
            results.append({
                'title': row['title'],
                'type': row['type'],
                'subtype': row['subtype'],
                'metadata': metadata_obj,
                'matched_paths': matched_paths
            })
        
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
    
    def get_passages_for_metadata(self, metadata_title: str) -> List[Dict]:
        """
        Get original passages associated with a metadata entry
        
        Args:
            metadata_title: Title of the metadata entry
            
        Returns:
            List of passages with their content
        """
        self.cursor.execute("""
            SELECT p.passage_id, p.title, p.content, p.sentence_count
            FROM passages p
            JOIN metadata_passage_mapping m ON p.id = m.passage_id
            JOIN metadata meta ON m.metadata_id = meta.id
            WHERE meta.title = ?
        """, (metadata_title,))
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'passage_id': row['passage_id'],
                'title': row['title'],
                'content': row['content'],
                'sentence_count': row['sentence_count']
            })
        
        return results
    
    def get_metadata_with_passages(self, entity_name: str) -> List[Dict]:
        """
        Search metadata and include associated passages
        
        Args:
            entity_name: Entity name to search for
            
        Returns:
            List of metadata entries with their passages
        """
        # Use FTS search
        results = self.search_by_entity_fts(entity_name)
        
        # Add passages to each result
        for result in results:
            passages = self.get_passages_for_metadata(result['title'])
            result['passages'] = passages
        
        return results
    
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
    import random
    
    print("="*80)
    print("SQLite Metadata Database V2 - Path-based Storage")
    print("="*80)
    
    # Load actual metadata from file
    metadata_file = 'HotpotQA/hotpotqa_sample_200_pure_metadata_v2.json'
    
    print(f"\nLoading metadata from: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    
    print(f"✓ Loaded {len(all_metadata)} metadata entries")
    
    # Randomly select 5 samples
    test_samples = random.sample(all_metadata, min(5, len(all_metadata)))
    print(f"✓ Selected {len(test_samples)} random samples for testing\n")
    
    # Create test database
    db_path = 'test_metadata_v2.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    with MetadataDBV2(db_path) as db:
        db.create_tables()
        
        # Test path extraction for each sample
        print("\n" + "="*80)
        print("Testing Path Extraction")
        print("="*80)
        
        for idx, sample_metadata in enumerate(test_samples, 1):
            title = sample_metadata.get('title', 'Unknown')
            entity_type = sample_metadata.get('type', 'Unknown')
            entity_subtype = sample_metadata.get('subtype', 'Unknown')
            
            print(f"\n{'='*80}")
            print(f"Sample {idx}: {title} ({entity_type} - {entity_subtype})")
            print('='*80)
            
            # Show original structure (truncated)
            print("\n[Original Structure - First 800 chars]")
            original_str = json.dumps(sample_metadata, indent=2, ensure_ascii=False)
            print(original_str[:800] + "..." if len(original_str) > 800 else original_str)
            
            # Extract paths
            paths = db.extract_paths(sample_metadata)
            
            print(f"\n[Extracted {len(paths)} paths]")
            for i, path in enumerate(paths, 1):
                print(f"{i:2d}. {path}")
            
            # Count fields in original
            def count_fields(obj, depth=0):
                """Count all leaf values in nested structure"""
                count = 0
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key in ['type', 'subtype']:  # These are metadata, not content
                            continue
                        if isinstance(value, (dict, list)):
                            count += count_fields(value, depth+1)
                        elif value is not None:
                            count += 1
                elif isinstance(obj, list):
                    for item in obj:
                        count += count_fields(item, depth+1)
                return count
            
            original_field_count = count_fields(sample_metadata)
            print(f"\n[Analysis]")
            print(f"  Original fields (leaf values): {original_field_count}")
            print(f"  Extracted paths: {len(paths)}")
            if len(paths) < original_field_count:
                print(f"  ⚠ WARNING: Missing {original_field_count - len(paths)} fields!")
            else:
                print(f"  ✓ All fields captured")
        
        # Insert metadata
        print("\n" + "="*80)
        print("Inserting metadata...")
        print("="*80)
        
        metadata_entries = [
            {"title": sample.get('title', 'Unknown'), "metadata": sample}
            for sample in test_samples
        ]
        db.insert_metadata_list(metadata_entries)
        print(f"✓ Inserted {len(metadata_entries)} entries")
        
        # Test search on first sample
        print("\n" + "="*80)
        print("Test: Search functionality (Sample 1)")
        print("="*80)
        
        first_title = test_samples[0].get('title', 'Unknown')
        search_term = first_title.split()[0] if first_title != 'Unknown' else 'test'
        results = db.search_by_entity_fts(search_term)
        
        if results:
            result = results[0]
            print(f"\nTitle: {result['title']}")
            print(f"Matched paths: {len(result['matched_paths'])}")
            print("First 5 paths:")
            for path in result['matched_paths'][:5]:
                print(f"  - {path}")
        
        print("\n" + "="*80)
        print("✓ Testing complete!")
        print("="*80)
