"""
Metadata to Passage Mapping
============================
메타데이터를 원본 passage와 연결하는 매핑 테이블 생성

방법 1: passage_mapping 테이블 추가
- metadata.id와 passage_id를 매핑
- 한 메타데이터가 여러 passage에 나타날 수 있음 (다대다 관계)

방법 2: metadata 테이블에 passage_ids 컬럼 추가
- JSON 배열로 저장
- 간단하지만 정규화가 안됨

방법 3: 별도 passage 테이블 생성
- passage를 독립적으로 관리
- metadata와 N:M 관계
"""

import json
import sqlite3
from typing import List, Dict, Set


class PassageMapper:
    """Map metadata to original passages"""
    
    def __init__(self, db_path='HotpotQA/metadata_v3.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def create_passage_tables(self):
        """Create tables for passage storage and mapping"""
        
        # 1. Passage 테이블: 원본 passage 저장
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                passage_id TEXT NOT NULL,           -- HotpotQA의 _id
                title TEXT NOT NULL,                -- Passage title
                content TEXT NOT NULL,              -- Full passage text
                sentence_count INTEGER,             -- 문장 수
                
                UNIQUE(passage_id, title)
            )
        """)
        
        # 2. Metadata-Passage 매핑 테이블 (다대다)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata_passage_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_id INTEGER NOT NULL,       -- metadata.id
                passage_id INTEGER NOT NULL,        -- passages.id
                
                FOREIGN KEY (metadata_id) REFERENCES metadata(id),
                FOREIGN KEY (passage_id) REFERENCES passages(id),
                UNIQUE(metadata_id, passage_id)
            )
        """)
        
        # 인덱스
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_passage_title
            ON passages(title)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mapping_metadata
            ON metadata_passage_mapping(metadata_id)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mapping_passage
            ON metadata_passage_mapping(passage_id)
        """)
        
        self.conn.commit()
        print("✓ Passage tables and indexes created")
    
    def load_passages_from_hotpotqa(self, hotpotqa_file: str):
        """
        Load passages from HotpotQA sample file
        
        Args:
            hotpotqa_file: Path to hotpotqa_sample_200.json
        """
        print(f"\nLoading passages from: {hotpotqa_file}")
        
        with open(hotpotqa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} QA entries")
        
        # 모든 passage 수집 (중복 제거)
        passages_dict = {}  # {(qa_id, title): [sentences]}
        
        for qa_entry in data:
            qa_id = qa_entry['_id']
            context = qa_entry['context']
            
            for title, sentences in context:
                key = (qa_id, title)
                if key not in passages_dict:
                    passages_dict[key] = sentences
        
        print(f"✓ Found {len(passages_dict)} unique passages")
        
        # Insert passages
        inserted = 0
        skipped = 0
        
        for (qa_id, title), sentences in passages_dict.items():
            # 문장들을 하나의 텍스트로 합치기
            content = ' '.join(sentences)
            
            try:
                self.cursor.execute("""
                    INSERT INTO passages (passage_id, title, content, sentence_count)
                    VALUES (?, ?, ?, ?)
                """, (qa_id, title, content, len(sentences)))
                inserted += 1
            except sqlite3.IntegrityError:
                skipped += 1
        
        self.conn.commit()
        print(f"✓ Inserted: {inserted}, Skipped: {skipped}")
        
        return inserted, skipped
    
    def create_metadata_passage_mappings(self):
        """
        Create mappings between metadata and passages based on title matching
        """
        print("\nCreating metadata-passage mappings...")
        
        # Get all metadata titles
        self.cursor.execute("SELECT id, title FROM metadata")
        metadata_entries = self.cursor.fetchall()
        
        print(f"Processing {len(metadata_entries)} metadata entries...")
        
        mapped = 0
        not_found = 0
        
        for meta in metadata_entries:
            meta_id = meta['id']
            meta_title = meta['title']
            
            # Find matching passages by title
            self.cursor.execute("""
                SELECT id FROM passages WHERE title = ?
            """, (meta_title,))
            
            matching_passages = self.cursor.fetchall()
            
            if matching_passages:
                # Create mappings
                for passage in matching_passages:
                    try:
                        self.cursor.execute("""
                            INSERT INTO metadata_passage_mapping (metadata_id, passage_id)
                            VALUES (?, ?)
                        """, (meta_id, passage['id']))
                        mapped += 1
                    except sqlite3.IntegrityError:
                        pass  # Already exists
            else:
                not_found += 1
        
        self.conn.commit()
        print(f"✓ Created {mapped} mappings")
        print(f"⚠ {not_found} metadata entries without matching passages")
        
        return mapped, not_found
    
    def get_passages_for_metadata(self, metadata_title: str) -> List[Dict]:
        """
        Get all passages associated with a metadata entry
        
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
    
    def get_metadata_for_passage(self, passage_title: str) -> List[Dict]:
        """
        Get all metadata entries associated with a passage
        
        Args:
            passage_title: Title of the passage
            
        Returns:
            List of metadata entries
        """
        self.cursor.execute("""
            SELECT meta.id, meta.title, meta.type, meta.subtype, meta.metadata_json
            FROM metadata meta
            JOIN metadata_passage_mapping m ON meta.id = m.metadata_id
            JOIN passages p ON m.passage_id = p.id
            WHERE p.title = ?
        """, (passage_title,))
        
        results = []
        for row in self.cursor.fetchall():
            results.append({
                'id': row['id'],
                'title': row['title'],
                'type': row['type'],
                'subtype': row['subtype'],
                'metadata': json.loads(row['metadata_json'])
            })
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get mapping statistics"""
        stats = {}
        
        # Total passages
        self.cursor.execute("SELECT COUNT(*) as count FROM passages")
        stats['total_passages'] = self.cursor.fetchone()['count']
        
        # Total mappings
        self.cursor.execute("SELECT COUNT(*) as count FROM metadata_passage_mapping")
        stats['total_mappings'] = self.cursor.fetchone()['count']
        
        # Metadata with passages
        self.cursor.execute("""
            SELECT COUNT(DISTINCT metadata_id) as count 
            FROM metadata_passage_mapping
        """)
        stats['metadata_with_passages'] = self.cursor.fetchone()['count']
        
        # Passages with metadata
        self.cursor.execute("""
            SELECT COUNT(DISTINCT passage_id) as count 
            FROM metadata_passage_mapping
        """)
        stats['passages_with_metadata'] = self.cursor.fetchone()['count']
        
        # Total metadata
        self.cursor.execute("SELECT COUNT(*) as count FROM metadata")
        stats['total_metadata'] = self.cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 사용 예시
if __name__ == "__main__":
    print("="*80)
    print("Metadata-Passage Mapping Tool")
    print("="*80)
    
    with PassageMapper() as mapper:
        # 1. Create tables
        print("\n[Step 1] Creating passage tables...")
        mapper.create_passage_tables()
        
        # 2. Load passages from HotpotQA
        print("\n[Step 2] Loading passages from HotpotQA...")
        inserted, skipped = mapper.load_passages_from_hotpotqa(
            'HotpotQA/hotpotqa_sample_200.json'
        )
        
        # 3. Create mappings
        print("\n[Step 3] Creating metadata-passage mappings...")
        mapped, not_found = mapper.create_metadata_passage_mappings()
        
        # 4. Statistics
        print("\n[Step 4] Statistics:")
        stats = mapper.get_statistics()
        print(f"  Total passages: {stats['total_passages']}")
        print(f"  Total metadata: {stats['total_metadata']}")
        print(f"  Total mappings: {stats['total_mappings']}")
        print(f"  Metadata with passages: {stats['metadata_with_passages']} / {stats['total_metadata']}")
        print(f"  Passages with metadata: {stats['passages_with_metadata']} / {stats['total_passages']}")
        
        # 5. Test query
        print("\n[Step 5] Test Query - 'The Wolf of Wall Street'")
        passages = mapper.get_passages_for_metadata("The Wolf of Wall Street (2013 film)")
        if passages:
            for p in passages:
                print(f"\n  Passage ID: {p['passage_id']}")
                print(f"  Title: {p['title']}")
                print(f"  Sentences: {p['sentence_count']}")
                print(f"  Content: {p['content'][:200]}...")
        else:
            print("  No passages found")
    
    print("\n" + "="*80)
    print("✓ Mapping complete!")
    print("="*80)
