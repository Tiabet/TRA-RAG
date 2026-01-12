#!/usr/bin/env python3
"""
BM25 Indexer for Metadata
=========================
Creates BM25 index from metadata using bm25s library.

Text format: title + key_path + value (space-separated)
Preprocessing: lowercase, stopword removal, punctuation removal, stemming
"""

import json
import re
import string
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import bm25s
import Stemmer  # PyStemmer


# English stopwords (standard list)
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'
}


class BM25Indexer:
    """BM25 indexer for metadata search."""
    
    def __init__(self, use_stemming: bool = True):
        """
        Args:
            use_stemming: Whether to apply stemming
        """
        self.use_stemming = use_stemming
        self.stemmer = Stemmer.Stemmer('english') if use_stemming else None
        self.index = None
        self.corpus = []  # Original texts
        self.metadata = []  # Original metadata entries
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for BM25.
        
        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Tokenize
        4. Remove stopwords
        5. Stemming (optional)
        
        Args:
            text: Raw text
            
        Returns:
            List of processed tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        # Stemming
        if self.use_stemming and self.stemmer:
            tokens = self.stemmer.stemWords(tokens)
        
        return tokens
    
    def build_bm25_text(self, entry: Dict) -> str:
        """
        Build BM25 text from embedding entry.
        
        Format: title + key_path + value
        Example: "The Wolf of Wall Street (2013 film) director Martin Scorsese"
        
        Args:
            entry: Dict with 'title', 'key_path', 'value'
            
        Returns:
            Combined text string
        """
        title = entry.get('title', '')
        key_path = entry.get('key_path', '').replace('.', ' ')  # director.name -> director name
        value = entry.get('value', '')
        
        # Combine: title + key + value
        return f"{title} {key_path} {value}"

    def strip_stopwords_from_text(self, text: str) -> str:
        """Remove stopwords from raw text (best-effort, English).

        Note: We still apply stopword removal during tokenization in `preprocess()`.
        This function exists to satisfy the requirement of stripping stopwords
        from embedding_texts.json's `text` field before BM25 indexing.
        """
        if not text:
            return ""

        lowered = text.lower()
        lowered = re.sub(r"[^\w\s]", " ", lowered)
        tokens = lowered.split()
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return " ".join(tokens)
    
    def build_index(
        self,
        embedding_texts_path: str = 'HotpotQA/embedding_texts.json',
        index_save_path: str = 'HotpotQA/bm25_index',
        use_embedding_text_field: bool = False,
        strip_stopwords_in_embedding_text: bool = False
    ):
        """
        Build BM25 index from embedding texts.
        
        Args:
            embedding_texts_path: Path to embedding_texts.json
            index_save_path: Directory to save index
        """
        print("="*80)
        print("Building BM25 Index")
        print("="*80)
        
        # Load embedding texts
        print(f"\n1. Loading data from: {embedding_texts_path}")
        with open(embedding_texts_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"   [OK] Loaded {len(self.metadata)} entries")

        if not self.metadata:
            raise ValueError(
                f"No entries found in {embedding_texts_path}. "
                "Upstream metadata->DB conversion or embedding-text generation likely produced an empty output."
            )
        
        # Build corpus
        print(f"\n2. Building corpus...")
        self.corpus = []
        for entry in self.metadata:
            if use_embedding_text_field and isinstance(entry, dict) and 'text' in entry:
                text = entry.get('text', '')
                if strip_stopwords_in_embedding_text:
                    text = self.strip_stopwords_from_text(text)
            else:
                text = self.build_bm25_text(entry)
            self.corpus.append(text)
        print(f"   [OK] Built {len(self.corpus)} documents")
        
        # Tokenize corpus
        print(f"\n3. Tokenizing corpus...")
        corpus_tokens = [self.preprocess(doc) for doc in self.corpus]
        if not corpus_tokens:
            raise ValueError(
                "BM25 corpus tokenization produced 0 documents. "
                "Check embedding_texts.json generation and filtering logic."
            )
        avg_tokens = sum(len(t) for t in corpus_tokens) / len(corpus_tokens)
        print(f"   [OK] Tokenized (avg tokens/doc: {avg_tokens:.1f})")
        
        # Build BM25 index
        print(f"\n4. Building BM25 index...")
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        print(f"   [OK] Index built")
        
        # Save index
        print(f"\n5. Saving index to: {index_save_path}")
        Path(index_save_path).mkdir(parents=True, exist_ok=True)
        self.index.save(index_save_path)
        
        # Save metadata separately (for retrieval)
        metadata_path = Path(index_save_path) / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"   [OK] Saved index and metadata")
        
        # Statistics
        print(f"\n[Statistics]")
        print(f"  Total documents: {len(self.corpus)}")
        print(f"  Unique titles: {len(set((e.get('title') or '') for e in self.metadata if isinstance(e, dict)))}")
        print(f"  Index location: {index_save_path}")
        
        print("\n" + "="*80)
        print("[OK] BM25 Index built successfully!")
        print("="*80)
    
    def load_index(self, index_path: str = 'HotpotQA/bm25_index'):
        """Load existing BM25 index."""
        print(f"Loading BM25 index from: {index_path}")
        
        self.index = bm25s.BM25.load(index_path)
        
        # Load metadata
        metadata_path = Path(index_path) / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"[OK] Loaded index with {len(self.metadata)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search BM25 index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with metadata and scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Preprocess query
        query_tokens = self.preprocess(query)
        
        # Search
        results, scores = self.index.retrieve([query_tokens], k=top_k)
        
        # Format results
        output = []
        for idx, score in zip(results[0], scores[0]):
            if idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry['score'] = float(score)
                output.append(entry)
        
        return output
    
    def search_by_title(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Search and group results by title.
        
        Args:
            query: Search query
            top_k: Number of results per title
            
        Returns:
            Dict mapping title -> list of matching entries
        """
        # Get more results to ensure coverage
        results = self.search(query, top_k=top_k * 5)
        
        # Group by title
        by_title = {}
        for r in results:
            title = r['title']
            if title not in by_title:
                by_title[title] = []
            if len(by_title[title]) < top_k:
                by_title[title].append(r)
        
        return by_title


def test_search():
    """Test BM25 search functionality."""
    print("="*80)
    print("Testing BM25 Search")
    print("="*80)
    
    indexer = BM25Indexer()
    indexer.load_index('HotpotQA/bm25_index')
    
    # Test queries
    test_queries = [
        "Martin Scorsese director",
        "Leonardo DiCaprio actor",
        "American film 2013",
        "Argentina education",
        "airport Myanmar"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = indexer.search(query, top_k=5)
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [Score: {r['score']:.3f}]")
            print(f"   Title: {r['title']}")
            print(f"   Key: {r['key_path']}")
            print(f"   Value: {r['value'][:80]}..." if len(r['value']) > 80 else f"   Value: {r['value']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_search()
    else:
        # Build index
        indexer = BM25Indexer(use_stemming=True)
        indexer.build_index(
            embedding_texts_path='HotpotQA/embedding_texts.json',
            index_save_path='HotpotQA/bm25_index'
        )
