#!/usr/bin/env python3
"""
BM25 Only Path Retriever
=========================
Uses only BM25 (sparse) search for metadata paths.
For ablation study.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import asyncio

import bm25s
import Stemmer


class BM25PathRetriever:
    """BM25-only retriever for ablation study."""
    
    def __init__(
        self,
        bm25_index_path: str = 'HotpotQA/bm25_index',
        embeddings_path: str = 'HotpotQA/path_embeddings.npz'
    ):
        """
        Args:
            bm25_index_path: Path to BM25 index directory
            embeddings_path: Path to embeddings .npz file (for titles/metadata)
        """
        # Load BM25 index
        print(f"Loading BM25 index from: {bm25_index_path}")
        self.bm25 = bm25s.BM25.load(bm25_index_path)
        
        # Load BM25 metadata
        with open(Path(bm25_index_path) / 'metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load path data (for titles, key_paths, values)
        print(f"Loading path data from: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        self.titles = data['titles']
        self.key_paths = data['key_paths']
        self.values = data['values']
        self.doc_ids = data['doc_ids'] if 'doc_ids' in data.files else None
        self.source_titles = data['source_titles'] if 'source_titles' in data.files else None
        self.entity_titles = data['entity_titles'] if 'entity_titles' in data.files else None
        
        # Stemmer for BM25 preprocessing
        self.stemmer = Stemmer.Stemmer('english')
        
        # Stopwords
        self.stopwords = {
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
        
        print(f"[OK] Loaded {len(self.metadata)} paths")
        print(f"[OK] BM25 Only Mode")

    @staticmethod
    def _opt_field(arr, idx):
        if arr is None:
            return None
        v = arr[idx]
        return None if v is None else str(v)
    
    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query for BM25."""
        import re
        
        # Lowercase
        text = query.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        # Stemming
        tokens = self.stemmer.stemWords(tokens)
        
        return tokens
    
    def search_bm25(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        BM25 search.
        
        Returns:
            List of (index, score) tuples
        """
        query_tokens = self.preprocess_query(query)
        
        if not query_tokens:
            return []
        
        results, scores = self.bm25.retrieve([query_tokens], k=top_k)
        
        return list(zip(results[0], scores[0]))
    
    async def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 50,
        dense_candidates: int = 50  # Ignored in BM25-only mode
    ) -> List[Dict]:
        """
        BM25-only search (same interface as hybrid for compatibility).
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: Number of BM25 candidates
            dense_candidates: Ignored
            
        Returns:
            List of result dicts with path metadata and scores
        """
        # Get BM25 results
        bm25_results = self.search_bm25(query, max(bm25_candidates, top_k))
        
        # Normalize scores to [0, 1] range
        if bm25_results:
            max_score = max(s for _, s in bm25_results)
            min_score = min(s for _, s in bm25_results)
            range_score = max_score - min_score if max_score != min_score else 1
        
        # Build results
        results = []
        for idx, score in bm25_results[:top_k]:
            normalized_score = (score - min_score) / range_score if bm25_results else 0
            results.append({
                'index': idx,
                'title': str(self.titles[idx]),
                'doc_id': self._opt_field(self.doc_ids, idx),
                'source_title': self._opt_field(self.source_titles, idx),
                'entity_title': self._opt_field(self.entity_titles, idx),
                'key_path': str(self.key_paths[idx]),
                'value': str(self.values[idx]),
                'score': normalized_score,
                'bm25_score': normalized_score,
                'dense_score': 0.0
            })
        
        return results
    
    def get_paths_by_title(self, title: str) -> List[Dict]:
        """Get all paths for a given title."""
        results = []
        for i, t in enumerate(self.titles):
            if str(t) == title:
                results.append({
                    'index': i,
                    'title': str(self.titles[i]),
                    'key_path': str(self.key_paths[i]),
                    'value': str(self.values[i])
                })
        return results


async def test_bm25_search():
    """Test BM25-only search functionality."""
    print("="*80)
    print("Testing BM25 Only Path Retriever")
    print("="*80)
    
    retriever = BM25PathRetriever()
    
    test_queries = [
        "Who directed The Wolf of Wall Street?",
        "Leonardo DiCaprio actor movies",
        "Argentina education system"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = await retriever.search_hybrid(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [Score: {r['score']:.3f}]")
            print(f"   Title: {r['title']}")
            print(f"   Path: {r['key_path']}")
            print(f"   Value: {r['value'][:80]}..." if len(r['value']) > 80 else f"   Value: {r['value']}")


if __name__ == "__main__":
    asyncio.run(test_bm25_search())
