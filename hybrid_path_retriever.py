#!/usr/bin/env python3
"""
Hybrid Path Retriever
======================
Combines BM25 (sparse) and Dense Embedding (semantic) search for metadata paths.

Search Flow:
1. BM25 search on paths
2. Dense embedding search on paths
3. Combine scores (weighted fusion)
4. Return top-k paths with metadata
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

import bm25s
import Stemmer


class HybridPathRetriever:
    """Hybrid retriever combining BM25 and dense embeddings."""
    
    def __init__(
        self,
        bm25_index_path: str = 'HotpotQA/bm25_index',
        embeddings_path: str = 'HotpotQA/path_embeddings.npz',
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5
    ):
        """
        Args:
            bm25_index_path: Path to BM25 index directory
            embeddings_path: Path to embeddings .npz file
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense scores (0-1)
        """
        load_dotenv()
        
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Load BM25 index
        print(f"Loading BM25 index from: {bm25_index_path}")
        self.bm25 = bm25s.BM25.load(bm25_index_path)
        
        # Load BM25 metadata
        with open(Path(bm25_index_path) / 'metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load dense embeddings
        print(f"Loading embeddings from: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.titles = data['titles']
        self.key_paths = data['key_paths']
        self.values = data['values']
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / (norms + 1e-8)
        
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
        
        # OpenAI client for query embedding
        self.api_key = os.getenv('ALICE_OPENAI_KEY')
        self.embed_url = os.getenv('ALICE_EMBED_URL')
        
        if self.api_key and self.embed_url:
            self.embed_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.embed_url
            )
        else:
            self.embed_client = None
            print("Warning: Embedding client not configured. Dense search disabled.")
        
        print(f"✓ Loaded {len(self.metadata)} paths")
        print(f"✓ BM25 weight: {bm25_weight}, Dense weight: {dense_weight}")
        
        # Build title to indices map for fast lookup
        self.title_to_indices = {}
        print("Building title index...")
        for idx, title in enumerate(self.titles):
            if title not in self.title_to_indices:
                self.title_to_indices[title] = []
            self.title_to_indices[title].append(idx)
            
    def get_indices_for_title(self, title: str) -> List[int]:
        """Get all path indices for a given title."""
        return self.title_to_indices.get(title, [])

    async def score_candidates_rrf(self, query: str, candidate_indices: List[int], top_k: int = 10) -> List[Dict]:
        """
        Score specific candidate paths using RRF of BM25 and Dense scores.
        
        Args:
            query: The query string
            candidate_indices: List of indices to score
            top_k: Number of results to return
            
        Returns:
            List of dicts with scored results
        """
        if not candidate_indices:
            return []
            
        # 1. Dense Scores
        # Embed query
        query_embedding = await self.embed_query(query)
        
        # Get candidate embeddings
        candidate_embeddings = self.embeddings_normalized[candidate_indices]
        
        # Compute Cosine Similarity (Dot product of normalized vectors)
        dense_raw_scores = np.dot(candidate_embeddings, query_embedding)
        
        # 2. BM25 Scores
        query_tokens = self.preprocess_query(query)
        
        bm25_raw_scores = np.zeros(len(candidate_indices))
        
        if query_tokens:
            # We retrieve top-k where k is large enough to likely include our candidates
            # or we can try to retrieve all. 
            # bm25s.retrieve with k=len(self.titles) gets all non-zero scores?
            # Actually, bm25s returns top-k. 
            # Let's use k=50000 as a safe upper bound for now, but capped at corpus size.
            bm25_k = min(50000, len(self.titles))
            results, scores = self.bm25.retrieve([query_tokens], k=bm25_k)
            
            # Create a map for fast lookup
            # results[0] are indices, scores[0] are scores
            bm25_map = dict(zip(results[0], scores[0]))
            
            for i, idx in enumerate(candidate_indices):
                bm25_raw_scores[i] = bm25_map.get(idx, 0.0)
        
        # 3. RRF Calculation
        rrf_k = 60
        
        # Get ranks for Dense
        # argsort gives indices that would sort the array. [::-1] reverses to descending.
        # We need the rank of each item in the original array.
        dense_sort_indices = np.argsort(dense_raw_scores)[::-1]
        dense_ranks = np.zeros(len(candidate_indices), dtype=int)
        for rank, i in enumerate(dense_sort_indices):
            dense_ranks[i] = rank
            
        # Get ranks for BM25
        bm25_sort_indices = np.argsort(bm25_raw_scores)[::-1]
        bm25_ranks = np.zeros(len(candidate_indices), dtype=int)
        for rank, i in enumerate(bm25_sort_indices):
            bm25_ranks[i] = rank
            
        combined_scores = []
        scored_candidates = []
        
        for i, idx in enumerate(candidate_indices):
            # RRF Score
            dense_rrf = 1.0 / (rrf_k + dense_ranks[i])
            bm25_rrf = 1.0 / (rrf_k + bm25_ranks[i])
            
            combined = (self.dense_weight * dense_rrf) + (self.bm25_weight * bm25_rrf)
            
            scored_candidates.append({
                'index': idx,
                'title': str(self.titles[idx]),
                'key_path': str(self.key_paths[idx]),
                'value': str(self.values[idx]),
                'score': float(combined),
                'dense_score': float(dense_raw_scores[i]),
                'bm25_score': float(bm25_raw_scores[i]),
                'dense_rank': int(dense_ranks[i]),
                'bm25_rank': int(bm25_ranks[i])
            })
            
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_candidates[:top_k]

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
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Get embedding for query."""
        if not self.embed_client:
            raise ValueError("Embedding client not configured")
        
        response = await self.embed_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
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
    
    async def search_dense(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Dense embedding search.
        
        Returns:
            List of (index, score) tuples
        """
        query_embedding = await self.embed_query(query)
        
        # Cosine similarity (dot product since normalized)
        similarities = np.dot(self.embeddings_normalized, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    async def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        bm25_candidates: int = 50,
        dense_candidates: int = 50
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and dense.
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: Number of BM25 candidates
            dense_candidates: Number of dense candidates
            
        Returns:
            List of result dicts with path metadata and scores
        """
        # Get BM25 results
        bm25_results = self.search_bm25(query, bm25_candidates)
        
        # Get dense results
        dense_results = await self.search_dense(query, dense_candidates)
        
        # RRF (Reciprocal Rank Fusion)
        # Score = 1 / (k + rank)
        rrf_k = 60
        
        all_indices = set()
        bm25_ranks = {}
        if bm25_results:
            for rank, (idx, _) in enumerate(bm25_results):
                bm25_ranks[idx] = rank
                all_indices.add(idx)
        
        dense_ranks = {}
        if dense_results:
            for rank, (idx, _) in enumerate(dense_results):
                dense_ranks[idx] = rank
                all_indices.add(idx)
        
        combined_scores = []
        for idx in all_indices:
            # Calculate RRF scores
            bm25_score = 0.0
            if idx in bm25_ranks:
                bm25_score = 1.0 / (rrf_k + bm25_ranks[idx])
            
            dense_score = 0.0
            if idx in dense_ranks:
                dense_score = 1.0 / (rrf_k + dense_ranks[idx])
            
            # Weighted RRF
            combined = (self.bm25_weight * bm25_score) + (self.dense_weight * dense_score)
            combined_scores.append((idx, combined, bm25_score, dense_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for idx, combined, bm25_s, dense_s in combined_scores[:top_k]:
            results.append({
                'index': idx,
                'title': str(self.titles[idx]),
                'key_path': str(self.key_paths[idx]),
                'value': str(self.values[idx]),
                'score': combined,
                'bm25_score': bm25_s,
                'dense_score': dense_s
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


async def test_hybrid_search():
    """Test hybrid search functionality."""
    print("="*80)
    print("Testing Hybrid Path Retriever")
    print("="*80)
    
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6
    )
    
    test_queries = [
        "Who directed The Wolf of Wall Street?",
        "What is the nationality of The Wolf of Wall Street film?",
        "Leonardo DiCaprio actor movies",
        "Argentina education system",
        "airport in Myanmar"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = await retriever.search_hybrid(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"\n{i}. [Score: {r['score']:.3f}] (BM25: {r['bm25_score']:.3f}, Dense: {r['dense_score']:.3f})")
            print(f"   Title: {r['title']}")
            print(f"   Path: {r['key_path']}")
            print(f"   Value: {r['value'][:80]}..." if len(r['value']) > 80 else f"   Value: {r['value']}")


if __name__ == "__main__":
    asyncio.run(test_hybrid_search())
