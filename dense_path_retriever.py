#!/usr/bin/env python3
"""
Dense Only Path Retriever
==========================
Uses only Dense Embedding (semantic) search for metadata paths.
For ablation study.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI


class DensePathRetriever:
    """Dense embedding-only retriever for ablation study."""
    
    def __init__(
        self,
        embeddings_path: str = 'HotpotQA/path_embeddings.npz'
    ):
        """
        Args:
            embeddings_path: Path to embeddings .npz file
        """
        load_dotenv()
        
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
        
        # OpenAI client for query embedding
        self.api_key = os.getenv('ALICE_OPENAI_KEY')
        self.embed_url = os.getenv('ALICE_EMBED_URL')
        
        if self.api_key and self.embed_url:
            self.embed_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.embed_url
            )
        else:
            raise ValueError("Embedding client not configured. Check ALICE_OPENAI_KEY and ALICE_EMBED_URL.")
        
        print(f"✓ Loaded {len(self.embeddings)} embeddings")
        print(f"✓ Dense Only Mode")
    
    async def embed_query(self, query: str) -> np.ndarray:
        """Get embedding for query."""
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
        bm25_candidates: int = 100,  # Ignored in Dense-only mode
        dense_candidates: int = 100
    ) -> List[Dict]:
        """
        Dense-only search (same interface as hybrid for compatibility).
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_candidates: Ignored
            dense_candidates: Number of dense candidates
            
        Returns:
            List of result dicts with path metadata and scores
        """
        # Get dense results
        dense_results = await self.search_dense(query, max(dense_candidates, top_k))
        
        # Z-score normalization
        results = []
        if dense_results:
            scores = [s for _, s in dense_results]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            if std_score == 0:
                std_score = 1.0

            for idx, score in dense_results[:top_k]:
                normalized_score = (score - mean_score) / std_score
                results.append({
                    'index': idx,
                    'title': str(self.titles[idx]),
                    'key_path': str(self.key_paths[idx]),
                    'value': str(self.values[idx]),
                    'score': normalized_score,
                    'bm25_score': 0.0,
                    'dense_score': normalized_score
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


async def test_dense_search():
    """Test Dense-only search functionality."""
    print("="*80)
    print("Testing Dense Only Path Retriever")
    print("="*80)
    
    retriever = DensePathRetriever()
    
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
    asyncio.run(test_dense_search())
