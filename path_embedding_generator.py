#!/usr/bin/env python3
"""
Path Embedding Generator
========================
Generates dense embeddings for metadata paths using text-embedding-3-small.

Uses Elice API (OpenAI compatible) for embedding generation.
"""

import json
import os
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI
import time


class PathEmbeddingGenerator:
    """Generate embeddings for metadata paths using OpenAI API."""
    
    def __init__(self, batch_size: int = 200, max_concurrency: int = 5):
        """
        Args:
            batch_size: Number of texts to embed in one API call
            max_concurrency: Maximum number of concurrent API calls
        """
        load_dotenv()
        
        self.api_key = os.getenv('ALICE_OPENAI_KEY')
        # Use EMBED_URL for embeddings
        self.base_url = os.getenv('ALICE_EMBED_URL')
        
        if not self.api_key:
            raise ValueError("ALICE_OPENAI_KEY not found in environment")
        if not self.base_url:
            raise ValueError("ALICE_EMBED_URL not found in environment")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.model = "text-embedding-3-small"
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.embedding_dim = 1536  # text-embedding-3-small dimension
    
    async def _embed_batch(self, batch_idx: int, batch: List[str]) -> tuple:
        """Embed a single batch and return (batch_idx, embeddings)."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            return (batch_idx, embeddings, None)
        except Exception as e:
            return (batch_idx, None, e)
    
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts with concurrent batch processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Prepare batches
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append((i // self.batch_size, texts[i:i + self.batch_size]))
        
        print(f"  Total batches: {len(batches)}, Concurrency: {self.max_concurrency}")
        
        # Results storage (ordered by batch_idx)
        results = [None] * len(batches)
        failed_batches = []
        
        # Process with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def process_with_semaphore(batch_idx, batch):
            async with semaphore:
                return await self._embed_batch(batch_idx, batch)
        
        # Create all tasks
        tasks = [process_with_semaphore(idx, batch) for idx, batch in batches]
        
        # Process with progress tracking
        completed = 0
        for coro in asyncio.as_completed(tasks):
            batch_idx, embeddings, error = await coro
            completed += 1
            
            if error:
                print(f"  Batch {batch_idx} failed: {error}")
                failed_batches.append((batch_idx, batches[batch_idx][1]))
            else:
                results[batch_idx] = embeddings
            
            # Progress every 10%
            if completed % max(1, len(batches) // 10) == 0 or completed == len(batches):
                print(f"  Progress: {completed}/{len(batches)} batches ({completed*100//len(batches)}%)")
        
        # Retry failed batches one by one
        if failed_batches:
            print(f"  Retrying {len(failed_batches)} failed batches individually...")
            for batch_idx, batch in failed_batches:
                batch_embeddings = []
                for text in batch:
                    try:
                        response = await self.client.embeddings.create(
                            model=self.model,
                            input=[text]
                        )
                        batch_embeddings.append(response.data[0].embedding)
                    except Exception as e:
                        print(f"  Single text failed: {e}")
                        batch_embeddings.append([0.0] * self.embedding_dim)
                results[batch_idx] = batch_embeddings
        
        # Flatten results
        all_embeddings = []
        for batch_embeddings in results:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    async def generate_embeddings(
        self,
        input_path: str = 'HotpotQA/embedding_texts.json',
        output_path: str = 'HotpotQA/path_embeddings.npz'
    ):
        """
        Generate embeddings for all paths.
        
        Args:
            input_path: Path to embedding_texts.json
            output_path: Path to save embeddings (.npz format)
        """
        print("="*80)
        print("Generating Path Embeddings")
        print("="*80)
        
        # Load texts
        print(f"\n1. Loading texts from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   ✓ Loaded {len(data)} entries")
        
        # Extract text field
        texts = [entry['text'] for entry in data]
        
        # Generate embeddings
        print(f"\n2. Generating embeddings (model: {self.model})...")
        print(f"   Batch size: {self.batch_size}")
        
        start_time = time.time()
        embeddings = await self.embed_texts(texts)
        elapsed = time.time() - start_time
        
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   ✓ Shape: {embeddings.shape}")
        print(f"   ✓ Time: {elapsed:.1f}s ({len(texts)/elapsed:.1f} texts/sec)")
        
        # Save embeddings
        print(f"\n3. Saving embeddings to: {output_path}")
        
        # Save as compressed numpy
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            # Also save metadata for reference
            titles=np.array([e['title'] for e in data]),
            key_paths=np.array([e['key_path'] for e in data]),
            values=np.array([e['value'] for e in data])
        )
        
        # Also save a JSON index file for easy lookup
        index_path = output_path.replace('.npz', '_index.json')
        index_data = [
            {
                'idx': i,
                'title': data[i]['title'],
                'key_path': data[i]['key_path'],
                'value': data[i]['value'][:200] if len(data[i]['value']) > 200 else data[i]['value']
            }
            for i in range(len(data))
        ]
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False)
        
        print(f"   ✓ Saved embeddings and index")
        
        # Statistics
        print(f"\n[Statistics]")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\n" + "="*80)
        print("✓ Path embeddings generated successfully!")
        print("="*80)
        
        return embeddings


async def test_embedding():
    """Test embedding generation with a few samples."""
    print("="*80)
    print("Testing Embedding Generation")
    print("="*80)
    
    generator = PathEmbeddingGenerator(batch_size=10)
    
    test_texts = [
        "The director of The Wolf of Wall Street (2013 film) is Martin Scorsese",
        "The nationality of The Wolf of Wall Street (2013 film) is American",
        "The cast of The Wolf of Wall Street (2013 film) is Leonardo DiCaprio",
        "The country of Roissy-en-France is France",
        "The description of Mandalay International Airport is largest and most modern airport in Myanmar"
    ]
    
    print(f"\nTest texts ({len(test_texts)}):")
    for i, t in enumerate(test_texts, 1):
        print(f"  {i}. {t[:70]}...")
    
    print(f"\nGenerating embeddings...")
    embeddings = await generator.embed_texts(test_texts)
    
    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    print(f"✓ First embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test similarity
    print(f"\n[Similarity Test]")
    from numpy.linalg import norm
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    # Wolf of Wall Street texts should be similar
    sim_01 = cosine_sim(embeddings[0], embeddings[1])
    sim_02 = cosine_sim(embeddings[0], embeddings[2])
    sim_03 = cosine_sim(embeddings[0], embeddings[3])  # Different movie
    
    print(f"  Similarity (Wolf Street director vs nationality): {sim_01:.3f}")
    print(f"  Similarity (Wolf Street director vs cast): {sim_02:.3f}")
    print(f"  Similarity (Wolf Street vs France): {sim_03:.3f}")
    
    print("\n✓ Test completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_embedding())
    else:
        generator = PathEmbeddingGenerator(batch_size=200, max_concurrency=5)
        asyncio.run(generator.generate_embeddings(
            input_path='HotpotQA/embedding_texts.json',
            output_path='HotpotQA/path_embeddings.npz'
        ))
