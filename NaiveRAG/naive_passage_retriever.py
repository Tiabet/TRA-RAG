import os
import json
import numpy as np
import asyncio
from typing import List, Dict, Tuple
from openai import AsyncOpenAI

class NaivePassageRetriever:
    def __init__(
        self,
        client: AsyncOpenAI,
        data_path: str,
        embedding_cache_path: str,
        model: str = "text-embedding-3-small"
    ):
        self.client = client
        self.data_path = data_path
        self.embedding_cache_path = embedding_cache_path
        self.model = model
        
        # Load passages
        self.passages = self._load_passages()
        self.titles = list(self.passages.keys())
        self.texts = list(self.passages.values())
        
        # Load or generate embeddings
        self.embeddings = self._load_or_generate_embeddings()
        
    def _load_passages(self) -> Dict[str, str]:
        """Load all unique passages from the dataset."""
        print(f"Loading passages from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        passages = {}
        for item in data:
            for title, sentences in item.get('context', []):
                if title not in passages:
                    # Format: Title + Content
                    content = "".join(sentences)
                    passages[title] = f"{title}\n{content}"
        print(f"Loaded {len(passages)} unique passages.")
        return passages

    def _load_or_generate_embeddings(self) -> np.ndarray:
        """Load embeddings from cache or generate them."""
        if os.path.exists(self.embedding_cache_path):
            print(f"Loading embeddings from {self.embedding_cache_path}...")
            data = np.load(self.embedding_cache_path)
            # Verify consistency
            if len(data['titles']) != len(self.titles):
                print("Cache mismatch! Regenerating...")
            else:
                return data['embeddings']
        
        # Generate embeddings
        print("Generating embeddings (this may take a while)...")
        # We need to run async generation in a sync method, or change init to async factory
        # For simplicity, we'll use asyncio.run if no loop is running, or just assume the caller handles it?
        # Actually, since __init__ is sync, we should probably do this lazily or use a helper.
        # But for a script, we can just run it here.
        
        try:
            loop = asyncio.get_running_loop()
            # If we are in a loop, we can't use run_until_complete. 
            # But this class is likely initialized inside an async main.
            # So we should probably make an async initialize method.
            # For now, let's assume we can block or use a separate runner if needed.
            # But wait, calling async from sync init is bad.
            # Let's make a static factory or just run it synchronously using a fresh loop if possible, 
            # but since we are likely in an async main, we should expose an async setup method.
            pass
        except RuntimeError:
            pass

        # To keep it simple for the user scripts, let's do the generation in a separate async method `initialize`.
        return None 

    async def initialize(self):
        """Async initialization to handle embedding generation."""
        if self.embeddings is not None:
            return

        if os.path.exists(self.embedding_cache_path):
            print(f"Loading embeddings from {self.embedding_cache_path}...")
            data = np.load(self.embedding_cache_path)
            if len(data['titles']) == len(self.titles):
                self.embeddings = data['embeddings']
                return
            print("Cache mismatch! Regenerating...")

        print(f"Generating embeddings for {len(self.texts)} passages...")
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(self.texts), batch_size):
            batch = self.texts[i:i + batch_size]
            # Replace newlines to avoid issues with some embedding models, though 3-small is robust
            batch = [t.replace("\n", " ") for t in batch]
            
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            # Ensure order is preserved
            batch_embeddings = [d.embedding for d in response.data]
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(self.texts))}/{len(self.texts)}")
            
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        # Save to cache
        np.savez(self.embedding_cache_path, embeddings=self.embeddings, titles=self.titles)
        print(f"Saved embeddings to {self.embedding_cache_path}")

    async def search(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k passages for a query."""
        if self.embeddings is None:
            await self.initialize()
            
        # Embed query
        response = await self.client.embeddings.create(
            input=query.replace("\n", " "),
            model=self.model
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Cosine similarity
        # Normalize embeddings if not already (OpenAI embeddings are usually normalized, but good to be safe)
        # Assuming OpenAI embeddings are normalized.
        scores = np.dot(self.embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'title': self.titles[idx],
                'text': self.texts[idx],
                'score': float(scores[idx])
            })
            
        return results
