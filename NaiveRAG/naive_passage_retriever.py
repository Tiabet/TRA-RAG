import os
import json
import numpy as np
import asyncio
from typing import List, Dict, Tuple, Optional
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
        self.passage_records = self._load_passage_records()
        self.doc_ids = [r["doc_id"] for r in self.passage_records]
        self.titles = [r["title"] for r in self.passage_records]
        self.texts = [r["text"] for r in self.passage_records]
        
        # Load or generate embeddings
        self.embeddings = self._load_or_generate_embeddings()
        
    def _load_passage_records(self) -> List[Dict[str, str]]:
        """Load all unique passages from the dataset.

        Supported formats:
          - HotpotQA corpus_idx: items[].context[] is dict with {title, sentences, corpus_idx}
          - MuSiQue corpus_idx: items[].paragraphs[] is dict with {title, paragraph_text, corpus_idx}
        """
        print(f"Loading passages from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build unique-by-doc_id (corpus_idx) passages.
        by_doc_id: Dict[str, Dict[str, str]] = {}
        for item in data or []:
            if not isinstance(item, dict):
                continue

            # MuSiQue-style
            paragraphs = item.get('paragraphs')
            if isinstance(paragraphs, list):
                for p in paragraphs:
                    if not isinstance(p, dict):
                        continue
                    corpus_idx = p.get('corpus_idx')
                    title = p.get('title')
                    text = p.get('paragraph_text') or p.get('text')
                    if corpus_idx is None or title is None or text is None:
                        continue
                    doc_id = str(corpus_idx)
                    if doc_id in by_doc_id:
                        continue
                    by_doc_id[doc_id] = {
                        'doc_id': doc_id,
                        'title': str(title),
                        'text': f"{title}\n{text}",
                    }
                continue

            # HotpotQA-style
            context = item.get('context')
            if isinstance(context, list):
                for c in context:
                    if not isinstance(c, dict):
                        # legacy list-based context is no longer expected here
                        continue
                    corpus_idx = c.get('corpus_idx')
                    title = c.get('title')
                    sentences = c.get('sentences')
                    if corpus_idx is None or title is None or sentences is None:
                        continue
                    if not isinstance(sentences, list):
                        continue
                    doc_id = str(corpus_idx)
                    if doc_id in by_doc_id:
                        continue
                    content = "".join([str(s) for s in sentences])
                    by_doc_id[doc_id] = {
                        'doc_id': doc_id,
                        'title': str(title),
                        'text': f"{title}\n{content}",
                    }

        records = list(by_doc_id.values())
        # Stable order: doc_id numeric ascending when possible
        def _sort_key(r: Dict[str, str]):
            s = r.get('doc_id', '')
            return (0, int(s)) if s.isdigit() else (1, s)

        records.sort(key=_sort_key)
        print(f"Loaded {len(records)} unique passages.")
        return records

    def _load_or_generate_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from cache if compatible; otherwise return None.

        Embedding generation is handled in `initialize()`.
        """
        if not os.path.exists(self.embedding_cache_path):
            return None

        print(f"Loading embeddings from {self.embedding_cache_path}...")
        data = np.load(self.embedding_cache_path)

        # New caches store doc_ids + titles; older caches may store only titles.
        cached_titles = list(data.get('titles', []))
        cached_doc_ids = list(data.get('doc_ids', []))

        if cached_doc_ids:
            if len(cached_doc_ids) == len(self.doc_ids) and all(str(a) == str(b) for a, b in zip(cached_doc_ids, self.doc_ids)):
                return data['embeddings']
            print("Cache mismatch (doc_ids)! Regenerating...")
            return None

        # Back-compat: accept title-only caches only when titles exactly match.
        if cached_titles and len(cached_titles) == len(self.titles) and all(str(a) == str(b) for a, b in zip(cached_titles, self.titles)):
            return data['embeddings']
        print("Cache mismatch (titles)! Regenerating...")
        return None

    async def initialize(self):
        """Async initialization to handle embedding generation."""
        if self.embeddings is not None:
            return

        # Try loading again (in case init-time load was skipped)
        loaded = self._load_or_generate_embeddings()
        if loaded is not None:
            self.embeddings = loaded
            return

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
        np.savez(self.embedding_cache_path, embeddings=self.embeddings, titles=self.titles, doc_ids=self.doc_ids)
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
                'doc_id': self.doc_ids[idx],
                'title': self.titles[idx],
                'text': self.texts[idx],
                'score': float(scores[idx])
            })
            
        return results
