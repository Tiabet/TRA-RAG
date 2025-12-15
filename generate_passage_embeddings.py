#!/usr/bin/env python3
"""
Generate Passage Embeddings
===========================
Generates embeddings for all unique passages in a dataset (HotpotQA/MuSiQue).
Used for pre-computation to avoid runtime embedding overhead.
"""

# python generate_passage_embeddings.py --data MuSiQue/musique_sample_200.json --output MuSiQue/passage_embeddings.npz

import json
import argparse
import numpy as np
import asyncio
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI

async def generate_embeddings(
    data_path: str,
    output_path: str,
    batch_size: int = 100
):
    load_dotenv()
    
    api_key = os.getenv('ALICE_OPENAI_KEY')
    embed_url = os.getenv('ALICE_EMBED_URL')
    
    if not api_key or not embed_url:
        raise ValueError("Please set ALICE_OPENAI_KEY and ALICE_EMBED_URL in .env")
        
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=embed_url
    )
    
    print(f"Loading passages from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Extract unique passages
    passages: Dict[str, str] = {}
    for item in data:
        # Handle both HotpotQA and MuSiQue formats if possible, 
        # but usually they have 'context' or 'paragraphs'
        # HotpotQA: 'context': [ [title, [sent1, sent2...]], ... ]
        # MuSiQue: 'paragraphs': [ {'title':..., 'paragraph_text':...}, ... ] 
        # Let's assume HotpotQA format based on previous context, or check structure.
        
        if 'context' in item:
            for title, sentences in item['context']:
                if title not in passages:
                    passages[title] = "".join(sentences).strip()
        elif 'paragraphs' in item:
             for p in item['paragraphs']:
                 title = p['title']
                 text = p['paragraph_text']
                 if title not in passages:
                     passages[title] = text
                     
    titles = list(passages.keys())
    texts = list(passages.values())
    print(f"Found {len(titles)} unique passages.")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Replace newlines
        batch = [t.replace("\n", " ") for t in batch]
        
        try:
            response = await client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [d.embedding for d in response.data]
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 1000 == 0:
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Retry or skip? For now, raise
            raise e
            
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings (important for cosine similarity)
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    embeddings_normalized = embeddings_np / (norms + 1e-8)
    
    print(f"Saving to {output_path}...")
    np.savez(
        output_path,
        titles=titles,
        embeddings=embeddings_normalized
    )
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input JSON (HotpotQA/MuSiQue)")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz file")
    args = parser.parse_args()
    
    asyncio.run(generate_embeddings(args.data, args.output))
