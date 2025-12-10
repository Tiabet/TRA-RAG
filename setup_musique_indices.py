import asyncio
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from setup_hotpotqa_db import convert_metadata_to_db
from embedding_text_generator import generate_embedding_texts_from_db
from path_embedding_generator import PathEmbeddingGenerator
from bm25_indexer import BM25Indexer

async def main():
    print("Starting MuSiQue Index Generation Pipeline...")
    
    # 1. Generate DB
    print("\n[Step 1] Generating Metadata DB...")
    convert_metadata_to_db(
        metadata_json_path='MuSiQue/musique_sample_200_metadata.json',
        db_path='MuSiQue/metadata_v3.db'
    )

    # 2. Generate Embedding Texts
    print("\n[Step 2] Generating Embedding Texts...")
    generate_embedding_texts_from_db(
        db_path='MuSiQue/metadata_v3.db',
        output_path='MuSiQue/embedding_texts.json',
        language="en"
    )

    # 3. Generate Dense Embeddings
    print("\n[Step 3] Generating Dense Embeddings...")
    generator = PathEmbeddingGenerator(batch_size=200, max_concurrency=5)
    await generator.generate_embeddings(
        input_path='MuSiQue/embedding_texts.json',
        output_path='MuSiQue/path_embeddings.npz'
    )

    # 4. Generate BM25 Index
    print("\n[Step 4] Generating BM25 Index...")
    indexer = BM25Indexer(use_stemming=True)
    indexer.build_index(
        embedding_texts_path='MuSiQue/embedding_texts.json',
        index_save_path='MuSiQue/bm25_index'
    )
    
    print("\nAll MuSiQue indices generated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
