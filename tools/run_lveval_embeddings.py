#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


# Ensure repo root is importable when running as `python tools/run_lveval_embeddings.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_embedding_generator import PathEmbeddingGenerator


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="LVEVAL/embedding_texts_v5.json")
    ap.add_argument("--output", default="LVEVAL/path_embeddings_v5.npz")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gen = PathEmbeddingGenerator(batch_size=int(args.batch_size), max_concurrency=int(args.concurrency))
    asyncio.run(gen.generate_embeddings(input_path=str(args.input), output_path=str(args.output)))


if __name__ == "__main__":
    main()
