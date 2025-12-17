#!/usr/bin/env python3
"""\
Prepare HotpotQA sample file
============================
Creates a deterministic sample JSON (default: 200 items) from HotpotQA/hotpotqa.json.

This repo's pipelines and index builders typically expect:
- HotpotQA/hotpotqa_sample_200.json
"""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sample HotpotQA JSON file")
    parser.add_argument("--input", type=str, default="HotpotQA/hotpotqa.json", help="Path to HotpotQA full JSON")
    parser.add_argument("--output", type=str, default="HotpotQA/hotpotqa_sample_200.json", help="Output sample path")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--head", action="store_true", help="Use first N items instead of random sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list")

    n = min(args.sample_size, len(data))
    if args.head:
        sampled = data[:n]
    else:
        rnd = random.Random(args.seed)
        sampled = rnd.sample(data, n)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(sampled)} items -> {out_path}")


if __name__ == "__main__":
    main()
