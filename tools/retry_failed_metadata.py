#!/usr/bin/env python3
"""Retry metadata generation only for failed items and merge back.

This script is intended for cases like LVEVAL where `build_metadata.py` produced
~99% success but some items have `context_metadata[*].error` due to transient
LLM/JSON parsing issues.

It:
- loads the original input JSON used for metadata generation
- loads an existing metadata JSON output
- finds items that are missing metadata or contain errors
- retries metadata generation for only those items (optionally in multiple rounds)
- merges successful results back into the metadata JSON

Example (LVEVAL):
  python tools/retry_failed_metadata.py \
    --input LVEVAL/lveval_corpus_for_pipeline.json \
    --existing LVEVAL/metadata_v5.json \
    --inplace \
    --max-rounds 3 \
    --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure repo root is importable when running as `python tools/retry_failed_metadata.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_metadata import initialize_llm_client, process_dataset


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _needs_retry(meta_item: Dict[str, Any], *, treat_empty_metadata_as_error: bool) -> bool:
    ctx_list = meta_item.get("context_metadata")
    if not isinstance(ctx_list, list) or len(ctx_list) == 0:
        return True

    for ctx in ctx_list:
        if not isinstance(ctx, dict):
            return True
        if ctx.get("error"):
            return True
        if "metadata" not in ctx:
            return True
        md = ctx.get("metadata")
        if md is None:
            return True
        if treat_empty_metadata_as_error and isinstance(md, dict) and len(md) == 0:
            return True

    return False


def _parse_ids_file(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retry failed metadata items and merge results")

    ap.add_argument("--input", required=True, help="Original input JSON used for metadata generation")
    ap.add_argument("--existing", required=True, help="Existing metadata JSON to repair (build_metadata output)")

    ap.add_argument(
        "--output",
        default=None,
        help="Output path for merged metadata JSON (default: <existing>_repaired.json unless --inplace)",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite --existing in-place (creates a .bak timestamped backup)",
    )

    ap.add_argument("-m", "--model", default="openai/gpt-4o-mini", help="Model name")
    ap.add_argument("--concurrency", type=int, default=50, help="Async concurrency for retries")
    ap.add_argument("--batch-size", type=int, default=10, help="Intermediate save batch size (unused when output_path=None)")

    ap.add_argument("--max-rounds", type=int, default=3, help="Max retry rounds")
    ap.add_argument("--sleep-between-rounds", type=float, default=1.0, help="Sleep seconds between rounds")

    ap.add_argument(
        "--treat-empty-metadata-as-error",
        action="store_true",
        help="Also retry items whose metadata dict is empty {}",
    )

    ap.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional text file with one id per line to retry (overrides auto-detection)",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    existing_path = Path(args.existing)

    input_data = _load_json(input_path)
    if not isinstance(input_data, list):
        raise ValueError(f"Expected list JSON for --input, got {type(input_data).__name__}: {input_path}")

    existing_data = _load_json(existing_path)
    if not isinstance(existing_data, list):
        raise ValueError(f"Expected list JSON for --existing, got {type(existing_data).__name__}: {existing_path}")

    input_by_id: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(input_data):
        if not isinstance(item, dict):
            continue
        iid = item.get("_id") or item.get("id")
        if iid is None:
            continue
        input_by_id[str(iid)] = item

    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(existing_data):
        if not isinstance(item, dict):
            continue
        iid = item.get("_id") or item.get("id")
        if iid is None:
            continue
        meta_by_id[str(iid)] = item

    # Determine target ids
    if args.ids_file:
        ids = _parse_ids_file(Path(args.ids_file))
        target_ids = [i for i in ids if i in input_by_id]
    else:
        target_ids = []
        for iid, inp in input_by_id.items():
            meta_item = meta_by_id.get(iid)
            if meta_item is None:
                target_ids.append(iid)
                continue
            if _needs_retry(meta_item, treat_empty_metadata_as_error=bool(args.treat_empty_metadata_as_error)):
                target_ids.append(iid)

    target_set: Set[str] = set(target_ids)
    print(f"Found {len(target_set)} items to retry")

    if not target_set:
        print("Nothing to do.")
        return

    client = initialize_llm_client()

    remaining: Set[str] = set(target_set)
    last_remaining_count = None

    for round_idx in range(int(args.max_rounds)):
        if not remaining:
            break

        if last_remaining_count is not None and len(remaining) >= last_remaining_count:
            print("No progress in remaining count; stopping early.")
            break
        last_remaining_count = len(remaining)

        batch_items = [input_by_id[iid] for iid in sorted(remaining, key=lambda x: int(x) if x.isdigit() else x)]
        print(f"\n=== Retry round {round_idx + 1}/{args.max_rounds}: {len(batch_items)} items ===")

        # Run build_metadata pipeline for only the remaining items.
        # Do not pass output_path to avoid partial overwrites mid-run.
        results: List[Dict[str, Any]] = asyncio.run(
            process_dataset(
                client=client,
                data=batch_items,
                model=args.model,
                max_passages=None,
                batch_size=int(args.batch_size),
                concurrency=int(args.concurrency),
                output_path=None,
                dry_run=False,
            )
        )

        # Merge results back.
        for item in results:
            if not isinstance(item, dict):
                continue
            iid = item.get("_id") or item.get("id")
            if iid is None:
                continue
            iid = str(iid)
            meta_by_id[iid] = item

        # Recompute remaining
        new_remaining: Set[str] = set()
        for iid in remaining:
            meta_item = meta_by_id.get(iid)
            if meta_item is None:
                new_remaining.add(iid)
                continue
            if _needs_retry(meta_item, treat_empty_metadata_as_error=bool(args.treat_empty_metadata_as_error)):
                new_remaining.add(iid)

        fixed = len(remaining) - len(new_remaining)
        remaining = new_remaining
        print(f"Round done. Fixed: {fixed}, Still failing: {len(remaining)}")

        if remaining and args.sleep_between_rounds > 0:
            time.sleep(float(args.sleep_between_rounds))

    # Write output
    if args.inplace:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = existing_path.with_suffix(existing_path.suffix + f".bak.{ts}")
        existing_path.replace(backup_path)
        out_path = existing_path
        print(f"Backup created: {backup_path}")
    else:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = existing_path.with_name(existing_path.stem + "_repaired" + existing_path.suffix)

    merged_list = list(meta_by_id.values())

    # Keep a stable ordering when ids are numeric.
    def _sort_key(x: Dict[str, Any]):
        iid = x.get("_id") or x.get("id")
        s = "" if iid is None else str(iid)
        return (0, int(s)) if s.isdigit() else (1, s)

    merged_list.sort(key=_sort_key)
    _dump_json(out_path, merged_list)

    print(f"\nSaved merged metadata: {out_path}")
    print(f"Remaining failures after retries: {len(remaining)}")
    if remaining:
        sample = sorted(remaining)[:50]
        print("Sample remaining ids (up to 50):")
        for iid in sample:
            print(iid)


if __name__ == "__main__":
    main()
