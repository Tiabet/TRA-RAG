#!/usr/bin/env python3
"""Prune build_metadata output by dropping failed/empty context_metadata entries.

Typical use-case:
- build_metadata.py succeeded for most items but a small remainder has
  `context_metadata[*].error` or missing metadata.
- For indexing, you may prefer dropping those failures and proceeding.

This script:
- Loads a metadata JSON produced by build_metadata.py (list of items)
- For each item, keeps only context_metadata entries that have a non-null
  `metadata` and no `error`
- Drops items that end up with 0 remaining contexts
- Writes a pruned JSON (and prints stats)

Example:
  python tools/prune_metadata_json.py \
    --input LVEVAL/metadata_v5.json \
    --output LVEVAL/metadata_v5.pruned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_ctx_ok(ctx: Any) -> bool:
    if not isinstance(ctx, dict):
        return False
    if ctx.get("error"):
        return False
    if "metadata" not in ctx:
        return False
    if ctx.get("metadata") is None:
        return False
    return True


def prune_item(item: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    """Return (new_item, kept_ctx, dropped_ctx)."""
    ctx_list = item.get("context_metadata")
    if not isinstance(ctx_list, list):
        ctx_list = []

    kept = [c for c in ctx_list if _is_ctx_ok(c)]
    dropped = len(ctx_list) - len(kept)

    new_item = dict(item)
    new_item["context_metadata"] = kept
    return new_item, len(kept), dropped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prune failed metadata items/contexts")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--drop-if-no-context",
        action="store_true",
        default=True,
        help="Drop items that have 0 contexts after pruning (default: true)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    data = _load_json(in_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON, got {type(data).__name__}: {in_path}")

    total_items = len(data)
    kept_items: List[Dict[str, Any]] = []

    items_dropped = 0
    ctx_kept = 0
    ctx_dropped = 0

    for item in data:
        if not isinstance(item, dict):
            items_dropped += 1
            continue
        new_item, k, d = prune_item(item)
        ctx_kept += k
        ctx_dropped += d

        if args.drop_if_no_context and k == 0:
            items_dropped += 1
            continue
        kept_items.append(new_item)

    _dump_json(out_path, kept_items)

    print("=" * 60)
    print("Prune metadata JSON")
    print("=" * 60)
    print(f"input items:          {total_items}")
    print(f"output items:         {len(kept_items)}")
    print(f"items dropped:        {items_dropped}")
    print(f"contexts kept:        {ctx_kept}")
    print(f"contexts dropped:     {ctx_dropped}")
    print(f"saved:               {out_path}")


if __name__ == "__main__":
    main()
