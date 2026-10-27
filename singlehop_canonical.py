"""Canonicalization helpers for single-hop datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_popqa_canonical_map(root: Path | None = None) -> dict[str, Any]:
    base = root or ROOT
    path = base / "POPQA" / "canonical_doc_id_map.json"
    if not path.exists():
        return {"canonical_by_doc_id": {}, "canonical_title_by_doc_id": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_doc_id(mapping: dict[str, Any] | None, doc_id: Any) -> str:
    s = str(doc_id)
    if not mapping:
        return s
    return str((mapping.get("canonical_by_doc_id") or {}).get(s, s))


def canonical_title(mapping: dict[str, Any] | None, doc_id: Any, title: Any) -> str:
    s = str(doc_id)
    if not mapping:
        return str(title or "")
    return str((mapping.get("canonical_title_by_doc_id") or {}).get(s, title or ""))


def should_canonicalize(dataset: str, disabled: bool = False) -> bool:
    return (not disabled) and str(dataset).lower() == "popqa"
