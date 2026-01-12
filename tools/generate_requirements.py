from __future__ import annotations

import ast
import sys
from pathlib import Path
import importlib.metadata as md


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    req_path = root / "requirements.txt"

    excluded_dirs = {".venv", ".git", "__pycache__"}

    def is_excluded(path: Path) -> bool:
        try:
            rel = path.relative_to(root)
        except Exception:
            return True
        return any(part in excluded_dirs for part in rel.parts)

    py_files = [p for p in root.rglob("*.py") if p.is_file() and not is_excluded(p)]

    # Collect local module/package candidates so we don't treat them as dependencies.
    local_toplevel: set[str] = set()
    for p in py_files:
        try:
            rel = p.relative_to(root)
        except Exception:
            continue
        parts = rel.parts
        if not parts:
            continue
        if len(parts) == 1 and parts[0].endswith(".py"):
            local_toplevel.add(parts[0][:-3])
        else:
            local_toplevel.add(parts[0])

    stdlib = set(getattr(sys, "stdlib_module_names", ()))

    used_modules: set[str] = set()
    parse_errors: list[tuple[str, str]] = []

    def add_mod(name: str) -> None:
        if not name:
            return
        top = name.split(".")[0]
        if top:
            used_modules.add(top)

    for p in py_files:
        try:
            src = _read_text(p)
        except Exception as e:
            parse_errors.append((str(p.relative_to(root)), f"read_error:{e}"))
            continue

        try:
            tree = ast.parse(src, filename=str(p))
        except Exception as e:
            parse_errors.append((str(p.relative_to(root)), f"parse_error:{e}"))
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_mod(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if getattr(node, "level", 0) and node.level > 0:
                    continue
                add_mod(node.module or "")

    candidates: set[str] = set()
    for m in used_modules:
        if m in stdlib:
            continue
        if m in local_toplevel:
            continue
        if m in {"__pycache__"}:
            continue
        candidates.add(m)

    pkg_map = md.packages_distributions()

    dists: set[str] = set()
    unresolved_modules: list[str] = []
    for m in sorted(candidates):
        dist_names = pkg_map.get(m)
        if not dist_names:
            unresolved_modules.append(m)
            continue
        for dn in dist_names:
            dists.add(dn)

    pinned: list[str] = []
    missing_dists: list[str] = []
    for dn in sorted(dists, key=str.lower):
        try:
            v = md.version(dn)
            pinned.append(f"{dn}=={v}")
        except Exception:
            missing_dists.append(dn)

    lines: list[str] = []
    lines.append("# Auto-generated from workspace imports (AST scan)")
    lines.append("# Generated on 2025-12-27")
    lines.append(f"# Pinned to versions installed in: {root / '.venv'}")
    lines.append(f"# Python: {sys.version.split()[0]}")
    lines.append("")

    if pinned:
        lines.append("# Directly-used (resolved) packages")
        lines.extend(pinned)
        lines.append("")

    if missing_dists:
        lines.append("# Distributions found but version lookup failed")
        lines.extend(missing_dists)
        lines.append("")

    if unresolved_modules:
        lines.append("# Unresolved imported top-level modules (not mapped in current env)")
        lines.extend([f"# - {m}" for m in unresolved_modules])
        lines.append("")

    if parse_errors:
        lines.append("# Files with parse/read errors during scan")
        for fp, err in parse_errors[:100]:
            lines.append(f"# - {fp}: {err}")
        if len(parse_errors) > 100:
            lines.append(f"# ... and {len(parse_errors)-100} more")
        lines.append("")

    req_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"WROTE {req_path}")
    print(f"py_files={len(py_files)} used_modules={len(used_modules)} candidates={len(candidates)}")
    print(f"pinned={len(pinned)} unresolved_modules={len(unresolved_modules)} parse_errors={len(parse_errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
