import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


def stream_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream-parse a JSON array from disk using stdlib JSONDecoder."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""

        # Seek '['
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                raise ValueError("Unexpected EOF while looking for '['")
            buf += chunk
            i = 0
            while i < len(buf) and buf[i].isspace():
                i += 1
            if i < len(buf) and buf[i] == "[":
                buf = buf[i + 1 :]
                break
            buf = buf[-1024:]

        while True:
            # skip ws + commas
            i = 0
            while True:
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i < len(buf) and buf[i] == ",":
                    i += 1
                    continue
                break

            if i >= len(buf):
                chunk = f.read(1024 * 1024)
                if not chunk:
                    raise ValueError("Unexpected EOF inside JSON array")
                buf = ""
                continue

            if buf[i] == "]":
                return

            try:
                obj, end = decoder.raw_decode(buf, idx=i)
            except json.JSONDecodeError:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    raise
                buf += chunk
                continue

            yield obj
            buf = buf[end:]


def _accumulate_shared(
    shared_values: List[Dict[str, Any]],
    value_stats: Dict[str, Dict[str, float]],
    value_question_seen: Dict[str, set],
    key_counter: Counter,
    keypair_counter: Counter,
    question_id: str,
    side: str,
):
    """Accumulate stats from rs_shared_values or rd_shared_values."""
    for entry in shared_values or []:
        val = entry.get("value")
        if not val:
            continue

        pair_count = int(entry.get("pair_count") or 0)
        links = entry.get("links") or []
        links_len = len(links)

        vs = value_stats.setdefault(
            val,
            {
                "total_pair_count": 0.0,
                "total_links": 0.0,
                "max_links": 0.0,
                "max_pair_count": 0.0,
                "entries": 0.0,
            },
        )
        vs["total_pair_count"] += pair_count
        vs["total_links"] += links_len
        vs["max_links"] = max(vs["max_links"], links_len)
        vs["max_pair_count"] = max(vs["max_pair_count"], pair_count)
        vs["entries"] += 1

        value_question_seen.setdefault(val, set()).add(question_id)

        # sample keys by scanning a limited number of links to keep runtime bounded
        # (links can be huge for RD)
        sample_links = links[:50]
        for l in sample_links:
            if side == "rs":
                r_keys = l.get("r_keys") or []
                t_keys = l.get("s_keys") or []
            else:
                r_keys = l.get("r_keys") or []
                t_keys = l.get("d_keys") or []

            for k in r_keys:
                key_counter[k] += 1
            for k in t_keys:
                key_counter[k] += 1
            for rk in r_keys:
                for tk in t_keys:
                    keypair_counter[(rk, tk)] += 1


def _top_value_table(value_stats: Dict[str, Dict[str, float]], value_question_seen: Dict[str, set], top_n: int = 30):
    rows = []
    for val, st in value_stats.items():
        rows.append(
            {
                "value": val,
                "questions": len(value_question_seen.get(val, set())),
                "total_pair_count": int(st.get("total_pair_count", 0)),
                "total_links": int(st.get("total_links", 0)),
                "max_links": int(st.get("max_links", 0)),
                "max_pair_count": int(st.get("max_pair_count", 0)),
                "entries": int(st.get("entries", 0)),
            }
        )

    # two rankings
    by_pair = sorted(rows, key=lambda r: (r["total_pair_count"], r["questions"], r["total_links"]), reverse=True)[:top_n]
    by_links = sorted(rows, key=lambda r: (r["total_links"], r["questions"], r["max_links"]), reverse=True)[:top_n]
    return {"top_by_total_pair_count": by_pair, "top_by_total_links": by_links}


def main():
    base = Path(r"c:\Development\ChunkRAG_v2")
    path = base / "Analysis" / "expanded_sd_links.json"
    out_json = base / "Analysis" / "expanded_sd_links_actionable_report.json"

    total_questions = 0

    rs_value_stats: Dict[str, Dict[str, float]] = {}
    rd_value_stats: Dict[str, Dict[str, float]] = {}
    rs_value_qseen: Dict[str, set] = {}
    rd_value_qseen: Dict[str, set] = {}

    rs_key_counter = Counter()
    rd_key_counter = Counter()
    rs_keypair_counter = Counter()
    rd_keypair_counter = Counter()

    # keep some examples for top bombs
    rd_bomb_examples: Dict[str, Dict[str, Any]] = {}

    for obj in stream_json_array(path):
        total_questions += 1
        qid = obj.get("question_id") or f"q{total_questions}"
        question = obj.get("question")

        rs_shared = obj.get("rs_shared_values") or []
        rd_shared = obj.get("rd_shared_values") or []

        _accumulate_shared(rs_shared, rs_value_stats, rs_value_qseen, rs_key_counter, rs_keypair_counter, qid, side="rs")
        _accumulate_shared(rd_shared, rd_value_stats, rd_value_qseen, rd_key_counter, rd_keypair_counter, qid, side="rd")

        # Track per-question biggest RD entry to attach examples
        if rd_shared:
            biggest = max(rd_shared, key=lambda e: len(e.get("links") or []))
            bval = biggest.get("value")
            if bval:
                cand = rd_bomb_examples.get(bval)
                entry = {
                    "question_id": qid,
                    "question": question,
                    "links_len": len(biggest.get("links") or []),
                    "pair_count": int(biggest.get("pair_count") or 0),
                }
                if cand is None or entry["links_len"] > cand.get("links_len", 0):
                    rd_bomb_examples[bval] = entry

    report = {
        "file": str(path),
        "total_questions": total_questions,
        "rs": {
            "values": _top_value_table(rs_value_stats, rs_value_qseen, top_n=40),
            "keys_top": rs_key_counter.most_common(50),
            "keypairs_top": [
                {"r_key": rk, "t_key": tk, "count": c}
                for (rk, tk), c in rs_keypair_counter.most_common(50)
            ],
        },
        "rd": {
            "values": _top_value_table(rd_value_stats, rd_value_qseen, top_n=40),
            "keys_top": rd_key_counter.most_common(50),
            "keypairs_top": [
                {"r_key": rk, "t_key": tk, "count": c}
                for (rk, tk), c in rd_keypair_counter.most_common(50)
            ],
            "bomb_examples_by_value": rd_bomb_examples,
        },
        "notes": {
            "key_sampling": "For performance, key/keypair counts sample at most 50 links per shared-value entry.",
            "value_rankings": "Values are ranked separately by total_pair_count and by total_links. RD 'bombs' are best seen in top_by_total_links/max_links.",
        },
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
