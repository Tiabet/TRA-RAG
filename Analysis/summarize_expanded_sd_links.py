import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


def stream_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream-parse a JSON array from disk using stdlib JSONDecoder.

    Works for a file like: [ {..}, {..}, ... ] without loading everything.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        # Read until we see '['
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
            # keep last part in case '[' spans chunks
            buf = buf[-1024:]

        while True:
            # Skip whitespace and commas
            i = 0
            while True:
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i < len(buf) and buf[i] == ",":
                    i += 1
                    continue
                break

            # Need more data
            if i >= len(buf):
                chunk = f.read(1024 * 1024)
                if not chunk:
                    raise ValueError("Unexpected EOF inside JSON array")
                buf = ""
                continue

            # End of array
            if buf[i] == "]":
                return

            # Try decode one object
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


class RunningStats:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = None
        self.max = None

    def add(self, x: float):
        self.n += 1
        self.sum += x
        self.sum_sq += x * x
        self.min = x if self.min is None else min(self.min, x)
        self.max = x if self.max is None else max(self.max, x)

    def mean(self) -> float:
        return self.sum / self.n if self.n else 0.0

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        mu = self.mean()
        var = max(0.0, self.sum_sq / self.n - mu * mu)
        return math.sqrt(var)


def _top_entries(shared_values, top_n: int = 3):
    # shared_values are already sorted by pair_count desc in producer script
    out = []
    for entry in (shared_values or [])[:top_n]:
        out.append(
            {
                "value": entry.get("value"),
                "pair_count": entry.get("pair_count"),
                "count": entry.get("count"),
                "links_len": len(entry.get("links") or []),
                "sample_r_keys": (entry.get("links") or [{}])[0].get("r_keys") if entry.get("links") else None,
                "sample_d_keys": (entry.get("links") or [{}])[0].get("d_keys") if entry.get("links") else None,
                "sample_s_keys": (entry.get("links") or [{}])[0].get("s_keys") if entry.get("links") else None,
            }
        )
    return out


def main():
    base = Path(r"c:\Development\ChunkRAG_v2")
    path = base / "Analysis" / "expanded_sd_links.json"
    out_json = base / "Analysis" / "expanded_sd_links_summary.json"

    total = 0

    # simple proportions / quantiles (N=200 is small)
    rs_zero = 0
    support_zero = 0
    direct_hit_zero = 0
    distractor_list = []
    support_list = []

    # stats fields
    rs_len_stats = RunningStats()
    rd_len_stats = RunningStats()
    retrieved_docs_stats = RunningStats()
    expanded_support_stats = RunningStats()
    expanded_distractor_stats = RunningStats()
    direct_hits_stats = RunningStats()
    ratio_stats = RunningStats()

    # key frequency from top-1 rd / rs entries (avoid huge memory)
    rd_top_value_counter = Counter()
    rs_top_value_counter = Counter()
    rd_top_key_counter = Counter()
    rs_top_key_counter = Counter()

    # extreme examples
    worst_distractors = []  # list of (num_d, qid, question)
    worst_ratio = []        # (ratio, qid, question)
    worst_rd_keys = []      # (rd_len, qid, question)

    examples = []

    for obj in stream_json_array(path):
        total += 1
        qid = obj.get("question_id")
        question = obj.get("question")
        st = obj.get("stats") or {}

        retrieved_docs_stats.add(float(st.get("num_retrieved_docs", 0)))
        num_support = float(st.get("num_expanded_support_docs", 0))
        num_distractor = float(st.get("num_expanded_distractor_docs", 0))
        num_direct = float(st.get("num_direct_hits", 0))

        expanded_support_stats.add(num_support)
        expanded_distractor_stats.add(num_distractor)
        direct_hits_stats.add(num_direct)

        support_list.append(int(num_support))
        distractor_list.append(int(num_distractor))

        if num_support == 0:
            support_zero += 1
        if num_direct == 0:
            direct_hit_zero += 1
        ratio_stats.add(float(st.get("ratio_rd_to_rs", 0)))

        rs_shared = obj.get("rs_shared_values") or []
        rd_shared = obj.get("rd_shared_values") or []
        rs_len_stats.add(float(len(rs_shared)))
        rd_len_stats.add(float(len(rd_shared)))

        if len(rs_shared) == 0:
            rs_zero += 1

        # top-1 value + sample keys
        if rd_shared:
            v = rd_shared[0].get("value")
            if v:
                rd_top_value_counter[v] += 1
            links0 = rd_shared[0].get("links") or []
            if links0:
                for k in links0[0].get("r_keys") or []:
                    rd_top_key_counter[k] += 1
                for k in links0[0].get("d_keys") or []:
                    rd_top_key_counter[k] += 1

        if rs_shared:
            v = rs_shared[0].get("value")
            if v:
                rs_top_value_counter[v] += 1
            links0 = rs_shared[0].get("links") or []
            if links0:
                for k in links0[0].get("r_keys") or []:
                    rs_top_key_counter[k] += 1
                for k in links0[0].get("s_keys") or []:
                    rs_top_key_counter[k] += 1

        # keep top 10 extreme examples
        num_d = int(st.get("num_expanded_distractor_docs", 0))
        worst_distractors.append((num_d, qid, question))
        worst_distractors.sort(reverse=True)
        worst_distractors = worst_distractors[:10]

        ratio = float(st.get("ratio_rd_to_rs", 0))
        worst_ratio.append((ratio, qid, question))
        worst_ratio.sort(reverse=True)
        worst_ratio = worst_ratio[:10]

        rd_len = int(st.get("rd_link_key_count", 0))
        worst_rd_keys.append((rd_len, qid, question))
        worst_rd_keys.sort(reverse=True)
        worst_rd_keys = worst_rd_keys[:10]

        # store a few representative examples for debugging
        if len(examples) < 5:
            examples.append(
                {
                    "question_id": qid,
                    "question": question,
                    "stats": st,
                    "direct_hits": obj.get("direct_hits") or [],
                    "rs_top": _top_entries(rs_shared, top_n=2),
                    "rd_top": _top_entries(rd_shared, top_n=2),
                }
            )

    summary = {
        "file": str(path),
        "total_questions": total,
        "stats": {
            "num_retrieved_docs": {
                "mean": retrieved_docs_stats.mean(),
                "min": retrieved_docs_stats.min,
                "max": retrieved_docs_stats.max,
            },
            "num_expanded_support_docs": {
                "mean": expanded_support_stats.mean(),
                "min": expanded_support_stats.min,
                "max": expanded_support_stats.max,
            },
            "num_expanded_distractor_docs": {
                "mean": expanded_distractor_stats.mean(),
                "min": expanded_distractor_stats.min,
                "max": expanded_distractor_stats.max,
            },
            "num_direct_hits": {
                "mean": direct_hits_stats.mean(),
                "min": direct_hits_stats.min,
                "max": direct_hits_stats.max,
            },
            "rs_link_key_count": {
                "mean": rs_len_stats.mean(),
                "min": rs_len_stats.min,
                "max": rs_len_stats.max,
            },
            "rd_link_key_count": {
                "mean": rd_len_stats.mean(),
                "min": rd_len_stats.min,
                "max": rd_len_stats.max,
            },
            "ratio_rd_to_rs": {
                "mean": ratio_stats.mean(),
                "min": ratio_stats.min,
                "max": ratio_stats.max,
            },
        },
        "proportions": {
            "rs_shared_values_zero_frac": rs_zero / total if total else 0,
            "expanded_support_docs_zero_frac": support_zero / total if total else 0,
            "direct_hits_zero_frac": direct_hit_zero / total if total else 0,
        },
        "quantiles": {
            "expanded_distractor_docs": {
                "median": int(sorted(distractor_list)[len(distractor_list) // 2]) if distractor_list else 0,
                "p90": int(sorted(distractor_list)[int(0.9 * (len(distractor_list) - 1))]) if distractor_list else 0,
            },
            "expanded_support_docs": {
                "median": int(sorted(support_list)[len(support_list) // 2]) if support_list else 0,
                "p90": int(sorted(support_list)[int(0.9 * (len(support_list) - 1))]) if support_list else 0,
            },
        },
        "top_rd_value_top1": rd_top_value_counter.most_common(20),
        "top_rs_value_top1": rs_top_value_counter.most_common(20),
        "top_rd_keys_top1": rd_top_key_counter.most_common(30),
        "top_rs_keys_top1": rs_top_key_counter.most_common(30),
        "worst_cases": {
            "expanded_distractors": worst_distractors,
            "ratio_rd_to_rs": worst_ratio,
            "rd_link_key_count": worst_rd_keys,
        },
        "examples": examples,
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
