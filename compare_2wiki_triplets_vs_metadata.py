#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI


MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Doc:
	doc_id: str
	title: str
	text: str


def _repo_root() -> Path:
	# This script is expected to live at the workspace root.
	return Path(__file__).resolve().parent


def _load_json(path: Path) -> Any:
	return json.loads(path.read_text(encoding="utf-8"))


def _iter_unique_docs_from_2wiki_qa(data: Any) -> Dict[str, Doc]:
	"""Extract unique documents from 2wiki QA JSON.

	We treat each paragraph as a document, uniquely identified by paragraph['corpus_idx'].
	"""
	out: Dict[str, Doc] = {}
	if not isinstance(data, list):
		return out

	for item in data:
		if not isinstance(item, dict):
			continue
		for para in (item.get("paragraphs") or []):
			if not isinstance(para, dict):
				continue
			doc_id = para.get("corpus_idx")
			if doc_id is None:
				continue
			k = str(doc_id)
			if k in out:
				continue
			title = str(para.get("title") or "")
			text = str(para.get("paragraph_text") or "").strip()
			if not text:
				continue
			out[k] = Doc(doc_id=k, title=title, text=text)
	return out


def _import_prompts(extract_mode: str) -> Tuple[str, str]:
	"""Return (BUILD_EXTRACT_PROMPT, EVALUATION_PROMPT) for the requested mode."""
	repo_root = _repo_root()
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

	try:
		extract_mode = str(extract_mode or "triplets").strip().lower()
		if extract_mode == "triplets":
			from Analysis.prompt.corpus_to_triplets import BUILD_TRIPLET_PROMPT  # type: ignore
			build_prompt = str(BUILD_TRIPLET_PROMPT)
		elif extract_mode in ("hyperedge", "hyperedges"):
			from Analysis.prompt.corpus_to_hyperedge import BUILD_HYPEREDGE_PROMPT  # type: ignore
			build_prompt = str(BUILD_HYPEREDGE_PROMPT)
		else:
			raise ValueError(f"Unknown extract_mode: {extract_mode}")

		from Prompt.compare_evaluation import EVALUATION_PROMPT  # type: ignore
		return build_prompt, str(EVALUATION_PROMPT)
	except Exception as e:
		raise RuntimeError(
			"Failed to import prompts. Expected Analysis/prompt/* and Prompt/compare_evaluation.py to be importable. "
			f"(error={e})"
		)


def _parse_json_array_best_effort(raw: str) -> Optional[List[Any]]:
	text = (raw or "").strip()
	if not text:
		return None

	# Common: models wrap JSON in Markdown fences.
	# Example: ```json\n[ ... ]\n```
	if text.startswith("```"):
		# Strip the first fence line and the trailing fence if present.
		parts = text.splitlines()
		if parts:
			parts = parts[1:]
			if parts and parts[-1].strip().startswith("```"):
				parts = parts[:-1]
			text = "\n".join(parts).strip()

	# First, try direct JSON parse.
	try:
		obj = json.loads(text)
		return obj if isinstance(obj, list) else None
	except Exception:
		pass

	# Best-effort: extract the first JSON array substring.
	# Handles cases like: "Here is the JSON:\n[ ... ]" or trailing commentary.
	start = text.find("[")
	end = text.rfind("]")
	if start != -1 and end != -1 and end > start:
		sub = text[start : end + 1].strip()
		try:
			obj = json.loads(sub)
			return obj if isinstance(obj, list) else None
		except Exception:
			return None
	return None


def _parse_triplets(raw: str) -> List[Tuple[str, str, str]]:
	"""Parse the triplet extractor output.

	The prompt asks for JSON, but the example uses Python-like tuples.
	We accept either:
	- JSON array of arrays/tuples
	- Python literal list of tuples
	"""
	text = (raw or "").strip()
	if not text:
		return []

	# Try JSON first
	try:
		obj = json.loads(text)
		triples: List[Tuple[str, str, str]] = []
		if isinstance(obj, list):
			for it in obj:
				if isinstance(it, (list, tuple)) and len(it) == 3:
					triples.append((str(it[0]), str(it[1]), str(it[2])))
		return triples
	except Exception:
		pass

	# Fallback: Python literal
	try:
		obj = ast.literal_eval(text)
		triples = []
		if isinstance(obj, list):
			for it in obj:
				if isinstance(it, tuple) and len(it) == 3:
					triples.append((str(it[0]), str(it[1]), str(it[2])))
				elif isinstance(it, list) and len(it) == 3:
					triples.append((str(it[0]), str(it[1]), str(it[2])))
		return triples
	except Exception:
		return []


def _extract_final_decision(text: str) -> Optional[str]:
	if not text:
		return None
	# Accept: "Final Decision: [Candidate 1 / Candidate 2]" or "Final Decision: Candidate 1"
	m = re.search(r"Final\s*Decision\s*:\s*\[?\s*(Candidate\s*[12])", text, flags=re.IGNORECASE)
	if not m:
		return None
	dec = m.group(1).strip().lower().replace(" ", "")
	if dec == "candidate1":
		return "Candidate 1"
	if dec == "candidate2":
		return "Candidate 2"
	return None


class MetadataDB:
	def __init__(self, db_path: Path):
		self.db_path = db_path
		self.conn = sqlite3.connect(str(db_path))
		self.conn.row_factory = sqlite3.Row

	def get_metadata_rows(self, doc_id: str) -> List[Dict[str, Any]]:
		cur = self.conn.cursor()
		cur.execute(
			"SELECT source_title, entity_title, metadata_json FROM metadata WHERE doc_id = ?",
			(str(doc_id),),
		)
		rows = cur.fetchall() or []
		out: List[Dict[str, Any]] = []
		for r in rows:
			mj = r["metadata_json"]
			try:
				obj = json.loads(mj) if isinstance(mj, str) else mj
			except Exception:
				obj = mj
			out.append(
				{
					"source_title": r["source_title"],
					"entity_title": r["entity_title"],
					"metadata": obj,
				}
			)
		return out

	def close(self) -> None:
		try:
			self.conn.close()
		except Exception:
			pass


async def _chat(client: AsyncOpenAI, prompt: str, max_tokens: int) -> str:
	resp = await client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "You are a precise assistant. Follow the requested output format."},
			{"role": "user", "content": prompt},
		],
		temperature=0.0,
		max_tokens=int(max_tokens),
	)
	return (resp.choices[0].message.content or "").strip()


async def _chat_json_object(client: AsyncOpenAI, prompt: str, max_tokens: int) -> str:
	resp = await client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "Return a single JSON object only."},
			{"role": "user", "content": prompt},
		],
		temperature=0.0,
		max_tokens=int(max_tokens),
		response_format={"type": "json_object"},
	)
	return (resp.choices[0].message.content or "").strip()


async def extract_triplets(client: AsyncOpenAI, build_triplet_prompt: str, passage: str) -> Tuple[str, List[Tuple[str, str, str]]]:
	prompt = build_triplet_prompt.replace("{{passage}}", passage)
	raw = await _chat(client, prompt=prompt, max_tokens=1200)
	triples = _parse_triplets(raw)
	return raw, triples


async def extract_hyperedges(client: AsyncOpenAI, build_hyperedge_prompt: str, corpus: str) -> Tuple[str, List[Any]]:
	# Prompt uses {{corpus}}
	prompt = build_hyperedge_prompt.replace("{{corpus}}", corpus)
	raw = await _chat(client, prompt=prompt, max_tokens=1600)
	obj = _parse_json_array_best_effort(raw)
	return raw, (obj or [])


async def compare_quality(
	client: AsyncOpenAI,
	evaluation_prompt: str,
	original_corpus: str,
	candidate1_text: str,
	candidate2_text: str,
) -> Tuple[str, Optional[str]]:
	prompt = evaluation_prompt
	prompt = prompt.replace("{{original_corpus}}", original_corpus)
	prompt = prompt.replace("{{triplet}}", candidate1_text)
	prompt = prompt.replace("{{ERA_structure}}", candidate2_text)
	text = await _chat_json_object(client, prompt=prompt, max_tokens=1600)
	return text, _extract_final_decision(text)


EVAL_CRITERIA = [
	"completeness",
	"faithfulness",
	"structural_clarity",
	"atomic_granularity",
]


def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
	if not text:
		return None
	try:
		obj = json.loads(text)
		return obj if isinstance(obj, dict) else None
	except Exception:
		return None


def _parse_eval_scores(eval_json: Dict[str, Any]) -> Tuple[Optional[Dict[str, Dict[str, int]]], Optional[Dict[str, float]], Optional[str]]:
	"""Return (scores, averages, final_decision) from evaluation JSON."""
	scores_obj = eval_json.get("scores")
	if not isinstance(scores_obj, dict):
		return None, None, None

	scores: Dict[str, Dict[str, int]] = {}
	for crit in EVAL_CRITERIA:
		v = scores_obj.get(crit)
		if not isinstance(v, dict):
			return None, None, None
		c1 = v.get("candidate1")
		c2 = v.get("candidate2")
		if not isinstance(c1, int) or not isinstance(c2, int):
			return None, None, None
		scores[crit] = {"candidate1": int(c1), "candidate2": int(c2)}

	# Prefer provided average if valid; otherwise compute.
	avg_obj = eval_json.get("average")
	avg_c1: Optional[float] = None
	avg_c2: Optional[float] = None
	if isinstance(avg_obj, dict):
		try:
			avg_c1 = float(avg_obj.get("candidate1"))
			avg_c2 = float(avg_obj.get("candidate2"))
		except Exception:
			avg_c1 = None
			avg_c2 = None
	if avg_c1 is None or avg_c2 is None:
		avg_c1 = sum(scores[c]["candidate1"] for c in EVAL_CRITERIA) / float(len(EVAL_CRITERIA))
		avg_c2 = sum(scores[c]["candidate2"] for c in EVAL_CRITERIA) / float(len(EVAL_CRITERIA))
	averages = {"candidate1": float(avg_c1), "candidate2": float(avg_c2)}

	final_decision = eval_json.get("final_decision")
	if isinstance(final_decision, str):
		fd = final_decision.strip()
		if fd in ("Candidate 1", "Candidate 2"):
			final_decision = fd
		else:
			final_decision = None
	else:
		final_decision = None

	return scores, averages, final_decision


def _winner_type(
	final_decision: Optional[str],
	candidate1_type: str,
	candidate2_type: str,
) -> Optional[str]:
	if not final_decision:
		return None
	if final_decision.strip().lower().replace(" ", "") == "candidate1":
		return candidate1_type
	if final_decision.strip().lower().replace(" ", "") == "candidate2":
		return candidate2_type
	return None


def _should_swap(seed: int, doc_id: str) -> bool:
	# Deterministic per-doc swap decision so runs are reproducible.
	rng = random.Random(f"swap::{seed}::{doc_id}")
	return bool(rng.getrandbits(1))


def _load_processed_doc_ids(path: Path) -> set[str]:
	seen: set[str] = set()
	if not path.exists():
		return seen
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			rec = json.loads(line)
			d = rec.get("doc_id")
			if d is not None:
				seen.add(str(d))
		except Exception:
			continue
	return seen


async def main_async(args: argparse.Namespace) -> None:
	# Load .env so OPENAI_API_KEY can be picked up without exporting it.
	# Do not override already-exported environment variables.
	load_dotenv(override=False)

	api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
	if not api_key:
		raise SystemExit("Missing env var OPENAI_API_KEY")

	repo_root = _repo_root()
	qa_path = repo_root / "2WikiMultihopQA" / "2wikimultihopqa.json"
	metadata_db_path = repo_root / "2WikiMultihopQA" / "metadata_v5.db"
	out_path = repo_root / (args.output or "Results/2wiki_triplets_vs_metadata_eval_gpt4o-mini.jsonl")
	out_path.parent.mkdir(parents=True, exist_ok=True)

	qa = _load_json(qa_path)
	docs_all = _iter_unique_docs_from_2wiki_qa(qa)
	min_text_len = int(getattr(args, "min_text_len", 0) or 0)
	if min_text_len > 0:
		docs = {k: v for k, v in docs_all.items() if len(v.text) >= min_text_len}
	else:
		docs = docs_all
	if len(docs) < int(args.n_docs):
		raise SystemExit(
			f"Not enough unique docs after filtering: have {len(docs)} (min_text_len={min_text_len}), need {int(args.n_docs)}"
		)

	rng = random.Random(int(args.seed))
	selected_ids = rng.sample(list(docs.keys()), int(args.n_docs))

	extract_mode = str(getattr(args, "extract_mode", "triplets") or "triplets").strip().lower()
	build_extract_prompt, evaluation_prompt = _import_prompts(extract_mode=extract_mode)
	client = AsyncOpenAI(api_key=api_key)
	md = MetadataDB(metadata_db_path)

	processed = _load_processed_doc_ids(out_path) if args.resume else set()
	remaining = [d for d in selected_ids if d not in processed]
	print(f"[OK] Unique docs in QA: {len(docs_all)}  Eligible: {len(docs)} (min_text_len={min_text_len})")
	print(f"[OK] Selected: {len(selected_ids)}  Remaining(after resume): {len(remaining)}")
	print(f"[OK] Output: {out_path}")

	sem = asyncio.Semaphore(int(args.concurrency))
	lock = asyncio.Lock()

	async def process_one(doc_id: str) -> None:
		async with sem:
			doc = docs[doc_id]
			meta_rows = md.get_metadata_rows(doc_id)
			meta_payload = {
				"doc_id": doc_id,
				"title": doc.title,
				"rows": meta_rows,
			}

			t0 = time.time()
			triplets_raw: Optional[str] = None
			triplets: List[Tuple[str, str, str]] = []
			hyperedges_raw: Optional[str] = None
			hyperedges: List[Any] = []

			if extract_mode == "triplets":
				triplets_raw, triplets = await extract_triplets(client, build_extract_prompt, doc.text)
				extracted_type = "triplets"
				extracted_text = json.dumps(triplets, ensure_ascii=False)
			else:
				hyperedges_raw, hyperedges = await extract_hyperedges(client, build_extract_prompt, doc.text)
				extracted_type = "hyperedge"
				extracted_text = json.dumps(hyperedges, ensure_ascii=False)

			metadata_json = json.dumps(meta_payload, ensure_ascii=False)

			shuffle_enabled = not bool(getattr(args, "no_shuffle_candidates", False))
			swap = _should_swap(int(args.seed), doc_id) if shuffle_enabled else False
			candidate1_type = extracted_type
			candidate2_type = "metadata"
			candidate1_text = extracted_text
			candidate2_text = metadata_json
			if swap:
				candidate1_type, candidate2_type = candidate2_type, candidate1_type
				candidate1_text, candidate2_text = candidate2_text, candidate1_text

			eval_text, decision = await compare_quality(
				client,
				evaluation_prompt,
				original_corpus=doc.text,
				candidate1_text=candidate1_text,
				candidate2_text=candidate2_text,
			)
			eval_json = _extract_json_obj(eval_text)
			score_table, avg_table, decision_json = (None, None, None)
			if isinstance(eval_json, dict):
				score_table, avg_table, decision_json = _parse_eval_scores(eval_json)
			# Prefer JSON decision when available
			decision = decision_json or decision
			elapsed = time.time() - t0
			winner = _winner_type(decision, candidate1_type=candidate1_type, candidate2_type=candidate2_type)

			# Compute extracted/metadata averages regardless of candidate order
			extracted_avg: Optional[float] = None
			metadata_avg: Optional[float] = None
			if isinstance(avg_table, dict):
				if candidate1_type == extracted_type:
					extracted_avg = float(avg_table.get("candidate1"))
					metadata_avg = float(avg_table.get("candidate2"))
				else:
					extracted_avg = float(avg_table.get("candidate2"))
					metadata_avg = float(avg_table.get("candidate1"))

			metadata_win_value: Optional[float] = None
			if extracted_avg is not None and metadata_avg is not None:
				if metadata_avg > extracted_avg:
					metadata_win_value = 1.0
				elif metadata_avg < extracted_avg:
					metadata_win_value = 0.0
				else:
					metadata_win_value = 0.5

			rec = {
				"doc_id": doc_id,
				"title": doc.title,
				"text_len": len(doc.text),
				"extract_mode": extract_mode,
				"extracted_type": extracted_type,
				"triplets_raw": triplets_raw,
				"triplets": triplets,
				"hyperedges_raw": hyperedges_raw,
				"hyperedges": hyperedges,
				"metadata": meta_payload,
				"eval_candidate1_type": candidate1_type,
				"eval_candidate2_type": candidate2_type,
				"evaluation": eval_text,
				"evaluation_json": eval_json,
				"eval_scores": score_table,
				"eval_avg": avg_table,
				"extracted_avg": extracted_avg,
				"metadata_avg": metadata_avg,
				"metadata_win_value": metadata_win_value,
				"final_decision": decision,
				"winner_type": winner,
				"model": MODEL,
				"elapsed_s": float(elapsed),
			}

			line = json.dumps(rec, ensure_ascii=False)
			async with lock:
				with out_path.open("a", encoding="utf-8") as f:
					f.write(line + "\n")

	# Run tasks
	tasks = [asyncio.create_task(process_one(doc_id)) for doc_id in remaining]
	# Wait while allowing partial progress even if some tasks fail
	results = await asyncio.gather(*tasks, return_exceptions=True)
	failed = sum(1 for r in results if isinstance(r, Exception))
	if failed:
		print(f"[WARN] {failed} tasks failed. See output file for completed items.")
	else:
		print("[OK] All tasks completed")

	# Summarize metadata win-rate over this sampled set (handles --resume)
	try:
		sampled_set = set(selected_ids)
		wins_sum = 0.0
		wins_n = 0

		extracted_sums: Dict[str, float] = {c: 0.0 for c in EVAL_CRITERIA}
		metadata_sums: Dict[str, float] = {c: 0.0 for c in EVAL_CRITERIA}
		extracted_counts: Dict[str, int] = {c: 0 for c in EVAL_CRITERIA}
		metadata_counts: Dict[str, int] = {c: 0 for c in EVAL_CRITERIA}
		extracted_avg_sum = 0.0
		metadata_avg_sum = 0.0
		avg_n = 0
		extracted_type_seen: Optional[str] = None

		if out_path.exists():
			for line in out_path.read_text(encoding="utf-8").splitlines():
				line = line.strip()
				if not line:
					continue
				try:
					rec = json.loads(line)
				except Exception:
					continue
				doc_id = str(rec.get("doc_id")) if rec.get("doc_id") is not None else ""
				if doc_id not in sampled_set:
					continue

				# Win-rate (metadata vs extracted) based on eval_avg
				mv = rec.get("metadata_win_value")
				if isinstance(mv, (int, float)):
					wins_sum += float(mv)
					wins_n += 1

				# Per-criterion averages
				scores = rec.get("eval_scores")
				c1t = str(rec.get("eval_candidate1_type") or "")
				c2t = str(rec.get("eval_candidate2_type") or "")
				ext_type = str(rec.get("extracted_type") or "")
				if extracted_type_seen is None and ext_type:
					extracted_type_seen = ext_type

				if (
					isinstance(scores, dict)
					and c1t
					and c2t
					and ext_type
					and (c1t == "metadata" or c2t == "metadata")
					and (c1t == ext_type or c2t == ext_type)
				):
					for crit in EVAL_CRITERIA:
						row = scores.get(crit)
						if not isinstance(row, dict):
							continue
						s1 = row.get("candidate1")
						s2 = row.get("candidate2")
						if not isinstance(s1, int) or not isinstance(s2, int):
							continue
						if c1t == "metadata":
							metadata_sums[crit] += float(s1)
							metadata_counts[crit] += 1
							extracted_sums[crit] += float(s2)
							extracted_counts[crit] += 1
						else:
							metadata_sums[crit] += float(s2)
							metadata_counts[crit] += 1
							extracted_sums[crit] += float(s1)
							extracted_counts[crit] += 1

				ea = rec.get("extracted_avg")
				ma = rec.get("metadata_avg")
				if isinstance(ea, (int, float)) and isinstance(ma, (int, float)):
					extracted_avg_sum += float(ea)
					metadata_avg_sum += float(ma)
					avg_n += 1
		if wins_n > 0:
			print(f"[SUMMARY] metadata_win_rate = {wins_sum / float(wins_n):.4f}  (n={wins_n}, tie=0.5)")
		else:
			print("[SUMMARY] metadata_win_rate = N/A (no valid scored rows yet)")

		# Print per-criterion averages (may be partial if some rows failed JSON scoring)
		type_label = extracted_type_seen or str(getattr(args, "extract_mode", "extracted") or "extracted")
		crit_lines: List[str] = []
		for crit in EVAL_CRITERIA:
			n = min(extracted_counts[crit], metadata_counts[crit])
			if n <= 0:
				continue
			ex_mean = extracted_sums[crit] / float(extracted_counts[crit])
			md_mean = metadata_sums[crit] / float(metadata_counts[crit])
			crit_lines.append(f"{crit}: {type_label}={ex_mean:.3f}  metadata={md_mean:.3f} (n={n})")
		if crit_lines:
			print(f"[SUMMARY] Per-criterion means ({type_label} vs metadata):")
			for ln in crit_lines:
				print(f"  - {ln}")
		else:
			print("[SUMMARY] Per-criterion means: N/A (no valid eval_scores rows yet)")

		if avg_n > 0:
			print(
				f"[SUMMARY] Overall avg (mean of 4 criteria): {type_label}={extracted_avg_sum/float(avg_n):.3f}  metadata={metadata_avg_sum/float(avg_n):.3f} (n={avg_n})"
			)
		else:
			print("[SUMMARY] Overall avg: N/A (no valid eval_avg rows yet)")
	except Exception as e:
		print(f"[WARN] Failed to compute metadata win-rate summary: {e}")

	md.close()


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Sample 1000 unique 2wiki docs, extract structures (triplets/hyperedges), fetch metadata, and compare quality with LLM.",
	)
	p.add_argument("--n_docs", type=int, default=1000)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--concurrency", type=int, default=100)
	p.add_argument(
		"--extract_mode",
		type=str,
		default="hyperedge",
		choices=["triplets", "hyperedge", "hyperedges"],
		help="Which structure to extract for Candidate 1 (default: hyperedge)",
	)
	p.add_argument(
		"--min_text_len",
		type=int,
		default=200,
		help="Filter out documents with text shorter than this many characters before sampling (default: 0 = no filter)",
	)
	p.add_argument("--output", type=str, default="Results/hyperedge_vs_metadata_eval_gpt4o-mini.jsonl")
	p.add_argument(
		"--no_shuffle_candidates",
		action="store_true",
		help="Disable candidate order shuffling (default: enabled)",
	)
	p.add_argument("--resume", action="store_true", help="Skip doc_ids already present in output JSONL")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	asyncio.run(main_async(args))


if __name__ == "__main__":
	main()
