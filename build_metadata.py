"""LLM-based metadata generation script
====================================
Generates structured metadata for passages using an LLM.

Usage:
    # Auto-generate output path (<input_stem>_metadata<ext>)
    python build_metadata.py -i MuSiQue/musique.json

    # Specify output explicitly
    python build_metadata.py -i HotpotQA/hotpotqa.json -o HotpotQA/metadata_v5.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
import asyncio

# Load environment variables
load_dotenv()

# Load prompt
from Prompt.metadata_construction_prompt import metadata_construction_prompt


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate metadata from passages using an LLM."
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input dataset JSON path (QA-style JSON list)."
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output metadata JSON path (default: <input_stem>_metadata<ext> in the same folder)."
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="LLM model name (default: openai/gpt-4o-mini)."
    )
    
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Maximum number of passages to process (default: all)."
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Intermediate snapshot batch size (default: 10)."
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=200,
        help="Concurrency (default: 200)."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input parsing and doc_id/ctx_idx mapping without calling the LLM."
    )
    
    return parser.parse_args()


def initialize_llm_client():
    """Initialize the LLM client."""
    base_url = os.getenv("ALICE_CHAT_URL")
    api_key = os.getenv("ALICE_OPENAI_KEY")

    print(f"Using ALICE_CHAT_URL: {base_url}")
    print(f"Using ALICE_OPENAI_KEY: {api_key[:4]}...{api_key[-4:] if api_key else 'None'}")
    
    if not base_url or not api_key:
        raise ValueError(
            "Missing required environment variables. "
            "Set ALICE_CHAT_URL and ALICE_OPENAI_KEY (e.g., in a .env file)."
        )
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    return client


def format_passage_for_prompt(title: str, sentences: List[str]) -> str:
    """
    Format a passage into the prompt input format.
    
    Args:
        title: Passage title
        sentences: List of sentence strings
    
    Returns:
        A string to inject into the prompt
    """
    # Format as [[title, [sentences]]]
    passage_str = json.dumps([[title, sentences]], ensure_ascii=False)
    return passage_str


async def generate_metadata(client: AsyncOpenAI, passage: List, model: str) -> Dict[str, Any]:
    """
    Generate metadata for a passage using an LLM (async).
    
    Args:
        client: AsyncOpenAI client
        passage: Passage in [title, [sentences]] format
        model: Model name
    
    Returns:
        Metadata result (JSON)
    """
    title = passage[0]
    sentences = passage[1]
    
    # Format passage into prompt input
    passage_input = format_passage_for_prompt(title, sentences)
    
    # Inject passage into prompt
    full_prompt = metadata_construction_prompt.replace("{{input}}", passage_input)
    
    try:
        # LLM call (async)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert metadata extraction engine. Extract structured metadata from passages and return only valid JSON."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=0.0,
            max_tokens=8192,
            response_format={"type": "json_object"}
        )
        
        # Extract JSON text from response
        metadata_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            # Strip Markdown code fences (```json ... ```)
            if metadata_text.startswith("```"):
                # Drop the first and last fence lines
                lines = metadata_text.split("\n")
                metadata_text = "\n".join(lines[1:-1])
            
            metadata = json.loads(metadata_text)
            return {
                "success": True,
                "metadata": metadata,
                "title": title,
                "error": None
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "metadata": None,
                "title": title,
                "error": f"JSON parse failed: {str(e)}",
                "raw_response": metadata_text
            }
    
    except Exception as e:
        return {
            "success": False,
            "metadata": None,
            "title": title,
            "error": f"LLM call failed: {str(e)}"
        }


async def process_dataset(
    client: AsyncOpenAI,
    data: List[Dict],
    model: str,
    max_passages: int = None,
    batch_size: int = 10,
    concurrency: int = 5,
    output_path: Path = None,
    dry_run: bool = False,
) -> List[Dict]:
    """
    Generate metadata for all passages in a dataset (async, concurrent).
    
    Args:
        client: AsyncOpenAI client
        data: Input dataset
        model: Model name
        max_passages: Max passages
        batch_size: Snapshot batch size
        concurrency: Concurrency
        output_path: Output path
    
    Returns:
        Dataset with metadata added
    """
    results = []
    total_passages = 0
    processed_passages = 0
    failed_passages = 0
    
    def _extract_passages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return passages with a stable per-item ordering.

        Supports:
        - HotpotQA legacy: context = [[title, [sentences]], ...]
        - HotpotQA corpus_idx: context = [{title, sentences, corpus_idx, local_idx}, ...]
        - MuSiQue corpus_idx: paragraphs = [{title, paragraph_text, corpus_idx, local_idx, ...}, ...]
        """
        passages: List[Dict[str, Any]] = []

        # 1) MuSiQue (preferred when present)
        if isinstance(item.get("paragraphs"), list):
            for i, p in enumerate(item.get("paragraphs") or []):
                if not isinstance(p, dict):
                    continue
                title = str(p.get("title") or "")
                text = str(p.get("paragraph_text") or "")
                # We don't require sentence splitting; prompt accepts list[str].
                sentences = [text] if text else []
                corpus_idx = p.get("corpus_idx")
                local_idx = p.get("local_idx", i)
                passages.append({
                    "title": title,
                    "sentences": sentences,
                    "ctx_idx": int(local_idx) if local_idx is not None else i,
                    "doc_id": str(corpus_idx) if corpus_idx is not None else None,
                })
            # Ensure stable by ctx_idx
            passages.sort(key=lambda x: int(x.get("ctx_idx", 0)))
            return passages

        # 2) HotpotQA (context)
        context = item.get("context", []) or []
        for i, c in enumerate(context):
            if isinstance(c, list) and len(c) >= 2:
                title = str(c[0])
                sentences = c[1] if isinstance(c[1], list) else [str(c[1])]
                passages.append({
                    "title": title,
                    "sentences": [str(s) for s in sentences],
                    "ctx_idx": i,
                    "doc_id": None,
                })
            elif isinstance(c, dict):
                title = str(c.get("title") or "")
                sentences = c.get("sentences")
                if isinstance(sentences, list):
                    sent_list = [str(s) for s in sentences]
                else:
                    sent_list = [str(sentences)] if sentences else []
                corpus_idx = c.get("corpus_idx")
                local_idx = c.get("local_idx", i)
                passages.append({
                    "title": title,
                    "sentences": sent_list,
                    "ctx_idx": int(local_idx) if local_idx is not None else i,
                    "doc_id": str(corpus_idx) if corpus_idx is not None else None,
                })

        passages.sort(key=lambda x: int(x.get("ctx_idx", 0)))
        return passages

    # Build UNIQUE tasks keyed by doc_id when available (e.g., corpus_idx),
    # while still keeping per-item ordering by ctx_idx.
    tasks_by_key: Dict[str, Dict[str, Any]] = {}
    item_context_titles: Dict[str, List[str]] = {}
    item_context_doc_ids: Dict[str, List[str]] = {}

    unique_task_count = 0
    for item in data:
        item_id = item.get("_id") or item.get("id")
        if not item_id:
            continue

        passages = _extract_passages(item)
        item_context_titles[item_id] = [p.get("title", "") for p in passages]
        # Fill doc_id list aligned with ctx_idx ordering; fallback to item-scoped doc_id if missing.
        doc_ids_aligned: List[str] = []
        for pi, p in enumerate(passages):
            did = p.get("doc_id")
            if did is None or str(did).strip() == "":
                did = f"{item_id}::ctx{pi}"
            doc_ids_aligned.append(str(did))
        item_context_doc_ids[item_id] = doc_ids_aligned

        for pi, p in enumerate(passages):
            if max_passages and unique_task_count >= max_passages:
                break

            # Use corpus_idx as key when provided; otherwise keep per-item unique id.
            doc_id = p.get("doc_id")
            if doc_id is None or str(doc_id).strip() == "":
                doc_id = f"{item_id}::ctx{pi}"
            doc_id = str(doc_id)

            key = doc_id
            if key not in tasks_by_key:
                tasks_by_key[key] = {
                    "doc_id": doc_id,
                    "passage": [p.get("title", ""), p.get("sentences", [])],
                    "refs": [],
                }
                unique_task_count += 1
            tasks_by_key[key]["refs"].append({
                "item": item,
                "item_id": str(item_id),
                "ctx_idx": int(pi),
                "context_len": int(len(passages)),
                "title": str(p.get("title", "")),
                "doc_id": doc_id,
            })

        if max_passages and unique_task_count >= max_passages:
            break

    tasks = list(tasks_by_key.values())
    total_passages = len(tasks)
    
    print(f"\nTarget passages: {total_passages}")
    print(f"Model: {model}")
    print(f"Concurrency: {concurrency}\n")

    if dry_run:
        print("DRY RUN: validating doc_id/ctx_idx mapping without LLM calls.\n")

        item_results: Dict[str, Dict[str, Any]] = {}

        def _ensure_item(item_id: str, item: Dict[str, Any], context_len: int):
            if item_id not in item_results:
                item_results[item_id] = {
                    "id": item_id,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "context_metadata": [None] * context_len,
                }

        for task_info in tasks:
            task_title = str((task_info.get("passage") or [""])[0])
            task_doc_id = str(task_info.get("doc_id") or "")
            for ref in task_info.get("refs", []) or []:
                item = ref.get("item") or {}
                item_id = str(ref.get("item_id"))
                ctx_idx = int(ref.get("ctx_idx", 0))
                context_len = int(ref.get("context_len", 0))
                ref_doc_id = str(ref.get("doc_id") or task_doc_id)
                _ensure_item(item_id, item, context_len)
                item_results[item_id]["context_metadata"][ctx_idx] = {
                    "title": task_title,
                    "metadata": {},
                    "ctx_idx": ctx_idx,
                    "doc_id": ref_doc_id,
                }

        def _materialize_context_metadata(item_id: str) -> List[Dict[str, Any]]:
            entry = item_results[item_id]
            ctx_list = entry.get("context_metadata", [])
            titles = item_context_titles.get(item_id, [])
            doc_ids = item_context_doc_ids.get(item_id, [])
            out: List[Dict[str, Any]] = []
            for i, v in enumerate(ctx_list):
                if v is not None:
                    out.append(v)
                    continue
                title = titles[i] if i < len(titles) else ""
                doc_id = doc_ids[i] if i < len(doc_ids) else f"{item_id}::ctx{i}"
                out.append({
                    "title": title,
                    "error": "metadata missing (not processed)",
                    "ctx_idx": i,
                    "doc_id": doc_id,
                })
            return out

        results: List[Dict[str, Any]] = []
        for iid, ent in item_results.items():
            ent_copy = dict(ent)
            ent_copy["context_metadata"] = _materialize_context_metadata(iid)
            results.append(ent_copy)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved dry-run output: {output_path}")

        return results
    
    # Limit concurrent execution with a semaphore
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_semaphore(task_info):
        async with semaphore:
            try:
                print(f"Processing: {task_info['passage'][0][:50]}...")
                result = await generate_metadata(client, task_info["passage"], model)
                print(f"Done: {task_info['passage'][0][:50]}...")
                # Small delay to reduce rate-limit pressure
                await asyncio.sleep(0.05)
                return task_info, result
            except Exception as e:
                print(f"Error: {task_info['passage'][0][:50]}... - {str(e)}")
                return task_info, {
                    "success": False,
                    "metadata": None,
                    "title": task_info["passage"][0],
                    "error": f"Exception during processing: {str(e)}"
                }
    
    # Progress bar
    pbar = tqdm(total=total_passages, desc="Generating metadata")
    
    # Accumulate results per item
    item_results: Dict[str, Dict[str, Any]] = {}

    def _materialize_context_metadata(item_id: str) -> List[Dict[str, Any]]:
        entry = item_results[item_id]
        ctx_list = entry.get("context_metadata", [])
        titles = item_context_titles.get(item_id, [])
        doc_ids = item_context_doc_ids.get(item_id, [])
        out: List[Dict[str, Any]] = []
        for i, v in enumerate(ctx_list):
            if v is not None:
                out.append(v)
                continue
            title = titles[i] if i < len(titles) else ""
            doc_id = doc_ids[i] if i < len(doc_ids) else f"{item_id}::ctx{i}"
            out.append({
                "title": title,
                "error": "metadata missing (not processed)",
                "ctx_idx": i,
                "doc_id": doc_id,
            })
        return out
    
    # Batch tasks so we don't create too many coroutines at once
    batch_process_size = concurrency * 10
    for i in range(0, len(tasks), batch_process_size):
        batch_tasks = tasks[i:i+batch_process_size]
        print(
            f"\nBatch {i//batch_process_size + 1}/{(len(tasks)-1)//batch_process_size + 1}: "
            f"processing {len(batch_tasks)} passages..."
        )
        
        # Run batch tasks and process them as they complete
        pending_tasks = [process_with_semaphore(t) for t in batch_tasks]
        
        for coro in asyncio.as_completed(pending_tasks):
            task_info, result = await coro
            
            task_doc_id = str(task_info.get("doc_id") or "")

            # A single task may correspond to many (item, ctx_idx) refs when doc_id is corpus_idx.
            for ref in task_info.get("refs", []) or []:
                item = ref.get("item") or {}
                item_id = ref.get("item_id")
                ctx_idx = int(ref.get("ctx_idx", 0))
                context_len = int(ref.get("context_len", 0))
                ref_doc_id = str(ref.get("doc_id") or task_doc_id)

                # Initialize per-item results
                if item_id not in item_results:
                    item_results[item_id] = {
                        "id": item_id,
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        # Keep list indexed by ctx_idx to preserve stable ordering.
                        "context_metadata": [None] * context_len,
                    }

                # Add metadata
                if result["success"]:
                    item_results[item_id]["context_metadata"][ctx_idx] = {
                        "title": result["title"],
                        "metadata": result["metadata"],
                        "ctx_idx": ctx_idx,
                        "doc_id": ref_doc_id,
                    }
                else:
                    item_results[item_id]["context_metadata"][ctx_idx] = {
                        "title": result["title"],
                        "error": result["error"],
                        "raw_response": result.get("raw_response"),
                        "ctx_idx": ctx_idx,
                        "doc_id": ref_doc_id,
                    }
                    failed_passages += 1
            
            processed_passages += 1
            pbar.update(1)
            
            # Intermediate snapshot
            if output_path and processed_passages % batch_size == 0:
                current_results = []
                for iid, ent in item_results.items():
                    ent_copy = dict(ent)
                    ent_copy["context_metadata"] = _materialize_context_metadata(iid)
                    current_results.append(ent_copy)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(current_results, f, indent=2, ensure_ascii=False)
                pbar.set_postfix({"saved": f"{len(current_results)} items"})
    
    pbar.close()
    
    # Final results list (materialize None slots)
    results = []
    for iid, ent in item_results.items():
        ent_copy = dict(ent)
        ent_copy["context_metadata"] = _materialize_context_metadata(iid)
        results.append(ent_copy)
    
    # Summary stats
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Success: {processed_passages - failed_passages}")
    print(f"Failed: {failed_passages}")
    print(f"Success rate: {(processed_passages - failed_passages) / processed_passages * 100:.1f}%")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Paths
    input_path = Path(args.input)
    
    # Auto-generate output path when not provided
    if args.output is None:
        # Extract input folder and file stem
        input_dir = input_path.parent
        input_stem = input_path.stem
        input_ext = input_path.suffix
        
        # Output path: <stem>_metadata<ext> in the same folder
        output_path = input_dir / f"{input_stem}_metadata{input_ext}"
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Metadata generation")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {args.model}")
    print(f"Concurrency: {args.concurrency}")
    if args.dry_run:
        print("DRY RUN: enabled")
    
    # Load data
    print("\nLoading data...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    
    client = None
    if not args.dry_run:
        # Initialize client
        print("\nInitializing LLM client...")
        client = initialize_llm_client()
        print("LLM client initialized")
    
    # Generate metadata (async)
    results = asyncio.run(process_dataset(
        client=client,
        data=data,
        model=args.model,
        max_passages=args.max_passages,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        output_path=output_path,
        dry_run=args.dry_run,
    ))
    
    # Final save
    print("Saving final results...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()