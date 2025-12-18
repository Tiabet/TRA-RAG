import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
from llm_logger import init_logger, finalize_log, log_llm_call

# Load env
load_dotenv()
API_KEY = os.getenv("ALICE_OPENAI_KEY")
BASE_URL = os.getenv("ALICE_CHAT_URL")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

QA_FILE = "HotpotQA/qa.json"
OUTPUT_FILE = "Results/qa_result_pure_gpt_4o_mini.json"

CONCURRENCY = 100
semaphore = asyncio.Semaphore(CONCURRENCY)


async def ask_model(index, question):
    async with semaphore:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You must answer in extremely concise form. "
                        "Follow HotpotQA answer style strictly: 1–5 words only, "
                        "no full sentences and no explanations. "
                        "Output only the final short answer phrase."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        "Give only the short answer (1–5 words). "
                        "No sentences. No explanation."
                    )
                }
            ]
        )

        answer = response.choices[0].message.content.strip()

        log_llm_call(
            call_type="Pure LLM QA",
            input_text=question,
            output_text=answer,
            context={
                "index": index,
                "model": "openai/gpt-4o-mini",
            },
        )

        return {
            "index": index,
            "query": question,
            "answer": answer
        }


async def run_qa_test():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    tasks = [
        asyncio.create_task(ask_model(i, item["query"]))
        for i, item in enumerate(qa_list)
    ]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        results.append(await coro)

    results.sort(key=lambda x: x["index"])

    cleaned = [{"query": r["query"], "answer": r["answer"]} for r in results]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Done! Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    init_logger()
    try:
        asyncio.run(run_qa_test())
    finally:
        finalize_log()
