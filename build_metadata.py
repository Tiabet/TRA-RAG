"""
LLM 기반 메타데이터 생성 스크립트
===================================
데이터셋의 각 passage에 대해 LLM을 사용하여 구조화된 메타데이터를 생성합니다.

Usage:
    # output 자동 생성 (입력 파일명_metadata.json)
    python build_metadata.py -i HotpotQA/hotpotqa_sample_200.json
    
    # output 수동 지정
    python build_metadata.py -i HotpotQA/hotpotqa_sample_200.json -o metadata/hotpot_metadata.json
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

# 환경 변수 로드
load_dotenv()

# Prompt 로드
from Prompt.metadata_construction_prompt import metadata_construction_prompt


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="LLM을 사용하여 passage에서 메타데이터를 생성합니다."
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="입력 데이터셋 JSON 파일 경로 (예: HotpotQA/HotpotQA_200.json)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 메타데이터 JSON 파일 경로 (기본값: 입력 파일명_metadata.json)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="사용할 LLM 모델 (기본값: openai/gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="처리할 최대 passage 수 (테스트용, 기본값: 전체)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="중간 저장 배치 크기 (기본값: 10)"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="동시 처리 개수 (기본값: 5)"
    )
    
    return parser.parse_args()


def initialize_llm_client():
    """LLM 클라이언트 초기화"""
    base_url = os.getenv("ALICE_CHAT_URL")
    api_key = os.getenv("ALICE_OPENAI_KEY")

    print(f"Using ALICE_CHAT_URL: {base_url}")
    print(f"Using ALICE_OPENAI_KEY: {api_key[:4]}...{api_key[-4:] if api_key else 'None'}")
    
    if not base_url or not api_key:
        raise ValueError(
            "환경 변수가 설정되지 않았습니다. "
            ".env 파일에 ALICE_CHAT_URL과 ALICE_OPENAI_KEY를 설정하세요."
        )
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    return client


def format_passage_for_prompt(title: str, sentences: List[str]) -> str:
    """
    Passage를 프롬프트 입력 형식으로 포맷팅
    
    Args:
        title: passage의 제목
        sentences: 문장 리스트
    
    Returns:
        프롬프트에 삽입할 형식의 문자열
    """
    # [[title, [sentences]]] 형식으로 포맷팅
    passage_str = json.dumps([[title, sentences]], ensure_ascii=False)
    return passage_str


async def generate_metadata(client: AsyncOpenAI, passage: List, model: str) -> Dict[str, Any]:
    """
    LLM을 사용하여 passage에서 메타데이터 생성 (비동기)
    
    Args:
        client: AsyncOpenAI 클라이언트
        passage: [title, [sentences]] 형식의 passage
        model: 사용할 모델명
    
    Returns:
        생성된 메타데이터 (JSON 형식)
    """
    title = passage[0]
    sentences = passage[1]
    
    # Passage를 프롬프트 형식으로 포맷팅
    passage_input = format_passage_for_prompt(title, sentences)
    
    # 프롬프트에 passage 삽입
    full_prompt = metadata_construction_prompt.replace("{{input}}", passage_input)
    
    try:
        # LLM 호출 (비동기)
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
            temperature=0.1,
            max_tokens=8192,
            response_format={"type": "json_object"}
        )
        
        # 응답에서 JSON 추출
        metadata_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        try:
            # Markdown 코드 블록 제거 (```json ... ```)
            if metadata_text.startswith("```"):
                # 첫 번째와 마지막 줄 제거
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
                "error": f"JSON 파싱 실패: {str(e)}",
                "raw_response": metadata_text
            }
    
    except Exception as e:
        return {
            "success": False,
            "metadata": None,
            "title": title,
            "error": f"LLM 호출 실패: {str(e)}"
        }


async def process_dataset(
    client: AsyncOpenAI,
    data: List[Dict],
    model: str,
    max_passages: int = None,
    batch_size: int = 10,
    concurrency: int = 5,
    output_path: Path = None
) -> List[Dict]:
    """
    데이터셋의 모든 passage에 대해 메타데이터 생성 (비동기 병렬 처리)
    
    Args:
        client: AsyncOpenAI 클라이언트
        data: 입력 데이터셋
        model: 사용할 모델명
        max_passages: 처리할 최대 passage 수
        batch_size: 중간 저장 배치 크기
        concurrency: 동시 처리 개수
        output_path: 출력 파일 경로
    
    Returns:
        메타데이터가 추가된 데이터셋
    """
    results = []
    total_passages = 0
    processed_passages = 0
    failed_passages = 0
    
    # 전체 passage 수 계산 및 작업 목록 생성
    tasks = []
    for item in data:
        for passage in item.get("context", []):
            if max_passages and len(tasks) >= max_passages:
                break
            tasks.append({
                "item": item,
                "passage": passage
            })
        if max_passages and len(tasks) >= max_passages:
            break
    
    total_passages = len(tasks)
    
    print(f"\n📊 처리 대상: {total_passages}개 passages")
    print(f"🤖 모델: {model}")
    print(f"⚡ 동시 처리: {concurrency}개\n")
    
    # Semaphore로 동시 실행 개수 제한
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_with_semaphore(task_info):
        async with semaphore:
            try:
                print(f"🔄 처리 중: {task_info['passage'][0][:50]}...")
                result = await generate_metadata(client, task_info["passage"], model)
                print(f"✅ 완료: {task_info['passage'][0][:50]}...")
                # 약간의 딜레이 (Rate Limit 방지)
                await asyncio.sleep(0.05)
                return task_info, result
            except Exception as e:
                print(f"❌ 오류: {task_info['passage'][0][:50]}... - {str(e)}")
                return task_info, {
                    "success": False,
                    "metadata": None,
                    "title": task_info["passage"][0],
                    "error": f"처리 중 예외 발생: {str(e)}"
                }
    
    # 진행률 표시를 위한 progress bar
    pbar = tqdm(total=total_passages, desc="메타데이터 생성")
    
    # 항목별로 결과 정리를 위한 딕셔너리
    item_results = {}
    
    # 배치 처리 (한번에 너무 많은 작업을 생성하지 않도록)
    batch_process_size = concurrency * 10  # 동시성의 10배씩 처리
    for i in range(0, len(tasks), batch_process_size):
        batch_tasks = tasks[i:i+batch_process_size]
        print(f"\n🔄 배치 {i//batch_process_size + 1}/{(len(tasks)-1)//batch_process_size + 1} 처리 중 ({len(batch_tasks)}개)...")
        
        # 배치 단위로 비동기 작업 실행 - as_completed로 완료되는 즉시 처리
        pending_tasks = [process_with_semaphore(t) for t in batch_tasks]
        
        for coro in asyncio.as_completed(pending_tasks):
            task_info, result = await coro
            
            item = task_info["item"]
            item_id = item.get("_id") or item.get("id")
            
            # 항목별 결과 초기화
            if item_id not in item_results:
                item_results[item_id] = {
                    "id": item_id,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "context_metadata": []
                }
            
            # 메타데이터 추가
            if result["success"]:
                item_results[item_id]["context_metadata"].append({
                    "title": result["title"],
                    "metadata": result["metadata"]
                })
            else:
                item_results[item_id]["context_metadata"].append({
                    "title": result["title"],
                    "error": result["error"],
                    "raw_response": result.get("raw_response")
                })
                failed_passages += 1
            
            processed_passages += 1
            pbar.update(1)
            
            # 중간 저장
            if output_path and processed_passages % batch_size == 0:
                current_results = list(item_results.values())
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(current_results, f, indent=2, ensure_ascii=False)
                pbar.set_postfix({"저장": f"{len(current_results)}개 항목"})
    
    pbar.close()
    
    # 최종 결과 리스트로 변환
    results = list(item_results.values())
    
    # 통계 출력
    print(f"\n{'='*60}")
    print("📊 처리 완료 통계")
    print(f"{'='*60}")
    print(f"✅ 성공: {processed_passages - failed_passages}개")
    print(f"❌ 실패: {failed_passages}개")
    print(f"📈 성공률: {(processed_passages - failed_passages) / processed_passages * 100:.1f}%")
    print(f"{'='*60}\n")
    
    return results


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 경로 설정
    input_path = Path(args.input)
    
    # output이 지정되지 않은 경우 자동 생성
    if args.output is None:
        # 입력 파일의 디렉토리와 파일명 추출
        input_dir = input_path.parent
        input_stem = input_path.stem  # 확장자 제외한 파일명
        input_ext = input_path.suffix  # 확장자
        
        # output 경로 생성: 같은 폴더에 _metadata 추가
        output_path = input_dir / f"{input_stem}_metadata{input_ext}"
    else:
        output_path = Path(args.output)
    
    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🚀 메타데이터 생성 시작")
    print("="*60)
    print(f"📂 입력: {input_path}")
    print(f"💾 출력: {output_path}")
    print(f"🤖 모델: {args.model}")
    print(f"⚡ 동시 처리: {args.concurrency}개")
    
    # 데이터 로드
    print(f"\n📖 데이터 로딩 중...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   ✅ {len(data)}개 항목 로드됨")
    
    # LLM 클라이언트 초기화
    print(f"\n🔧 LLM 클라이언트 초기화 중...")
    client = initialize_llm_client()
    print(f"   ✅ 클라이언트 초기화 완료")
    
    # 메타데이터 생성 (비동기 실행)
    results = asyncio.run(process_dataset(
        client=client,
        data=data,
        model=args.model,
        max_passages=args.max_passages,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        output_path=output_path
    ))
    
    # 최종 저장
    print(f"💾 최종 결과 저장 중...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 저장 완료: {output_path}")
    
    print(f"\n{'='*60}")
    print("✨ 모든 작업 완료!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
