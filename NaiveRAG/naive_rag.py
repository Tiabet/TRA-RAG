"""
NaiveRAG Implementation (Universal)
===================================
다양한 QA 데이터셋에 적용 가능한 범용 NaiveRAG 구현
- Embedding: text-embedding-3-small
- Generation: openai/gpt-4o-mini
- 각 passage를 별도 청크로 관리
- Gold context 기반 retrieval 평가
"""

import json
import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize OpenAI clients
EMBED_CLIENT = OpenAI(
    api_key=os.getenv("ALICE_OPENAI_KEY"),
    base_url=os.getenv("ALICE_EMBED_URL")
)

CHAT_CLIENT = OpenAI(
    api_key=os.getenv("ALICE_OPENAI_KEY"),
    base_url=os.getenv("ALICE_CHAT_URL")
)

EMBED_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "openai/gpt-4o-mini"


class PassageChunk:
    """각 passage를 표현하는 청크"""
    def __init__(self, title: str, content: str, doc_id: str, passage_idx: int):
        self.title = title
        self.content = content
        self.doc_id = doc_id  # 원본 question의 _id
        self.passage_idx = passage_idx
        self.embedding = None
        
    def get_text(self) -> str:
        """임베딩을 위한 텍스트 반환"""
        # title과 content를 함께 임베딩
        sentences = [s.strip() for s in self.content if s.strip()]
        full_text = f"{self.title}\n" + " ".join(sentences)
        return full_text
    
    def __repr__(self):
        return f"PassageChunk(title={self.title}, doc_id={self.doc_id}, idx={self.passage_idx})"


class NaiveRAG:
    """범용 NaiveRAG 시스템"""
    
    def __init__(self):
        self.chunks: List[PassageChunk] = []
        self.embeddings = None
        self.gold_mappings = {}  # question_id -> set of (title, passage_idx)
        
    def load_data(self, jsonl_path: str):
        """JSONL 데이터 로드 및 청크 생성
        
        지원 포맷:
        - HotpotQA: context, supporting_facts 필드
        - 2WikiMultihopQA: context, supporting_facts 필드
        - MuSiQue: 유사한 구조
        """
        print(f"Loading data from {jsonl_path}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading documents"):
                data = json.loads(line)
                doc_id = data['_id']
                context = data['context']
                supporting_facts = data.get('supporting_facts', [])
                
                # Gold context 매핑 저장
                gold_set = set()
                for fact in supporting_facts:
                    title, sent_idx = fact
                    gold_set.add((title, sent_idx))
                self.gold_mappings[doc_id] = gold_set
                
                # 각 passage를 별도 청크로 생성
                for passage in context:
                    title = passage[0]
                    sentences = passage[1]
                    
                    chunk = PassageChunk(
                        title=title,
                        content=sentences,
                        doc_id=doc_id,
                        passage_idx=0  # passage 전체를 하나로
                    )
                    self.chunks.append(chunk)
        
        print(f"Loaded {len(self.chunks)} passage chunks")
        print(f"Loaded {len(self.gold_mappings)} gold mappings")
    
    def create_embeddings(self, batch_size: int = 100):
        """모든 청크에 대한 임베딩 생성"""
        print("Creating embeddings...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding chunks"):
            batch_chunks = self.chunks[i:i + batch_size]
            batch_texts = [chunk.get_text() for chunk in batch_chunks]
            
            try:
                response = EMBED_CLIENT.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # 각 청크에 임베딩 저장
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    chunk.embedding = embedding
                    
            except Exception as e:
                print(f"Error creating embeddings for batch {i}: {e}")
                raise
        
        self.embeddings = np.array(all_embeddings)
        print(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PassageChunk, float]]:
        """쿼리에 대한 top-k 청크 검색 (코사인 유사도)"""
        # Query embedding
        response = EMBED_CLIENT.embeddings.create(
            model=EMBED_MODEL,
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Cosine similarity (정규화된 벡터이므로 내적만 계산)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[PassageChunk, float]]) -> str:
        """검색된 청크를 기반으로 답변 생성"""
        # Context 구성
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            sentences = " ".join([s.strip() for s in chunk.content if s.strip()])
            context_parts.append(f"[{i}] {chunk.title}: {sentences}")
        
        context_text = "\n\n".join(context_parts)
        
        # Prompt 구성
        prompt = f"""Answer the question based on the given context.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = CHAT_CLIENT.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""
    
    def evaluate_retrieval(self, query_doc_id: str, retrieved_chunks: List[Tuple[PassageChunk, float]]) -> Dict[str, float]:
        """검색 성능 평가 (gold context 기반)
        
        Sentence-level 평가:
        - 같은 문서에서 여러 문장이 supporting fact인 경우를 올바르게 처리
        - 문서를 찾으면 해당 문서의 모든 supporting sentences를 찾은 것으로 간주
        """
        if query_doc_id not in self.gold_mappings:
            return {
                "recall": 0.0, 
                "precision": 0.0, 
                "f1": 0.0,
                "gold_count": 0,
                "retrieved_count": 0,
                "correct_count": 0
            }
        
        gold_facts = self.gold_mappings[query_doc_id]  # set of (title, sent_idx)
        
        # Retrieved titles 추출
        retrieved_titles = set([chunk.title for chunk, score in retrieved_chunks])
        
        # Correct: 검색된 title에 해당하는 모든 gold facts
        correct_facts = set()
        for gold_title, sent_idx in gold_facts:
            if gold_title in retrieved_titles:
                correct_facts.add((gold_title, sent_idx))
        
        # Recall 계산: 찾은 gold facts / 전체 gold facts
        if len(gold_facts) == 0:
            recall = 0.0
        else:
            recall = len(correct_facts) / len(gold_facts)
        
        # Precision 계산: 찾은 gold titles / 검색된 titles
        # (passage 단위로 평가)
        if len(retrieved_chunks) == 0:
            precision = 0.0
        else:
            gold_titles_only = set([title for title, _ in gold_facts])
            correct_titles = retrieved_titles & gold_titles_only
            precision = len(correct_titles) / len(retrieved_chunks)
        
        # F1 계산
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        
        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "gold_count": len(gold_facts),
            "retrieved_count": len(retrieved_chunks),
            "correct_count": len(correct_facts),
            "gold_unique_titles": len(gold_titles_only),
            "correct_titles": len(correct_titles) if len(retrieved_chunks) > 0 else 0
        }
    
    def save_index(self, output_path: str):
        """인덱스 저장 (chunks + embeddings)"""
        data = {
            'chunks': [
                {
                    'title': c.title,
                    'content': c.content,
                    'doc_id': c.doc_id,
                    'passage_idx': c.passage_idx
                }
                for c in self.chunks
            ],
            'embeddings': self.embeddings.tolist(),
            'gold_mappings': {
                k: list(v) for k, v in self.gold_mappings.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Index saved to {output_path}")
    
    def load_index(self, index_path: str):
        """저장된 인덱스 로드"""
        print(f"Loading index from {index_path}...")
        
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Chunks 복원
        self.chunks = [
            PassageChunk(
                title=c['title'],
                content=c['content'],
                doc_id=c['doc_id'],
                passage_idx=c['passage_idx']
            )
            for c in data['chunks']
        ]
        
        # Embeddings 복원
        self.embeddings = np.array(data['embeddings'])
        
        # Gold mappings 복원
        self.gold_mappings = {
            k: set(tuple(v) for v in vals)
            for k, vals in data['gold_mappings'].items()
        }
        
        # 각 chunk에 embedding 연결
        for chunk, embedding in zip(self.chunks, self.embeddings):
            chunk.embedding = embedding
        
        print(f"Loaded {len(self.chunks)} chunks with embeddings")
        print(f"Loaded {len(self.gold_mappings)} gold mappings")


def main():
    """메인 실행 함수 - 사용 예시"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python naive_rag.py <data_path> [index_path]")
        print("Example: python naive_rag.py ../HotpotQA/hotpot.jsonl ../HotpotQA/naive_rag_index.json")
        sys.exit(1)
    
    DATA_PATH = sys.argv[1]
    INDEX_PATH = sys.argv[2] if len(sys.argv) > 2 else DATA_PATH.replace('.jsonl', '_index.json')
    
    # NaiveRAG 초기화
    rag = NaiveRAG()
    
    # 데이터 로드 및 임베딩 생성
    if not os.path.exists(INDEX_PATH):
        rag.load_data(DATA_PATH)
        rag.create_embeddings(batch_size=100)
        rag.save_index(INDEX_PATH)
    else:
        rag.load_index(INDEX_PATH)
    
    # 샘플 쿼리 테스트
    print("\n" + "="*50)
    print("Sample Query Test")
    print("="*50)
    
    # 첫 번째 질문으로 테스트
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sample_data = json.loads(first_line)
    
    query = sample_data['question']
    doc_id = sample_data['_id']
    gold_answer = sample_data.get('answer', 'N/A')
    
    print(f"\nQuestion: {query}")
    print(f"Gold Answer: {gold_answer}")
    
    # Retrieval
    retrieved = rag.retrieve(query, top_k=5)
    
    print(f"\nTop-{len(retrieved)} Retrieved Passages:")
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"{i}. [{score:.4f}] {chunk.title}")
    
    # Evaluation
    eval_results = rag.evaluate_retrieval(doc_id, retrieved)
    print(f"\nRetrieval Metrics:")
    print(f"  Recall: {eval_results['recall']:.4f}")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  F1: {eval_results['f1']:.4f}")


if __name__ == "__main__":
    main()
