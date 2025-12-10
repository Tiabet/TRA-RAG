import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from new_multihop_pipeline_v3 import NewMultihopPipelineV3
from hybrid_path_retriever import HybridPathRetriever

async def run_debug():
    load_dotenv()
    
    # Load regressions
    analysis_path = 'Results/comparison_analysis.json'
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    regressions = analysis.get('regressions', [])
    print(f"Found {len(regressions)} regression cases.")
    
    # Initialize pipeline
    client = AsyncOpenAI(
        api_key=os.getenv('ALICE_OPENAI_KEY'),
        base_url=os.getenv('ALICE_CHAT_URL')
    )
    
    retriever = HybridPathRetriever(
        bm25_weight=0.4,
        dense_weight=0.6,
        bm25_index_path='HotpotQA/bm25_index',
        embeddings_path='HotpotQA/path_embeddings.npz'
    )
    
    pipeline = NewMultihopPipelineV3(
        client=client,
        retriever=retriever,
        hotpotqa_path='HotpotQA/hotpotqa_sample_200.json',
        db_path='HotpotQA/metadata_v3.db',
        top_k=5, # Increase top_k slightly to see near misses
        verbose=False
    )
    
    log_path = 'Logs/debug_regressions.txt'
    os.makedirs('Logs', exist_ok=True)
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("Debug Analysis of Regression Cases\n")
        log_file.write("==================================\n\n")
        
        for i, case in enumerate(regressions):
            question = case['question']
            gold_answer = case['gold_answer']
            missing_titles = case.get('missing_titles', [])
            
            print(f"Processing {i+1}/{len(regressions)}: {question[:50]}...")
            
            log_file.write(f"Case {i+1}:\n")
            log_file.write(f"Question: {question}\n")
            log_file.write(f"Gold Answer: {gold_answer}\n")
            log_file.write(f"Missing Titles: {missing_titles}\n")
            log_file.write("-" * 40 + "\n")
            log_file.flush()
            
            try:
                print(f"   Invoking pipeline...")
                result = await pipeline.process_question(question)
                print(f"   Pipeline finished. Success: {result['success']}")
                
                if not result['success']:
                    log_file.write(f"Pipeline Error: {result.get('error')}\n\n")
                    log_file.flush()
                    continue
                
                decomposition = result['decomposition']
                # Handle both dict and object decomposition if necessary, but result['decomposition'] is usually a dict from to_dict()
                
                # If it's a dict (from pipeline.process_question return value)
                subquestions = decomposition.get('subquestions', [])
                
                for sq in subquestions:
                    log_file.write(f"  SQ: {sq['question']}\n")
                    log_file.write(f"  Answer: {sq['answer']}\n")
                    log_file.write(f"  Retrieved Passages:\n")
                    
                    retrieved = sq.get('retrieved_passages', [])
                    for idx, p in enumerate(retrieved):
                        title = p['title']
                        score = p.get('score', 0)
                        bm25 = p.get('bm25_score', 0)
                        dense = p.get('dense_score', 0)
                        path = p.get('matched_path', 'N/A')
                        
                        marker = " [MISSING]" if title in missing_titles else ""
                        
                        log_file.write(f"    {idx+1}. {title}{marker}\n")
                        log_file.write(f"       Score: {score:.4f} (BM25: {bm25:.4f}, Dense: {dense:.4f})\n")
                        log_file.write(f"       Path: {path}\n")
                    
                    log_file.write("\n")
                
                log_file.write(f"Final Prediction: {result['final_answer']}\n")
                
            except Exception as e:
                log_file.write(f"Exception: {str(e)}\n")
            
            log_file.write("\n" + "=" * 60 + "\n\n")
            log_file.flush()
            
    print(f"Analysis complete. Log saved to {log_path}")

if __name__ == "__main__":
    asyncio.run(run_debug())
