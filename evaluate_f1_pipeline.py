#!/usr/bin/env python
"""
F1 Score Evaluation for Pipeline Results
==========================================
Evaluates test_new_pipeline_200_results_v2.json using F1 metrics
"""
import json, re, string
from collections import Counter
from pathlib import Path

# ---------- 하드코딩된 파일 경로 ----------
PRED_PATH = Path("Results/ablation_dense_200_results.json")

# ---------- text normalization ----------
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

# ---------- metrics ----------
def compute_metrics(pred: str, gold: str):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        em = int(pred_tokens == gold_tokens)
        return em, 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    em = int(pred_tokens == gold_tokens)

    return em, f1, precision, recall

# ---------- driver ----------
def main():
    print(f"Loading: {PRED_PATH}")
    
    with PRED_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    print(f"Total results: {len(results)}")

    em_sum = f1_sum = precision_sum = recall_sum = 0
    contain_correct = 0  # Accuracy용 카운터
    missing = 0

    for item in results:
        gold_ans = item.get("gold_answer", "")
        pred_ans = item.get("predicted_answer", "")
        
        if not pred_ans or "Insufficient information" in pred_ans:
            # Still evaluate but may get 0 score
            pass
        
        em, f1_val, prec, rec = compute_metrics(pred_ans, gold_ans)
        em_sum += em
        f1_sum += f1_val
        precision_sum += prec
        recall_sum += rec
        
        # 정답 문자열이 예측 안에 '포함'되어 있으면 correct
        if normalize(gold_ans) in normalize(pred_ans):
            contain_correct += 1

    compared = len(results)
    em         = em_sum         / compared if compared else 0
    f1         = f1_sum         / compared if compared else 0
    precision  = precision_sum  / compared if compared else 0
    recall     = recall_sum     / compared if compared else 0
    accuracy   = contain_correct / compared if compared else 0

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"#items compared : {compared}")
    print(f"Exact-Match     : {em:.3f} ({int(em_sum)}/{compared})")
    print(f"F1              : {f1:.3f}")
    print(f"Precision       : {precision:.3f}")
    print(f"Recall          : {recall:.3f}")
    print(f"Accuracy        : {accuracy:.3f} ({contain_correct}/{compared}) (gold ⊆ prediction)")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
