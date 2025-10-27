#!/usr/bin/env python
# evaluate_kgrag_hardcoded.py
import argparse
import json, re, string
from collections import Counter
from pathlib import Path

# Default file paths (will try sensible fallbacks)
DEFAULT_PRED_CANDIDATES = [
    Path("Results/multihop_pipeline_200_results.json"),
    Path("Results/multihop_pipeline_200_checkpoint.json"),
    Path("NaiveRAG/hotpot_result.json")
]
DEFAULT_GOLD = Path("HotpotQA/qa.json")

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
def load_pairs(path: Path, answer_key: str):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
        # Support several shapes:
        # 1) dict with 'results' list (each: question, gold_answer, predicted_answer)
        # 2) list of {query, answer/result}
        # 3) dict keyed by query -> answer
        if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
            mapping = {}
            for item in data['results']:
                q = item.get('question') or item.get('query') or item.get('question_id')
                if not q:
                    continue
                # Prefer explicit key, else try common alternatives
                if answer_key in item:
                    a = item[answer_key]
                else:
                    # When loading predictions, prefer predicted fields first
                    a = item.get('predicted_answer') or item.get('result') or item.get('answer') or item.get('gold_answer')
                mapping[q] = a
            return mapping

        if isinstance(data, dict):
            # assume mapping from query -> answer
            return data

        # list of dicts
        return {d.get('query') or d.get('question'): d.get(answer_key) or d.get('answer') or d.get('result') for d in data}

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions with EM/F1/Accuracy metrics")
    parser.add_argument('--pred', type=Path, default=None, help='Predictions JSON file (default: first existing candidate)')
    parser.add_argument('--gold', type=Path, default=DEFAULT_GOLD, help='Gold JSON file')
    parser.add_argument('--answer-key', type=str, default='answer', help='Key name for gold answers')
    parser.add_argument('--pred-key', type=str, default='result', help='Key name for predictions')
    parser.add_argument('--out', type=Path, default=None, help='Optional JSON output file to write aggregated metrics')
    args = parser.parse_args()

    # Resolve prediction path: use provided or first existing candidate
    pred_path = args.pred
    if pred_path is None:
        for cand in DEFAULT_PRED_CANDIDATES:
            if cand.exists():
                pred_path = cand
                break
    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"No predictions file found. Tried: {DEFAULT_PRED_CANDIDATES}")

    gold = load_pairs(args.gold, args.answer_key)
    pred = load_pairs(pred_path, args.pred_key)

    em_sum = f1_sum = precision_sum = recall_sum = 0
    ### --- NEW --- ###
    contain_correct = 0          # Accuracy용 카운터
    ### ------------- ###
    missing = 0

    for q, gold_ans in gold.items():
        if q not in pred or "[Error]" in pred[q]:
            missing += 1
            continue
        pred_ans = pred[q]
        em, f1_val, prec, rec = compute_metrics(pred_ans, gold_ans)
        em_sum += em
        f1_sum += f1_val
        precision_sum += prec
        recall_sum += rec
        ### --- NEW --- ###
        # 정답 문자열이 예측 안에 '포함'되어 있으면 correct
        if normalize(gold_ans) in normalize(pred_ans):
            contain_correct += 1
        ### ------------- ###

    compared = len(gold) - missing
    em         = em_sum         / compared if compared else 0
    f1         = f1_sum         / compared if compared else 0
    precision  = precision_sum  / compared if compared else 0
    recall     = recall_sum     / compared if compared else 0
    accuracy   = contain_correct / compared if compared else 0
    ### ------------- ###

    print(f"#items compared : {compared}/{len(gold)} (missing={missing})")
    print(f"Exact‑Match     : {em:.3f}")
    print(f"F1              : {f1:.3f}")
    print(f"Precision       : {precision:.3f}")
    print(f"Recall          : {recall:.3f}")
    print(f"Accuracy        : {accuracy:.3f}  (gold answer ⊆ prediction)")
    ### ------------- ###

    if args.out:
        out_data = {
            'compared': compared,
            'missing': missing,
            'exact_match': em,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', encoding='utf-8') as fo:
            json.dump(out_data, fo, ensure_ascii=False, indent=2)
        print(f"Wrote summary to {args.out}")

if __name__ == "__main__":
    main()
