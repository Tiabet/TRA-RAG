"""
Align sampled HotpotQA QA items with original hotpot.jsonl entries, preserving context and supporting_facts.
Outputs:
 - Result/Samples/HotpotQA_aligned_200.json  (list of full original examples matched to sample order)
 - Result/Samples/HotpotQA_alignment_summary.json

Matching strategy:
 - exact normalized question match (case/whitespace normalized)
 - fallback: use difflib to find the closest question (ratio >= 0.80)
 - fallback 2: match by identical answer (normalized)
"""
from pathlib import Path
import json
import difflib
from collections import defaultdict

ROOT = Path(__file__).parent
SAMPLES = ROOT.parent / 'Result' / 'Samples' / 'HotpotQA_sample_200.json'
HOTPOT = ROOT / 'hotpot.jsonl'
OUT_ALIGNED = ROOT.parent / 'Result' / 'Samples' / 'HotpotQA_aligned_200.json'
OUT_SUM = ROOT.parent / 'Result' / 'Samples' / 'HotpotQA_alignment_summary.json'


def norm(s):
    if s is None:
        return ''
    return ' '.join(str(s).lower().strip().split())


def load_hotpot(path):
    data = []
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception as e:
                # skip malformed
                print('Warning: malformed hotpot line skipped:', e)
    return data


def main():
    samples = json.loads(SAMPLES.read_text(encoding='utf-8'))
    origs = load_hotpot(HOTPOT)
    print('Loaded samples:', len(samples), 'orig hotpot entries:', len(origs))

    # build index by normalized question
    q_to_entries = defaultdict(list)
    all_questions = []
    for obj in origs:
        q = obj.get('question') or obj.get('query')
        nq = norm(q)
        q_to_entries[nq].append(obj)
        all_questions.append(q or '')

    aligned = []
    summary = {'total_samples': len(samples), 'matched_exact':0, 'matched_fuzzy':0, 'matched_answer':0, 'unmatched':0, 'unmatched_items':[]}

    # prepare list of normalized questions for difflib matching
    questions_for_search = list(q_to_entries.keys())

    for s in samples:
        sq = s.get('query') or s.get('question') or ''
        ans = s.get('answer')
        nsq = norm(sq)
        matched_obj = None
        match_type = None

        # 1) exact normalized question match
        if nsq in q_to_entries:
            matched_obj = q_to_entries[nsq][0]
            match_type = 'exact'
            summary['matched_exact'] += 1
        else:
            # 2) fuzzy match using difflib on normalized question keys
            # find best close normalized question
            candidates = difflib.get_close_matches(nsq, questions_for_search, n=1, cutoff=0.80)
            if candidates:
                cand = candidates[0]
                matched_obj = q_to_entries[cand][0]
                match_type = 'fuzzy'
                summary['matched_fuzzy'] += 1
            else:
                # 3) try matching by answer text (normalized)
                if ans is not None:
                    nans = norm(ans)
                    # scan origs for equal answer
                    for obj in origs:
                        oans = norm(obj.get('answer'))
                        if oans and oans == nans:
                            matched_obj = obj
                            match_type = 'answer'
                            summary['matched_answer'] += 1
                            break

        if matched_obj is None:
            summary['unmatched'] += 1
            summary['unmatched_items'].append({'sample_query': sq, 'sample_answer': ans})
            # as fallback, append a minimal object containing sample fields
            fallback = {'question': sq, 'answer': ans, 'context': [], 'supporting_facts': []}
            aligned.append({'_aligned': False, 'sample': s, 'original': fallback})
        else:
            aligned.append({'_aligned': True, 'match_type': match_type, 'sample': s, 'original': matched_obj})

    # write outputs
    # write aligned originals in order (but include match metadata)
    with OUT_ALIGNED.open('w', encoding='utf-8') as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    with OUT_SUM.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Wrote aligned:', OUT_ALIGNED)
    print('Wrote summary:', OUT_SUM)
    print('Summary:', summary)

if __name__ == '__main__':
    main()
