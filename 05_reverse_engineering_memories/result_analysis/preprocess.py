'''
This file is used to compute evaluation metrics such as BLEU, ROUGE, and BERTScorewenxin zhou usc
for the predicted memories or rephrased messages against the ground truth, query, and context.

This is a preprocessing step before human evaluation to get a quantitative sense of model performance.
'''

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from bert_score import score
from .generation import _rouge_n_prf

from collections import Counter

def unigram_precision_recall(candidate: str, reference: str):
    """
    Compute unigram precision and recall for BLEU.
    
    Args:
        candidate (str): Candidate text (system output).
        reference (str): Reference text (ground truth).
    
    Returns:
        precision (float), recall (float)
    """
    # print(f"Computing unigram precision/recall for candidate: {candidate}\n and reference: \n{reference}")
    # Tokenize (naively split on whitespace)
    cand_tokens = candidate.strip().split()
    ref_tokens = reference.strip().split()
    
    # Count unigrams
    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)
    
    # Overlap between candidate and reference
    overlap = sum(min(cand_counts[w], ref_counts[w]) for w in cand_counts)
    
    # Precision = overlap / candidate length
    precision = overlap / len(cand_tokens) if cand_tokens else 0.0
    
    # Recall = overlap / reference length
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    
    return precision, recall



def safe_bertscore(pred: str, ref: str) -> Dict[str, float]:
    """
    Returns {'precision','recall','f1'} using bert-score if available, else zeros.
    Keeping it single-pair to avoid loading overhead for the whole corpus.
    Default to use bert-base-uncased model.
    """
    P, R, F1 = score([pred], [ref], model_type='xlm-roberta-large', verbose=True)
    # P, R, F1 = np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    # print(f"BERTScore P: {P}, R: {R}, F1: {F1}")
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean())
    }

def get_all_metrics(
    model_name: str,
    test_dataset_name: str,
    predict_data: str,
    open_content: Optional[bool] = False,
    user_ids: List[int] = [0, 1, 4, 8, 9],
    write_back: bool = True,
):
    """
    Computes BLEU-4 (as unigram precision/recall), ROUGE-L (P/R/F1), and BERTScore (P/R/F1)
    for each sample, comparing predictions against (a) ground truth, (b) query, and (c) context.
    - Writes per-sample metrics back into the original CSV. Only computes missing values.
    - If a row has no ground truth (or 'No memory' when predict_data=='memory'), writes 'N/A'
      for *all* metrics on that row and excludes it from the summary.
    - Returns a macro-averaged summary over users (simple mean over valid rows).

    Returns
    -------
    pd.DataFrame
        Index: ['ground_truth', 'query', 'context']
        Columns:
            'count',
            'bleu4_p','bleu4_r',
            'rougeL_p','rougeL_r','rougeL_f1',
            'bert_precision','bert_recall','bert_f1'
    """
    # -------- internal helpers --------
    METRIC_KEYS = [
        'bleu4_p','bleu4_r',
        'rougeL_p','rougeL_r','rougeL_f1',
        'bert_precision','bert_recall','bert_f1'
    ]
    REFS = ['ground_truth', 'query', 'context']

    def col_name(ref: str, metric: str) -> str:
        # Namespaced columns per reference to avoid collisions
        return f'{ref}__{metric}'

    def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
        for ref in REFS:
            for m in METRIC_KEYS:
                cn = col_name(ref, m)
                if cn not in df.columns:
                    df[cn] = np.nan
        # a convenience flag to know we processed this row (optional)
        if 'metrics_status' not in df.columns:
            df['metrics_status'] = ''
        return df

    def rouge_l_prf_fallback(pred: str, ref: str) -> Tuple[float, float, float]:
        """ROUGE-L P/R/F via LCS fallback."""
        def lcs(a, b):
            n, m = len(a), len(b)
            dp = [[0]*(m+1) for _ in range(n+1)]
            for i in range(1, n+1):
                ai = a[i-1]
                for j in range(1, m+1):
                    if ai == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[n][m]

        a = pred.split()
        b = ref.split()
        if not a or not b:
            return 0.0, 0.0, 0.0
        l = lcs(a, b)
        prec = l / len(a)
        rec  = l / len(b)
        if prec + rec == 0:
            return 0.0, 0.0, 0.0
        f1 = 2 * prec * rec / (prec + rec)
        return float(prec), float(rec), float(f1)

    def safe_rouge_prf(pred: str, ref: str, n):
        """
        Call _rouge_n_prf(pred, ref, n) and return (p, r, f).
        Use LCS fallback for ROUGE-L if needed; otherwise return zeros on error.
        """
        try:
            p, r, f = _rouge_n_prf(pred, ref, n=n)
            return float(p), float(r), float(f)
        except Exception:
            if str(n).lower() == 'l':
                return rouge_l_prf_fallback(pred, ref)
            return 0.0, 0.0, 0.0

    def compute_metrics(pred: str, ref: str) -> dict:
        """Compute all metrics for a single pred/ref. Returns dict keyed by METRIC_KEYS."""
        if not isinstance(pred, str) or not isinstance(ref, str) or not pred.strip() or not ref.strip():
            # Empty or invalid: return NaNs (caller can coerce to 'N/A' if desired)
            return {k: np.nan for k in METRIC_KEYS}

        # "BLEU-4" placeholder: using unigram precision/recall per original code
        bleu_p, bleu_r = unigram_precision_recall(pred, ref)

        # ROUGE-L
        rL_p, rL_r, rL_f = safe_rouge_prf(pred, ref, n='L')

        # BERTScore
        bert_scores = safe_bertscore(pred, ref)
        bert_p = float(bert_scores.get('precision', 0.0))
        bert_r = float(bert_scores.get('recall', 0.0))
        bert_f = float(bert_scores.get('f1', 0.0))

        return {
            'bleu4_p': float(bleu_p),
            'bleu4_r': float(bleu_r),
            'rougeL_p': float(rL_p),
            'rougeL_r': float(rL_r),
            'rougeL_f1': float(rL_f),
            'bert_precision': bert_p,
            'bert_recall': bert_r,
            'bert_f1': bert_f,
        }

    def write_na_row(df: pd.DataFrame, row_idx):
        """Write 'N/A' for all per-ref metric columns for this row."""
        for ref in REFS:
            for m in METRIC_KEYS:
                df.at[row_idx, col_name(ref, m)] = "N/A"
        df.at[row_idx, 'metrics_status'] = 'no_ground_truth'

    # -------- aggregation buckets (for summary over valid rows only) --------
    buckets = {
        'ground_truth': {k: [] for k in METRIC_KEYS},
        'query':        {k: [] for k in METRIC_KEYS},
        'context':      {k: [] for k in METRIC_KEYS},
    }

    # -------- iterate users --------
    for user_id in user_ids:
        if open_content:
            data_path = f'.../result/icl_csv/{model_name}/results_user-user_{user_id}_{test_dataset_name}_{model_name}_{predict_data}.csv'
        else:
            data_path = f'.../result/ft_csv/{model_name}/results_user-user_{user_id}_{test_dataset_name}_{model_name}_{predict_data}.csv'

        if not os.path.exists(data_path):
            print(f'File not found: {data_path}')
            continue

        df = pd.read_csv(data_path)
        df = ensure_metric_columns(df)

        # Row identifier for stability across saves
        row_id_col = 'index' if 'index' in df.columns else None
        if row_id_col is None:
            # materialize a stable id if not present
            df.insert(0, 'index', range(len(df)))
            row_id_col = 'index'

        # Collect fields with safe defaults
        preds   = df['response'].astype(str).tolist() if 'response' in df.columns else [''] * len(df)
        gts     = df['ground_truth'].astype(str).tolist() if 'ground_truth' in df.columns else [''] * len(df)
        queries = df['query'].astype(str).tolist() if 'query' in df.columns else [''] * len(df)
        prompts = df['prompt'].astype(str).tolist() if 'prompt' in df.columns else [''] * len(df)

        # extract context per your special end token rules
        if model_name.startswith('gpt'):
            special_end_token = '<|end|>'
        elif model_name.startswith('qwen2.5'):
            special_end_token = '<|im_end|>'
        elif model_name.startswith('gemma'):
            special_end_token = '<end_of_turn>'
        else:
            print('Wrong model name!')
            if write_back:
                # still save if we changed df (e.g., added cols)
                df.to_csv(data_path, index=False)
            continue

        contexts = []
        for p in prompts:
            p = str(p)
            if 'Context:' in p and special_end_token in p:
                try:
                    context = p.rsplit('Context:', 1)[-1].split(special_end_token)[0].strip()
                except Exception:
                    print(f'Error extracting context from prompt: {p}')
                    context = ''
            else:
                context = ''
            contexts.append(context)

        # per-row compute-or-skip
        for i, (pred, gt, q, c) in enumerate(zip(preds, gts, queries, contexts)):
            # detect "no ground truth"
            gt_is_missing = (
                (predict_data == 'memory' and str(gt).strip().lower() in ['no memory', 'na', 'nan', ''])
                or
                (predict_data == 'rephrased_message' and str(gt).strip().lower() in ['na', 'nan', ''])
                or
                pd.isna(gt)
            )
            if gt_is_missing:
                # write 'N/A' for all metrics on this row; exclude from summary
                write_na_row(df, i)
                continue

            # For each ref, compute only if all needed inputs exist and values are missing
            per_ref_inputs = {
                'ground_truth': gt,
                'query': q,
                'context': c,
            }

            for ref, ref_text in per_ref_inputs.items():
                # If reference text is empty for this ref, mark ref metrics as 'N/A' (but others can still exist)
                if not isinstance(ref_text, str) or ref_text.strip() == '':
                    for m in METRIC_KEYS:
                        cn = col_name(ref, m)
                        if pd.isna(df.at[i, cn]) or df.at[i, cn] == "N/A":
                            df.at[i, cn] = "N/A"
                    continue

                # Determine if this ref already has all metrics computed (not NaN and not 'N/A')
                # already_has_all = True
                # for m in METRIC_KEYS:
                #     val = df.at[i, col_name(ref, m)]
                #     if pd.isna(val) or val == "N/A":
                #         already_has_all = False
                #         break

                already_has_all = False
                # already_has_all = False
                if not already_has_all:
                    # compute metrics for this ref
                    mvals = compute_metrics(pred, ref_text)
                    for m, v in mvals.items():
                        cn = col_name(ref, m)
                        # If either pred/ref was empty we got NaN; store 'N/A' for consistency
                        df.at[i, cn] = v if not (isinstance(v, float) and np.isnan(v)) else "N/A"

            df.at[i, 'metrics_status'] = 'ok'

        # Save back to the same CSV
        if write_back:
            df.to_csv(data_path, index=False)

        # -------- Build aggregation input by reading back valid rows only --------
        # Valid means: has ground truth (per your rule) and non-empty prediction
        valid_mask = (
            df['response'].astype(str).str.strip().ne('') &
            df['ground_truth'].astype(str).str.strip().ne('') &
            (
                (predict_data != 'memory') |
                (df['ground_truth'].astype(str).str.strip().ne('No memory'))
            )
        )
        if not valid_mask.any():
            print(f'No valid data in: {data_path}')
            continue

        # Add to buckets (macro average) using the per-row stored values
        for ref in REFS:
            for _, row in df[valid_mask].iterrows():
                # Only add numeric values
                def _get_num(cname):
                    val = row[cname]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        return float(val)
                    # numeric strings
                    try:
                        return float(val)
                    except Exception:
                        return None

                for m in METRIC_KEYS:
                    v = _get_num(col_name(ref, m))
                    if v is not None:
                        buckets[ref][m].append(v)

    # -------- Build summary DataFrame with macro-averages --------
    rows = []
    for ref in REFS:
        metrics = buckets[ref]
        counts = {m: len(v) for m, v in metrics.items()}
        if counts['bleu4_p'] == 0:
            rows.append({
                'reference': ref,
                'count': 0,
                **{m: np.nan for m in METRIC_KEYS}
            })
            continue

        row = {'reference': ref, 'count': counts['bleu4_p']}
        for m in METRIC_KEYS:
            vals = metrics[m]
            row[m] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index('reference')
    return summary_df

if __name__ == "__main__":
    base_models = [
        'gpt-oss-20b',
        'qwen2.5-32b-instruct',
        'gemma-3-27b-it'
    ]
          # 'qwen2.5-32b-instruct' or 'gpt-oss-20b' or 'gemma-3-27b-it'
    train_dataset_names = [
        # 'query-all3',
        'query-all3-full-conversation',
        # 'all-user-limit',
    ]
    test_dataset_names = [
        # 'query-all3',
        # 'query-all3-full-conversation',
        # 'all-user-limit',
        'all-user-full'
    ]
      # or 'query-all3-full-conversation' or 'query-no-memory' or 'query-memory-train' or 'query-memory-test'
    checkpoint = 40
    predict_datas = ['memory', 'rephrased_message']  # 'rephrased_message' or 'memory'
    open_contents = [
        False,
        True
     ] # Whether to use open_content results
    # user_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 59, 60, 61, 63, 64, 65, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78]
    # flag = 'all'
    user_ids = [0, 1, 12, 13, 16, 27, 28, 29, 31, 32, 34, 35, 38, 39, 4, 42, 45, 46, 51, 54, 55, 56, 59, 63, 64, 65, 68, 70, 71, 72, 76, 77, 78, 8, 9]
    flag = 'english'
    # user_ids = [2, 3, 5, 6, 7, 10, 15, 17, 19, 21, 23, 25, 26, 30, 33, 36, 37, 40, 43, 44, 47, 48, 49, 53, 57, 60, 61]
    # flag = 'non-english'
    for base_model in base_models:
        for test_dataset_name in test_dataset_names:
            for train_dataset_name in train_dataset_names:
                for predict_data in predict_datas:
                    for open_content in open_contents:
                        print(f'Processing model: {base_model}, dataset: {test_dataset_name}, predict_data: {predict_data}, open_content: {open_content}')
                        if open_content:
                            model_name = base_model
                        else:
                            if train_dataset_name == 'all-user-full' or train_dataset_name == 'all-user-limit':
                                model_name = f'{base_model}_{train_dataset_name}_train-sft_ratio-1.0_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-{checkpoint}'
                            else:
                                model_name = f'{base_model}_{train_dataset_name}_train-sft_ratio-0.6_1_5e-05_cosine_chat-template_0-gacc_8_checkpoint-{checkpoint}'
                        # save to CSV, create dir first
                        if open_content:
                            output_dir = f'.../memory/result/summary/{flag}/xlm-roberta-large/{model_name}/icl/{test_dataset_name}/'
                        else:
                            output_dir = f'.../memory/result/summary/{flag}/xlm-roberta-large/{model_name}/ft/{test_dataset_name}/'
                        
                        os.makedirs(output_dir, exist_ok=True)

                        df = get_all_metrics(model_name, test_dataset_name, predict_data, open_content=open_content, user_ids=user_ids)
                        print(df)
                        output_path = os.path.join(output_dir, f'summary_{predict_data}.csv')
                        df.to_csv(output_path)
                        print(f'Saved summary metrics to {output_path}')
