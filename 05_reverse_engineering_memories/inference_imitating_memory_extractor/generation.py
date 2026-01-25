'''
This script is to check the generation of each LKEs to evaluate the exact matching accuracy.
'''
import yaml
import os
import json
import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import math

from .construct_prompt import ConstructPrompt
from ..util_public.inference.vllm.vllm_inference import VllmInference
from ..util_public.get_model_path import ModelPath

import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# ---------- simple metric helpers (no extra deps) ----------

def _simple_tokenize(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    # Treat NaN and other non-string values safely
    if not isinstance(text, str):
        # Handle float NaN explicitly
        try:
            if isinstance(text, float) and math.isnan(text):
                return []
        except Exception:
            pass
        text = str(text)
    return [t for t in re.findall(r"\w+", text.lower()) if t]

def _lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if n > 0 else Counter()

def rouge_l_f1(pred: str, ref: str) -> float:
    p_tokens = _simple_tokenize(pred)
    r_tokens = _simple_tokenize(ref)
    if not p_tokens or not r_tokens:
        return 0.0
    lcs = _lcs_length(p_tokens, r_tokens)
    prec = lcs / len(p_tokens)
    rec = lcs / len(r_tokens)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def _rouge_n_f1(pred: str, ref: str, n: int) -> float:
    p = _simple_tokenize(pred)
    r = _simple_tokenize(ref)
    if len(p) < n or len(r) < n:
        return 0.0
    p_ng = _ngram_counts(p, n)
    r_ng = _ngram_counts(r, n)
    overlap = sum((p_ng & r_ng).values())
    prec = overlap / max(sum(p_ng.values()), 1)
    rec = overlap / max(sum(r_ng.values()), 1)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def rouge_1_f1(pred: str, ref: str) -> float:
    return _rouge_n_f1(pred, ref, 1)

def rouge_2_f1(pred: str, ref: str) -> float:
    return _rouge_n_f1(pred, ref, 2)

def token_f1(pred: str, ref: str) -> Dict[str, float]:
    p = _simple_tokenize(pred)
    r = _simple_tokenize(ref)
    if not p and not r:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    precision = overlap / len(p) if p else 0.0
    recall = overlap / len(r) if r else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

def bleu4(pred: str, ref: str) -> float:
    """
    Corpus-agnostic, single-pair BLEU-4 with uniform weights and brevity penalty.
    Matches the standard formula, no external libraries.
    """
    p = _simple_tokenize(pred)
    r = _simple_tokenize(ref)
    if not p:
        return 0.0
    # modified n-gram precisions
    precisions = []
    for n in range(1, 5):
        p_ng = _ngram_counts(p, n)
        r_ng = _ngram_counts(r, n)
        if not p_ng:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, r_ng.get(ng, 0)) for ng, count in p_ng.items())
        precisions.append(overlap / sum(p_ng.values()))
    # geometric mean of precisions
    # if any precision is 0, BLEU is 0 (no smoothing here)
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    # brevity penalty
    ref_len = len(r)
    pred_len = len(p)
    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / pred_len)
    return float(bp * geo_mean)

def _rouge_n_prf(pred: str, ref: str, n: int) -> Dict[str, float]:
    p = _simple_tokenize(pred)
    r = _simple_tokenize(ref)
    if len(p) < n or len(r) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p_ng = _ngram_counts(p, n)
    r_ng = _ngram_counts(r, n)
    overlap = sum((p_ng & r_ng).values())
    prec = overlap / max(sum(p_ng.values()), 1)
    rec  = overlap / max(sum(r_ng.values()), 1)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}

def rouge1_prf(pred: str, ref: str) -> Dict[str, float]:
    return _rouge_n_prf(pred, ref, 1)

def rouge2_prf(pred: str, ref: str) -> Dict[str, float]:
    return _rouge_n_prf(pred, ref, 2)

def rougeL_prf(pred: str, ref: str) -> Dict[str, float]:
    p_tokens = _simple_tokenize(pred)
    r_tokens = _simple_tokenize(ref)
    if not p_tokens or not r_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(p_tokens, r_tokens)
    prec = lcs / len(p_tokens)
    rec  = lcs / len(r_tokens)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}

def safe_bertscore(pred: str, ref: str) -> Dict[str, float]:
    """
    Returns {'precision','recall','f1'} using bert-score if available, else zeros.
    Keeping it single-pair to avoid loading overhead for the whole corpus.
    Default to use  bert-base-uncased model
    """
    print('skip bertscore now')
    # P, R, F1 = score([pred], [ref], lang='en', verbose=True)
    # print(f"BERTScore P: {P}, R: {R}, F1: {F1}")
    # return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---------- core pipeline ----------

class Generation:
    '''
    Implementation based on vLLM, check the next 50 tokens as the model response.
    '''
    def __init__(self, config_path):
        self.config = self.read_config(config_path)
        self.config_path = config_path
        self.model_name = self.config.get('model_name')
        self.model_len = self.config.get('max_model_len')
        self.model_path = ModelPath(config_path).get_model_path()
        self.tokenizer_path = ModelPath(config_path).get_tokenizer_path()
        self.test_dataset_name = self.config.get('test_dataset_name')
        self.random_seed = self.config.get('random_seed')
        self.open_content = self.config.get('open_content')
        self.save_path = self.config.get('save_path')
        self.chat_template = self.config.get('chat_template')
    
    def read_config(self, path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def inference(self):
        prompt_constructor = ConstructPrompt(self.config_path)
        base_prompts, grouped_targets, user_ids, conversion_ids, message_ids = prompt_constructor.construct_prompt()
        queries, context, memory, personal_data, rephrased_message = grouped_targets
        inputs = []
        # filter out prompts that are too long
        for i, p in enumerate(base_prompts):
            tokenized = prompt_constructor.load_tokenizer()(p, return_tensors="pt", truncation=True, max_length=self.model_len)
            if tokenized['input_ids'].shape[1] < self.model_len - self.config.get('max_new_tokens', 200):
                inputs.append(p)
            else:
                inputs.append(f"Input {i} too long, no generation.")

        inference_agent = VllmInference(self.config_path)
        model, sampling_params = inference_agent.load_model()
        outputs = model.generate(inputs, sampling_params)

        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)
        return responses, inputs, queries, context, memory, personal_data, rephrased_message, user_ids, conversion_ids, message_ids

    def get_generation_acc(
        self,
        evaluation_type: str,
        responses: List[str],
        inputs: List[str],
        ground_truth: List[str],
        queries: List[str],
        user_ids: List[str],
        conversation_ids: List[str],
        message_ids: List[str], 
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        per_example: List[Dict[str, Any]] = []

        # Running sums for averages (GT and Query)
        agg = {
            "gt": {"r1_p": [], "r1_r": [], "r1_f": [], "r2_p": [], "r2_r": [], "r2_f": [], "rL_p": [], "rL_r": [], "rL_f": [],
                "tok_p": [], "tok_r": [], "tok_f": [], "bleu": [], "fuzz": [], "bert_p": [], "bert_r": [], "bert_f": [],
                "em": []},
            "q":  {"r1_p": [], "r1_r": [], "r1_f": [], "r2_p": [], "r2_r": [], "r2_f": [], "rL_p": [], "rL_r": [], "rL_f": [],
                "tok_p": [], "tok_r": [], "tok_f": [], "bleu": [], "fuzz": [], "bert_p": [], "bert_r": [], "bert_f": [],
                "em": []},
        }

        def _score_pair(pred, ref) -> Dict[str, Any]:
            # Coerce to safe strings with NaN handling
            def _to_text(x):
                try:
                    if x is None:
                        return ""
                    if isinstance(x, float):
                        import math
                        if math.isnan(x):
                            return ""
                    return x if isinstance(x, str) else str(x)
                except Exception:
                    return ""
            pred = _to_text(pred)
            ref  = _to_text(ref)

            r1 = rouge1_prf(pred, ref)
            r2 = rouge2_prf(pred, ref)
            rL = rougeL_prf(pred, ref)
            tf = token_f1(pred, ref)
            rf = fuzz.ratio(pred, ref) / 100.0
            bl = bleu4(pred, ref)
            bs = safe_bertscore(pred, ref)
            em = (pred.strip().lower() == ref.strip().lower())
            return {
                "rouge1": r1, "rouge2": r2, "rougeL": rL,
                "token": tf, "bleu4": bl, "fuzz_ratio": rf,
                "bertscore": bs, "exact_match": em
            }


        scored = 0
        for i, (resp, prompt, gt, q) in enumerate(zip(responses, inputs, ground_truth, queries)):
            # Skip if GT explicitly "no memory" like your current behavior
            if isinstance(gt, str) and gt.strip().lower() == "no memory":
                per_example.append({
                    "index": i,
                    'user_id': user_ids[i],
                    'conversation_id': conversation_ids[i],
                    'message_id': message_ids[i],
                    'message_id': message_ids[i],
                    "prompt": prompt,
                    "query": q,
                    "response": resp,
                    "ground_truth": gt,
                    "metrics": None,
                    "skipped_reason": "no memory"
                })
                continue

            m_gt = _score_pair(resp, gt)
            m_q  = _score_pair(resp, q)

            # aggregate
            for tag, metrics in [("gt", m_gt), ("q", m_q)]:
                agg[tag]["r1_p"].append(metrics["rouge1"]["precision"])
                agg[tag]["r1_r"].append(metrics["rouge1"]["recall"])
                agg[tag]["r1_f"].append(metrics["rouge1"]["f1"])
                agg[tag]["r2_p"].append(metrics["rouge2"]["precision"])
                agg[tag]["r2_r"].append(metrics["rouge2"]["recall"])
                agg[tag]["r2_f"].append(metrics["rouge2"]["f1"])
                agg[tag]["rL_p"].append(metrics["rougeL"]["precision"])
                agg[tag]["rL_r"].append(metrics["rougeL"]["recall"])
                agg[tag]["rL_f"].append(metrics["rougeL"]["f1"])
                agg[tag]["tok_p"].append(metrics["token"]["precision"])
                agg[tag]["tok_r"].append(metrics["token"]["recall"])
                agg[tag]["tok_f"].append(metrics["token"]["f1"])
                agg[tag]["bleu"].append(metrics["bleu4"])
                agg[tag]["fuzz"].append(metrics["fuzz_ratio"])
                agg[tag]["bert_p"].append(metrics["bertscore"]["precision"])
                agg[tag]["bert_r"].append(metrics["bertscore"]["recall"])
                agg[tag]["bert_f"].append(metrics["bertscore"]["f1"])
                agg[tag]["em"].append(1 if metrics["exact_match"] else 0)

            per_example.append({
                "index": i,
                "user_id": user_ids[i],
                "conversation_id": conversation_ids[i],
                "message_id": message_ids[i],
                "prompt": prompt,
                "query": q,
                "response": resp,
                "ground_truth": gt,
                "metrics": {
                    "vs_ground_truth": m_gt,
                    "vs_query": m_q
                }
            })
            scored += 1

        def _avg(arr): return float(np.mean(arr)) if arr else None

        summary = {
            "num_examples_total": len(responses),
            "num_examples_scored": scored,
            "num_examples_skipped": len(responses) - scored,

            # Averages vs Ground Truth
            "avg_vs_ground_truth": {
                "rouge1": {"precision": _avg(agg["gt"]["r1_p"]), "recall": _avg(agg["gt"]["r1_r"]), "f1": _avg(agg["gt"]["r1_f"])},
                "rouge2": {"precision": _avg(agg["gt"]["r2_p"]), "recall": _avg(agg["gt"]["r2_r"]), "f1": _avg(agg["gt"]["r2_f"])},
                "rougeL": {"precision": _avg(agg["gt"]["rL_p"]), "recall": _avg(agg["gt"]["rL_r"]), "f1": _avg(agg["gt"]["rL_f"])},
                "token_f1": {"precision": _avg(agg["gt"]["tok_p"]), "recall": _avg(agg["gt"]["tok_r"]), "f1": _avg(agg["gt"]["tok_f"])},
                "bleu4": _avg(agg["gt"]["bleu"]),
                "fuzz_ratio": _avg(agg["gt"]["fuzz"]),
                "bertscore": {"precision": _avg(agg["gt"]["bert_p"]), "recall": _avg(agg["gt"]["bert_r"]), "f1": _avg(agg["gt"]["bert_f"])},
                "exact_match_rate": _avg(agg["gt"]["em"]),
            },

            # Averages vs Query
            "avg_vs_query": {
                "rouge1": {"precision": _avg(agg["q"]["r1_p"]), "recall": _avg(agg["q"]["r1_r"]), "f1": _avg(agg["q"]["r1_f"])},
                "rouge2": {"precision": _avg(agg["q"]["r2_p"]), "recall": _avg(agg["q"]["r2_r"]), "f1": _avg(agg["q"]["r2_f"])},
                "rougeL": {"precision": _avg(agg["q"]["rL_p"]), "recall": _avg(agg["q"]["rL_r"]), "f1": _avg(agg["q"]["rL_f"])},
                "token_f1": {"precision": _avg(agg["q"]["tok_p"]), "recall": _avg(agg["q"]["tok_r"]), "f1": _avg(agg["q"]["tok_f"])},
                "bleu4": _avg(agg["q"]["bleu"]),
                "fuzz_ratio": _avg(agg["q"]["fuzz"]),
                "bertscore": {"precision": _avg(agg["q"]["bert_p"]), "recall": _avg(agg["q"]["bert_r"]), "f1": _avg(agg["q"]["bert_f"])},
                "exact_match_rate": _avg(agg["q"]["em"]),
            }
        }
        
        def json_to_csv_per_user(
            evaluation_type: str,
            per_example: List[Dict[str, Any]], 
            model_name: str = self.model_name,
            test_dataset_name: str = self.test_dataset_name,
            save_path: str = '.../result/ft_csv',
        ) -> None:
            """
            Save per-example results to CSV, grouped by user_id.
            """
            df = pd.DataFrame(per_example)
            
            if 'user_id' in df.columns:
                user_ids = df['user_id'].unique()
                for user_id in user_ids:
                    user_df = df[df['user_id'] == user_id]
                    # order the rows by 'Create Time'
                    user_df = user_df.sort_values(by='conversation_id')
                    user_save_path = os.path.join(save_path, f'results_user-{user_id}_{test_dataset_name}_{model_name}_{evaluation_type}.csv')
                    user_df.to_csv(user_save_path, index=False)
            else:
                print(df)
                raise ValueError("user_id column not found in per_example data.")
            return None
        if self.open_content:
            save_dir = f'.../result/icl_csv/{self.model_name}'
        else:
            save_dir =  f'.../result/ft_csv/{self.model_name}'
        os.makedirs(save_dir, exist_ok=True)

        json_to_csv_per_user(evaluation_type, per_example,model_name=self.model_name, test_dataset_name=self.test_dataset_name, save_path=save_dir)

        return per_example, summary


    def save_outputs(self, evaluation_type,per_example: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
        """
        Save per-example results and summary to JSON. Returns the output path.
        """
        save_dir = '.../result/raw/'
        os.makedirs(save_dir, exist_ok=True)

        payload = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": self.model_name,
                "model_path": self.model_path,
                "test_dataset_name": self.test_dataset_name,
                "chat_template": self.chat_template,
                "random_seed": self.random_seed,
                "open_example": self.open_content,
                "config_path": self.config_path,
            },
            "summary": summary,
            "results": per_example
        }

        fname = f"{self.test_dataset_name or 'dataset'}_e-{self.open_content}__{self.model_name or 'model'}_{evaluation_type}.json"
        out_path = os.path.join(save_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

def parse_response(response: Optional[str]) -> Tuple[str, str, str]:
    """
    Extract:
      - memory/key information
      - personal data
      - rephrased user query

    Handles JSON and plain-text with various label synonyms.
    Returns empty strings when missing.
    """
    text = (response or "").strip()
    if not text:
        return "", "", ""

    MEMORY_KEYS = {
        "key information", "memory", "relevant memory", "generated memory",
        "key info", "key_information", "relevant_memory"
    }
    PERSONAL_KEYS = {
        "personal data", "personal information", "personal info",
        "personal_data", "personal_information", "special category personal data"
    }
    REPHRASED_KEYS = {
        "rephrased query", "rephrased user query", "rephrased message",
        "paraphrased query", "reworded query", "reformulated query",
        "rephrased_user_query", "rephrased_message"
    }

    def tidy(s: str) -> str:
        s = (s or "").strip()
        # remove fenced code wrappers if present
        s = re.sub(r"^```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
        # strip balanced quotes
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s

    def try_parse_json(txt: str):
        fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
        m = fenced or re.search(r"(\{.*\})", txt, flags=re.DOTALL)
        if not m:
            return None
        norm = (m.group(1)
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
        norm = re.sub(r",\s*([}\]])", r"\1", norm)  # remove trailing commas
        try:
            return json.loads(norm)
        except json.JSONDecodeError:
            tmp = re.sub(r"'", '"', norm)
            tmp = re.sub(r",\s*([}\]])", r"\1", tmp)
            try:
                return json.loads(tmp)
            except Exception:
                return None

    memory = ""
    personal = ""
    rephrased = ""

    # 1) JSON path
    obj = try_parse_json(text)
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.strip().lower().replace("-", " ").replace("_", " ")
            val = tidy(str(v))
            if lk in MEMORY_KEYS and not memory:
                memory = val
            elif lk in PERSONAL_KEYS and not personal:
                personal = val
            elif lk in REPHRASED_KEYS and not rephrased:
                rephrased = val
        if memory or personal or rephrased:
            return memory, personal, rephrased

    # 2) Plain-text labels
    all_labels = sorted(MEMORY_KEYS | PERSONAL_KEYS | REPHRASED_KEYS, key=len, reverse=True)
    label_alt = "|".join(re.escape(x) for x in all_labels)

    pattern = rf"""
        ^\s*(?:[-*]|\d+\.)?\s*
        (?:\*\*|__|`|")?
        (?P<label>{label_alt})
        (?:\*\*|__|`|")?
        \s*:\s*
        (?P<value>.*?)
        (?=(?:\n\s*(?:[-*]|\d+\.)?\s*(?:\*\*|__|`|")?(?:{label_alt})(?:\*\*|__|`|")?\s*:)|\Z)
    """
    regex = re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE)

    for m in regex.finditer(text):
        lab = m.group("label").strip().lower().replace("-", " ").replace("_", " ")
        val = tidy(m.group("value"))
        if lab in MEMORY_KEYS and not memory:
            memory = val
        elif lab in PERSONAL_KEYS and not personal:
            personal = val
        elif lab in REPHRASED_KEYS and not rephrased:
            rephrased = val

    return memory, personal, rephrased

# ---------- entrypoint ----------

def main(
    config_path,
):
    generation = Generation(config_path)
    responses, inputs, queries, context, memory, personal_data, rephrased_message, user_ids, conversation_ids = generation.inference()
    
    response_memory, response_personal_data, response_rephrased_message = [], [], []
    for resp in responses:
        mem, pdata, rmsg = parse_response(resp)
        response_memory.append(mem)
        response_personal_data.append(pdata)
        response_rephrased_message.append(rmsg)
    
    ground_truth_memory = memory
    ground_truth_personal_data = personal_data
    ground_truth_rephrased_message = rephrased_message

    per_example_memory, summary_memory = generation.get_generation_acc(
        evaluation_type="memory",
        responses=response_memory,
        inputs=inputs,
        ground_truth=ground_truth_memory,
        queries=queries,
        user_ids=user_ids,
        conversation_ids=conversation_ids
    )

    per_example_personal_data, summary_personal_data = generation.get_generation_acc(
        evaluation_type="personal_data",
        responses=response_personal_data,
        inputs=inputs,
        ground_truth=ground_truth_personal_data,
        queries=queries,
        user_ids=user_ids,
        conversation_ids=conversation_ids
    )

    per_example_rephrased_message, summary_rephrased_message = generation.get_generation_acc(
        evaluation_type="rephrased_message",
        responses=response_rephrased_message,
        inputs=inputs,
        ground_truth=ground_truth_rephrased_message,
        queries=queries,
        user_ids=user_ids,
        conversation_ids=conversation_ids
    )   

    per_example_all, summary_all = generation.get_generation_acc(
        evaluation_type="all",
        responses=responses,
        inputs=inputs,
        ground_truth=[f"Memory: {m}\nPersonal Data: {p}\nRephrased User Query: {r}" for m, p, r in zip(ground_truth_memory, ground_truth_personal_data, ground_truth_rephrased_message)],
        queries=queries,
        user_ids=user_ids,
        conversation_ids=conversation_ids
    )
    out_path_memory = generation.save_outputs("memory", per_example_memory, summary_memory)
    out_path_personal_data = generation.save_outputs("personal_data", per_example_personal_data, summary_personal_data)
    out_path_rephrased_message = generation.save_outputs("rephrased_message", per_example_rephrased_message, summary_rephrased_message)
    out_path_all = generation.save_outputs("all", per_example_all, summary_all)

    print(f"Saved memory results to: {out_path_memory}")
    print(f"Saved personal data results to: {out_path_personal_data}")
    print(f"Saved rephrased message results to: {out_path_rephrased_message}")
    print(f"Saved all results to: {out_path_all}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    main(config_path=args.config_path)
