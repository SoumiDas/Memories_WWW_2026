import gc
import os
import argparse
from typing import Optional

import torch
import yaml

# vLLM parallel state cleanup (for vllm >= 0.4.0)
try:
    from vllm.distributed.parallel_state import destroy_model_parallel  # type: ignore
except Exception:  # pragma: no cover
    def destroy_model_parallel():
        return None

from .generation import Generation, parse_response


def eval(
    config_path: str,
    model_name: str,
    test_dataset_name: str,
    chat_template: Optional[str] = None,
    max_model_len: int = 20000,
    max_new_tokens: int = 200,
    test_type: str = "close",
    open_content: Optional[int] = None,
) -> str:
    """
    Build a temp config from the provided args, run Generation, and save results for
    memory / personal_data / rephrased_message, matching the Generation class API.

    Returns:
        Path to the memory results JSON (other paths are printed).
    """
    # Load base config
    with open(config_path, "r", encoding="utf-8") as f:
        test_config = yaml.safe_load(f) or {}

    # Override fields to align with Generation.__init__ expectations
    test_config["model_name"] = model_name
    test_config["test_dataset_name"] = test_dataset_name
    test_config["max_model_len"] = max_model_len
    test_config["max_new_tokens"] = max_new_tokens
    test_config["chat_template"] = chat_template
    test_config["open_content"] = open_content
    test_config["test_type"] = test_type

    # Compose a safe, informative temp config filename
    safe_model_name = model_name
    if len(safe_model_name) > 200:
        head, tail = safe_model_name[:50], safe_model_name[-150:]
        safe_model_name = f"{head}...{tail}"

    tmp_dir = ".../_temp"
    os.makedirs(tmp_dir, exist_ok=True)
    config_file_name = f"{safe_model_name}_{test_dataset_name}_eval_config.yaml"
    new_config_path = os.path.join(tmp_dir, config_file_name)

    # Write the temp config
    with open(new_config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(test_config, file, sort_keys=False, allow_unicode=True)

    # Run evaluation
    memory_path = ""
    try:
        generation = Generation(new_config_path)
        (
            responses,
            inputs,
            queries,
            context,
            memory,
            personal_data,
            rephrased_message,
            user_ids,
            conversation_ids,
            message_ids,    
        ) = generation.inference()

        # Parse model outputs into the three channels
        resp_memory = []
        resp_personal = []
        resp_rephrased = []
        for r in responses:
            m, p, rr = parse_response(r)
            resp_memory.append(m)
            resp_personal.append(p)
            resp_rephrased.append(rr)

            print(f"Model response: {r}")
            print(f"Parsed into:")
            print(f"  -> memory: {m}")
            print(f"  -> personal data: {p}")
            print(f"  -> rephrased message: {rr}")
            print("-----")

        # Coerce GT lists to strings (empty when missing)
        gt_memory = [m or "" for m in memory]
        gt_personal = [p or "" for p in personal_data]
        gt_rephrased = [r or "" for r in rephrased_message]

        # Score memory
        per_ex_mem, sum_mem = generation.get_generation_acc(
            responses=resp_memory,
            inputs=inputs,
            ground_truth=gt_memory,
            queries=queries,
            user_ids=user_ids,
            conversation_ids=conversation_ids,
            message_ids=message_ids,
            evaluation_type="memory",
        )
        memory_path = generation.save_outputs("memory", per_ex_mem, sum_mem)

        # Score personal data
        per_ex_pd, sum_pd = generation.get_generation_acc(
            responses=resp_personal,
            inputs=inputs,
            ground_truth=gt_personal,
            queries=queries,
            user_ids=user_ids,
            conversation_ids=conversation_ids,
            message_ids=message_ids,
            evaluation_type="personal_data",
        )
        personal_path = generation.save_outputs("personal_data", per_ex_pd, sum_pd)

        # Score rephrased message
        per_ex_rp, sum_rp = generation.get_generation_acc(
            responses=resp_rephrased,
            inputs=inputs,
            ground_truth=gt_rephrased,
            queries=queries,
            user_ids=user_ids,
            conversation_ids=conversation_ids,
            message_ids=message_ids,
            evaluation_type="rephrased_message",
        )
        rephrased_path = generation.save_outputs("rephrased_message", per_ex_rp, sum_rp)

        per_example_all, summary_all = generation.get_generation_acc(
        evaluation_type="all",
        responses=responses,
        inputs=inputs,
        ground_truth=[f"Key Information: {m}\nPersonal Data: {p}\nRephrased User Query: {r}" for m, p, r in zip(gt_memory, gt_personal, gt_rephrased)],
        queries=queries,
        user_ids=user_ids,
        conversation_ids=conversation_ids,
        message_ids=message_ids,
    )
        all_path = generation.save_outputs("all", per_example_all, summary_all)

        print(f"Saved memory results to: {memory_path}")
        print(f"Saved personal data results to: {personal_path}")
        print(f"Saved rephrased message results to: {rephrased_path}")
        print(f"Saved all results to: {all_path}")

    finally:
        # Clean up vLLM distributed state and GPU memory
        try:
            destroy_model_parallel()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    return memory_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the model with the new data.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the base YAML config.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or HF path.")
    parser.add_argument("--test_dataset_name", type=str, required=True, help="Name of the test dataset.")
    parser.add_argument("--chat_template", type=str, default=None, help="Chat template identifier to use.")
    parser.add_argument("--max_model_len", type=int, default=12000, help="Max model length (tokens).")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens for generation.")
    parser.add_argument("--test_type", type=str, default="close", help="Test type flag (e.g., 'close' or 'open').")
    parser.add_argument("--open_content", type=int, default=None, help="Optional open-content switch.")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    eval(
        config_path=args.config_path,
        model_name=args.model_name,
        test_dataset_name=args.test_dataset_name,
        chat_template=args.chat_template,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        test_type=args.test_type,
        open_content=args.open_content,
    )
