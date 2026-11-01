"""
Evaluation and inference script for Structured Entity and Intent Extraction.
Tests model output against Pydantic validation and schema compliance.
Supports testing base model, adapter-loaded model, and fused model.
"""

import argparse
import json
import time
from typing import Dict, Any, List
from pydantic import ValidationError
from schema import StructuredOutput
import mlx.core as mx
from mlx_lm import load, generate

def run_benchmark(
    model_path: str,
    adapter_path: str = None,
    test_file: str = "data/test.jsonl",
    max_samples: int = 50
):
    print(f"\n==========================================")
    print(f"Benchmarking Model: {model_path}")
    if adapter_path:
        print(f"With Adapter: {adapter_path}")
    print(f"Device: {mx.default_device()}")
    print(f"==========================================\n")

    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    # Read test samples
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break

    valid_json_count = 0
    pydantic_valid_count = 0
    exact_action_count = 0
    exact_human_approval_count = 0
    total_tokens = 0
    start_time = time.time()

    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        target_json = json.loads(messages[2]["content"])

        # Format input using tokenizer chat template
        prompt_chat = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        prompt = tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=True
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            verbose=False
        )

        clean_resp = response.strip()
        # Clean markdown wrappers if any were emitted
        if clean_resp.startswith("```json"):
            clean_resp = clean_resp[7:]
        if clean_resp.startswith("```"):
            clean_resp = clean_resp[3:]
        if clean_resp.endswith("```"):
            clean_resp = clean_resp[:-3]
        clean_resp = clean_resp.strip()

        is_valid_json = False
        is_pydantic_valid = False
        parsed_obj = None

        try:
            parsed = json.loads(clean_resp)
            is_valid_json = True
            valid_json_count += 1
            
            validated = StructuredOutput.model_validate(parsed)
            is_pydantic_valid = True
            pydantic_valid_count += 1
            parsed_obj = validated
        except (json.JSONDecodeError, ValidationError) as e:
            pass

        if is_pydantic_valid and parsed_obj:
            if parsed_obj.action == target_json.get("action"):
                exact_action_count += 1
            if parsed_obj.requires_human_approval == target_json.get("requires_human_approval"):
                exact_human_approval_count += 1

        if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
            print(f"Processed {idx + 1}/{len(samples)} samples...")

    elapsed = time.time() - start_time
    total = len(samples)

    print("\n---------------- Benchmark Results ----------------")
    print(f"Evaluated Samples: {total}")
    print(f"Total Inference Time: {elapsed:.2f}s ({elapsed/total:.3f}s per sample)")
    print(f"Valid JSON Rate: {valid_json_count}/{total} ({valid_json_count/total*100:.1f}%)")
    print(f"Strict Pydantic Schema Accuracy: {pydantic_valid_count}/{total} ({pydantic_valid_count/total*100:.1f}%)")
    print(f"Action Classification Accuracy: {exact_action_count}/{total} ({exact_action_count/total*100:.1f}%)")
    print(f"Human Approval Flag Accuracy: {exact_human_approval_count}/{total} ({exact_human_approval_count/total*100:.1f}%)")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--test-file", type=str, default="data/test.jsonl")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        max_samples=args.samples
    )
