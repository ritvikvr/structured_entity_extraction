# Structured Entity & Intent Extraction Pipeline

An end-to-end QLoRA fine-tuning and local inference pipeline for `mlx-community/Qwen2.5-3B-Instruct-4bit` on Apple Silicon (M-series / Metal) using Apple MLX.

The system extracts structured JSON actions and parameters from messy, unstructured customer service queries with a **neuro-symbolic hybrid pattern** combining LLM semantic parsing and deterministic Pydantic v2 guardrails.

---

## Architecture Overview

```
Customer Message (Messy / Informal)
           │
           ▼
Apple MLX + Metal GPU (Qwen 2.5-3B-Instruct-4bit + LoRA Adapters)
           │
           ▼ (Draft JSON Output)
Pydantic v2 Schema & Deterministic Business Policy
           │
           ▼
Validated Operational Dispatch (requires_human_approval: true/false)
```

---

## Target JSON Schema

```json
{
  "action": "process_refund | cancel_subscription | escalate_ticket | update_profile",
  "parameters": {
    "user_id": "string",
    "transaction_id": "string | null",
    "amount": "number | null",
    "currency": "string | null",
    "reason": "string"
  },
  "requires_human_approval": "boolean"
}
```

### Deterministic Policy Rules
- `escalate_ticket` $\to$ `requires_human_approval: true`
- `process_refund` with `amount > 100` $\to$ `requires_human_approval: true`
- `process_refund` with `amount <= 100` $\to$ `requires_human_approval: false`
- `update_profile` with compliance keywords (KYC, Legal name, VAT) $\to$ `requires_human_approval: true`
- `cancel_subscription` $\to$ `requires_human_approval: false`

---

## Directory Structure

```
.
├── .gitignore
├── adapters/
│   ├── adapter_config.json      # LoRA training configuration
│   └── adapters.safetensors     # Trained LoRA adapter weights (51MB)
├── data/
│   ├── train.jsonl              # 500 instruction-tuning examples
│   ├── valid.jsonl              # 80 validation examples
│   └── test.jsonl               # 80 holdout test examples
├── schema.py                    # Pydantic schema contracts & policy engine
├── generate_dataset.py          # Synthetic dataset generation engine
├── lora_config.yaml             # MLX QLoRA training parameters
├── evaluate.py                  # Quantitative benchmarking harness
├── inference.py                 # Serving client with Metal acceleration
└── README.md
```

---

## Quickstart

### 1. Setup Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm pydantic huggingface_hub datasets
```

### 2. Run Inference Demo
```bash
python inference.py
```

### 3. Run Benchmark Evaluation
```bash
python evaluate.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter-path adapters --samples 15
```

---

## Evaluation Benchmark

| Metric | Zero-Shot Base Model | Fine-Tuned (Adapters + Policy) |
| :--- | :---: | :---: |
| **Valid JSON Rate** | 93.3% | **100.0%** |
| **Strict Pydantic Schema Accuracy** | 0.0% | **100.0%** |
| **Action Classification Accuracy** | 0.0% | **100.0%** |
| **Human Approval Flag Accuracy** | 0.0% | **100.0%** |
| **Average Latency / Sample** | 3.28s | **2.18s** |
