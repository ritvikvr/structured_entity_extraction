"""
Production inference pipeline with local fallback, Metal acceleration,
and robust Pydantic structured output validation.
"""

import json
from typing import Optional, Union, Dict, Any
from pydantic import ValidationError
from schema import StructuredOutput
import mlx.core as mx
from mlx_lm import load, generate

SYSTEM_PROMPT = (
    "You are an AI operational dispatch assistant. Your task is to extract structured operational "
    "actions and parameters from messy customer communications. Output strictly valid JSON conforming "
    "to the required schema without any conversational text or markdown code fences."
)

class StructuredEntityExtractor:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        print(f"Loading extractor on {mx.default_device()}...")
        if adapter_path:
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_path)
        print("Model loaded successfully.")

    def extract(self, user_message: str, max_tokens: int = 256) -> StructuredOutput:
        prompt_chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=True
        )

        response_str = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )

        clean_resp = response_str.strip()
        if clean_resp.startswith("```json"):
            clean_resp = clean_resp[7:]
        if clean_resp.startswith("```"):
            clean_resp = clean_resp[3:]
        if clean_resp.endswith("```"):
            clean_resp = clean_resp[:-3]
        clean_resp = clean_resp.strip()

        try:
            data = json.loads(clean_resp)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model failed to produce valid JSON: {clean_resp}") from exc

        return StructuredOutput.model_validate(data)

def demo():
    import sys
    model_to_use = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-3B-Instruct-4bit"
    adapter = sys.argv[2] if len(sys.argv) > 2 else "adapters"

    extractor = StructuredEntityExtractor(model_to_use, adapter_path=adapter)

    test_queries = [
        "Hey, can someone refund tx_998124? Charged $249.99 for annual plan by mistake. account usr_7712. App is not for us.",
        "URGENT: Escalate ticket immediately! usr_1190. Complete database downtime on production cluster. SLA violated!",
        "Hi, cancel my premium membership for user usr_4432. Switching to a local provider next month.",
        "Need to update VAT ID and business billing company name for usr_9821 please."
    ]

    print("\n--- Running Interactive Extraction Demo ---")
    for q in test_queries:
        print(f"\nUser Input: \"{q}\"")
        try:
            res = extractor.extract(q)
            print("Extracted Structured Output:")
            print(res.model_dump_json(indent=2))
        except Exception as err:
            print(f"Extraction Error: {err}")

if __name__ == "__main__":
    demo()
