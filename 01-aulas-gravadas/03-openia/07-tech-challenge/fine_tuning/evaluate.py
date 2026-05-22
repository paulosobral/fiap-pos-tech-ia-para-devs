"""
evaluate.py
===========
Fase 3 — Model evaluation:
  - Load fine-tuned LoRA adapter
  - Run inference on PubMedQA test split
  - Compute ROUGE-L and BLEU scores
  - Print qualitative examples

Run:
    python fine_tuning/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nltk
import torch
from datasets import Dataset
from evaluate import load as load_metric
from unsloth import FastLanguageModel

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA = BASE_DIR / "data" / "processed" / "medical_test.jsonl"
ADAPTER_DIR = Path(__file__).resolve().parent / "output" / "lora_model"

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 256
NUM_QUALITATIVE_EXAMPLES = 3

ALPACA_TEMPLATE = (
    "Below is a clinical question. Write an evidence-based medical response.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def load_test_data() -> list[dict]:
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA}\n"
            "Run `python fine_tuning/prepare_dataset.py` first."
        )
    records: list[dict] = []
    with TEST_DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_response(model, tokenizer, record: dict, device: str) -> str:
    prompt = ALPACA_TEMPLATE.format(
        instruction=record["instruction"],
        input=record.get("input", ""),
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIR}.\n"
            "Run `python fine_tuning/train.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading fine-tuned model from {ADAPTER_DIR} on {device} …")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    records = load_test_data()
    print(f"Evaluating on {len(records)} test examples …")

    rouge = load_metric("rouge")
    bleu = load_metric("bleu")

    predictions: list[str] = []
    references: list[str] = []

    for i, rec in enumerate(records):
        pred = generate_response(model, tokenizer, rec, device)
        ref = rec["output"]
        predictions.append(pred)
        references.append(ref)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(records)} done …")

    # ROUGE
    rouge_result = rouge.compute(predictions=predictions, references=references)
    # BLEU
    tokenised_preds = [p.split() for p in predictions]
    tokenised_refs = [[r.split()] for r in references]
    bleu_result = bleu.compute(predictions=tokenised_preds, references=tokenised_refs)

    print("\n=== Evaluation Results ===")
    print(f"ROUGE-1:  {rouge_result['rouge1']:.4f}")
    print(f"ROUGE-2:  {rouge_result['rouge2']:.4f}")
    print(f"ROUGE-L:  {rouge_result['rougeL']:.4f}")
    print(f"BLEU:     {bleu_result['bleu']:.4f}")

    print(f"\n=== Qualitative Examples (first {NUM_QUALITATIVE_EXAMPLES}) ===")
    for i in range(min(NUM_QUALITATIVE_EXAMPLES, len(records))):
        rec = records[i]
        print(f"\n[Example {i + 1}]")
        print(f"  Instruction: {rec['instruction'][:120]} …")
        print(f"  Expected:    {references[i][:200]} …")
        print(f"  Generated:   {predictions[i][:200]} …")

    # Save results to JSON
    results_path = ADAPTER_DIR.parent / "eval_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rouge1": rouge_result["rouge1"],
                "rouge2": rouge_result["rouge2"],
                "rougeL": rouge_result["rougeL"],
                "bleu": bleu_result["bleu"],
                "num_examples": len(records),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
