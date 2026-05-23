"""
train.py
========
Fase 2 — Fine-tuning pipeline:
  - Model:   unsloth/llama-3-8b-bnb-4bit  (4-bit quantised LLaMA-3-8B)
  - Method:  LoRA via Unsloth + SFTTrainer (TRL)
  - Data:    data/processed/medical_train.jsonl  (Alpaca format)
  - Output:  fine_tuning/output/lora_model/

Run:
    python fine_tuning/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import unsloth  # must be first — patches torch/transformers before other imports
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DATA = BASE_DIR / "data" / "processed" / "medical_train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "lora_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
# llama-3.2-3b fits in ~6 GB VRAM at 4-bit; use llama-3-8b-bnb-4bit if you have ≥12 GB
MODEL_NAME = "unsloth/llama-3.2-3b-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_ARGS = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",
)

ALPACA_TEMPLATE = (
    "Below is a clinical question. Write an evidence-based medical response.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_train_dataset(tokenizer) -> Dataset:
    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_DATA}\n"
            "Run `python fine_tuning/prepare_dataset.py` first."
        )

    records: list[dict] = []
    with TRAIN_DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    eos = tokenizer.eos_token

    def format_row(row: dict) -> dict:
        text = (
            ALPACA_TEMPLATE.format(
                instruction=row["instruction"],
                input=row.get("input", ""),
                output=row["output"],
            )
            + eos
        )
        return {"text": text}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_row)
    print(f"Loaded {len(dataset)} training examples.")
    return dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto-detect
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_train_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TRAIN_ARGS,
    )

    print("Starting training …")
    trainer_stats = trainer.train()
    print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

    print(f"Saving LoRA adapter to {OUTPUT_DIR} …")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Adapter saved.")

    # Quick sanity-check inference
    print("\nSanity check inference:")
    FastLanguageModel.for_inference(model)
    sample_prompt = (
        ALPACA_TEMPLATE.format(
            instruction="Quais os critérios diagnósticos para sepse?",
            input="",
            output="",
        )
    )
    inputs = tokenizer(sample_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
