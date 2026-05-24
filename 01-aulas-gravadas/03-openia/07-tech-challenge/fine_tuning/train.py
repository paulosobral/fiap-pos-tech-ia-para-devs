"""
train.py
========
Pipeline de fine-tuning da Fase 2:
    - Modelo:  unsloth/Qwen2.5-3B-Instruct-bnb-4bit  (Qwen2.5-3B quantizado em 4 bits)
    - Método:  LoRA via Unsloth + SFTTrainer (TRL)
    - Dados:   data/processed/medical_train.jsonl  (formato Alpaca)
    - Saída:   fine_tuning/output/lora_model/

Execução:
        python fine_tuning/train.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Deve ser definido antes de importar torch / unsloth para o alocador CUDA usar.
# expandable_segments: reduz fragmentação e melhora reuso de blocos de VRAM.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import unsloth  # deve ser o primeiro — aplica patches em torch/transformers antes dos outros imports
from unsloth import FastLanguageModel, is_bfloat16_supported

import psutil
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DATA = BASE_DIR / "data" / "processed" / "medical_train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "lora_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
# Qwen2.5-3B-Instruct: melhor multilíngue (PT-BR) e instruction-following que LLaMA-3.2-3B
# no mesmo tamanho. Pesos 4-bit ~1.9 GB — cabe folgado em 6 GB VRAM.
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
# MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # se tiver ≥ 12 GB VRAM
MAX_SEQ_LENGTH = 1024   # 2048 → 1024: reduz pela metade a memória por passo; perguntas e respostas médicas raramente ultrapassam isso
LOAD_IN_4BIT = True
LORA_R = 8              # 16 → 8: menor custo computacional por camada do adapter; suficiente para adaptação de domínio
LORA_ALPHA = 16         # manter alpha > r para escala efetiva (alpha/r = 2)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # apenas atenção: convergência mais rápida
    "gate_proj", "up_proj", "down_proj",       # MLP — melhora retenção de fatos clínicos
]

TRAIN_ARGS = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=4,   # 2 → 4: VRAM liberada por sequências mais curtas
    gradient_accumulation_steps=2,  # 4 → 2: batch efetivo mantido em ~8, menos pontos de sincronização
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
    # Desativa pinned memory para o SO poder usar RAM nas camadas offloaded para CPU.
    dataloader_pin_memory=False,
)

# ── Helpers de memória ────────────────────────────────────────────────────────

def _build_max_memory() -> dict[int | str, str]:
    """Calcula o orçamento de memória: 90 % da VRAM dedicada livre + 50 % da RAM do sistema livre.

    Ao passar este dicionário para FastLanguageModel.from_pretrained (via device_map="auto"),
    o HuggingFace/Accelerate consegue descarregar camadas do modelo que não cabem na VRAM
    dedicada para a RAM do sistema (memória compartilhada/unificada), ampliando a memória
    efetivamente disponível para o modelo.
    """
    budget: dict[int | str, str] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            gpu_mib = int(free * 0.90 / 1024**2)
            budget[i] = f"{gpu_mib}MiB"
    ram_free = psutil.virtual_memory().available
    cpu_mib = int(ram_free * 0.50 / 1024**2)
    budget["cpu"] = f"{cpu_mib}MiB"
    print(f"  Memory budget: {budget}")
    return budget


ALPACA_TEMPLATE = (
    "Below is a clinical question. Write an evidence-based medical response.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


# ── Carregador de dados ───────────────────────────────────────────────────────

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


# ── Principal ─────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # detecção automática
        load_in_4bit=LOAD_IN_4BIT,
        # device_map="auto" + max_memory permite ao Accelerate/HuggingFace
        # descarregar camadas que excedem a VRAM dedicada para a RAM do sistema.
        device_map="auto",
        max_memory=_build_max_memory(),
    )
    torch.cuda.empty_cache()

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
        dataset_num_proc=4,   # mais workers em paralelo para tokenização
        packing=True,         # empacota vários exemplos curtos por janela de contexto → ~2x throughput
        args=TRAIN_ARGS,
    )

    print("Starting training …")
    trainer_stats = trainer.train()
    print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

    print(f"Saving LoRA adapter to {OUTPUT_DIR} …")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Adapter saved.")

    # Inferência rápida para verificação de sanidade.
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
