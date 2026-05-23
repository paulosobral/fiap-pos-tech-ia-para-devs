"""
llm_loader.py
=============
Loads the fine-tuned LoRA model (or falls back to base model) and exposes a
LangChain-compatible HuggingFacePipeline wrapper.

Usage:
    from assistant.llm_loader import build_llm
    llm = build_llm()
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from unsloth import FastLanguageModel

# ── Unsloth thread-safety patch ───────────────────────────────────────────────
# unsloth_zoo.rl_environments.time_limit uses signal.alarm() which only works
# on the main thread. Streamlit runs scripts in worker threads, so we make
# time_limit a no-op when called outside the main thread.
import unsloth_zoo.rl_environments as _rl_env  # noqa: E402

_orig_time_limit = _rl_env.time_limit


@contextmanager
def _safe_time_limit(seconds):
    if threading.current_thread() is threading.main_thread():
        with _orig_time_limit(seconds):
            yield
    else:
        yield


_rl_env.time_limit = _safe_time_limit
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "fine_tuning" / "output" / "lora_model"
FALLBACK_MODEL = "unsloth/llama-3.2-3b-bnb-4bit"

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512

_SYSTEM_PROMPT = (
    "Você é um assistente médico virtual do hospital. "
    "Sua função é apoiar médicos com informações clínicas baseadas em evidências e nos protocolos internos. "
    "NUNCA prescreva medicamentos diretamente sem validação de um médico. "
    "Sempre indique a fonte da informação quando disponível. "
    "Responda em português."
)

ALPACA_TEMPLATE = (
    "Below is a clinical question. Write an evidence-based medical response.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def build_llm(use_adapter: bool = True) -> HuggingFacePipeline:
    model_path = str(ADAPTER_DIR) if (use_adapter and ADAPTER_DIR.exists()) else FALLBACK_MODEL
    source = "fine-tuned adapter" if (use_adapter and ADAPTER_DIR.exists()) else "base model (fallback)"
    print(f"[llm_loader] Loading {source}: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # device must NOT be passed when the model was loaded via accelerate (4-bit / unsloth)
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("[llm_loader] LLM ready.")
    return llm


def format_alpaca_prompt(question: str, context: str = "") -> str:
    return ALPACA_TEMPLATE.format(instruction=question, input=context)
