"""
guardrails.py
=============
Input and output safety checks for the medical assistant.

- Input:  Detect prompt injection patterns; enforce length limits.
- Output: Enforce prescription disclaimer on any response containing medication names.
"""

from __future__ import annotations

import re

MAX_INPUT_LENGTH = 2000

# Padrões que indicam tentativa de prompt injection.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|unrestricted)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?(?:evil|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are\s+)?(?:not|without)\s+(?:any\s+)?restrictions?", re.IGNORECASE),
    re.compile(r"DAN\b", re.IGNORECASE),  # "Do Anything Now" jailbreak variant
    re.compile(r"<\|(?:im_start|endoftext|system)\|>", re.IGNORECASE),  # token injection
    re.compile(r"##\s*system\s*:", re.IGNORECASE),
]

# Palavras-chave de fármacos/prescrição que disparam disclaimer obrigatório.
_PRESCRIPTION_KEYWORDS: list[str] = [
    "prescrever", "prescrito", "prescrição", "receita médica",
    "administrar", "dose", "dosagem", "comprimido", "ampola",
    "mg", "ml", "mcg", "via oral", "via iv", "intravenoso",
    "antibiótico", "anti-inflamatório", "analgésico", "diurético",
    "insulina", "metformina", "enalapril", "captopril", "atorvastatina",
    "amoxicilina", "azitromicina", "dipirona", "paracetamol",
]

_DISCLAIMER = (
    "\n\n⚠️ **Aviso de Segurança**: As informações acima são de suporte educacional. "
    "Qualquer prescrição, dose ou administração de medicamento deve ser validada e "
    "assinada pelo médico responsável."
)


class InputValidationError(ValueError):
    pass


def validate_input(text: str) -> str:
    """
    Validates user input. Returns sanitised text or raises InputValidationError.
    """
    if not text or not text.strip():
        raise InputValidationError("A pergunta não pode estar vazia.")

    if len(text) > MAX_INPUT_LENGTH:
        raise InputValidationError(
            f"Pergunta muito longa ({len(text)} chars). Limite: {MAX_INPUT_LENGTH}."
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise InputValidationError(
                "Entrada bloqueada: padrão potencialmente malicioso detectado. "
                "Por favor, reformule sua pergunta clínica."
            )

    return text.strip()


def enforce_prescription_disclaimer(response: str) -> str:
    """
    Appends a safety disclaimer if the response contains medication-related content.
    Does NOT modify the response otherwise.
    """
    response_lower = response.lower()
    if any(kw in response_lower for kw in _PRESCRIPTION_KEYWORDS):
        if _DISCLAIMER.strip() not in response:
            return response + _DISCLAIMER
    return response


def sanitise_for_display(text: str) -> str:
    """Strip control characters that could break the Streamlit UI."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
