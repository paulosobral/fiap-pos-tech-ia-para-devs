"""
guardrails.py
=============
Verificações de segurança de entrada e saída para o assistente médico.

- Entrada: Detectar padrões de prompt injection; impor limites de tamanho.
- Saída:   Impor aviso de prescrição em qualquer resposta que contenha nomes de medicamentos.
"""

from __future__ import annotations

import re

MAX_INPUT_LENGTH = 2000

# Padrões que indicam tentativa de prompt injection.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Inglês
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|unrestricted)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?(?:evil|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are\s+)?(?:not|without)\s+(?:any\s+)?restrictions?", re.IGNORECASE),
    # Português
    re.compile(r"ignore\s+(todas\s+as\s+)?instru[cç][oõ]es\s+anteriores?", re.IGNORECASE),
    re.compile(r"desconsidere?\s+(o\s+)?prompt\s+do\s+sistema", re.IGNORECASE),
    re.compile(r"voc[eê]\s+agora\s+[eé]\s+(?:um?\s+)?(?:diferente|novo|irrestrito|sem\s+restri[cç][oõ]es)", re.IGNORECASE),
    re.compile(r"aja\s+como\s+(?:se\s+voc[eê]\s+(?:fosse|seja)\s+)?(?:um?\s+)?(?:malicioso|irrestrito|sem\s+restri[cç][oõ]es)", re.IGNORECASE),
    re.compile(r"finja\s+(?:que\s+voc[eê]\s+)?(?:n[aã]o\s+tem?|est[aá]\s+sem?)\s+restri[cç][oõ]es?", re.IGNORECASE),
    # Agnóstico de idioma
    re.compile(r"DAN\b", re.IGNORECASE),  # variante de jailbreak "Do Anything Now"
    re.compile(r"<\|(?:im_start|endoftext|system)\|>", re.IGNORECASE),  # injeção de tokens especiais
    re.compile(r"##\s*s(?:ystem|istema)\s*:", re.IGNORECASE),
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
    Valida a entrada do usuário. Retorna o texto higienizado ou lança InputValidationError.
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
    Adiciona um aviso de segurança se a resposta contiver conteúdo relacionado a medicamentos.
    Não modifica a resposta caso contrário.
    """
    response_lower = response.lower()
    if any(kw in response_lower for kw in _PRESCRIPTION_KEYWORDS):
        if _DISCLAIMER.strip() not in response:
            return response + _DISCLAIMER
    return response


def sanitise_for_display(text: str) -> str:
    """Remove caracteres de controle que podem quebrar a interface Streamlit."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
