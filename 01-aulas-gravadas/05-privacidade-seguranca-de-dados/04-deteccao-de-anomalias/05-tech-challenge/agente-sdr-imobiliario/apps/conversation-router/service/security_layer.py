from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "NOME": re.compile(r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)+)\b"),
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "TELEFONE": re.compile(r"\+55\s?\d{2}\s?\d{4,5}-?\d{4}"),
    "CNPJ": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
}


class SecurityLayer:
    def __init__(self, pii_store: Any | None = None) -> None:
        self._pii_store = pii_store

    def mask(self, text: str, session_id: str | None = None) -> str:
        masked = text
        extracted: dict[str, list[str]] = {}
        for label, pattern in PII_PATTERNS.items():
            matches = pattern.findall(masked)
            if not matches:
                continue
            extracted[label] = matches if label != "NOME" else [m[0] if isinstance(m, tuple) else m for m in matches]
            masked = pattern.sub(f"[{label}]", masked)
        if extracted and self._pii_store is not None and session_id:
            self._pii_store.save(session_id, extracted)
        return masked

    def check_output_leak(self, text: str) -> tuple[bool, str | None]:
        for label in ("EMAIL", "TELEFONE", "CNPJ"):
            if PII_PATTERNS[label].search(text):
                logger.error("PII leakage detected (%s) in LLM output; blocking", label)
                return True, label
        return False, None

    def guard(self, text: str) -> tuple[bool, str | None]:
        injection_patterns = ("ignore previous instructions", "ignore as instruções anteriores", "system prompt", "revele seu prompt")
        lowered = text.lower()
        for pattern in injection_patterns:
            if pattern in lowered:
                logger.warning("Prompt injection attempt blocked")
                return True, "prompt_injection"
        denied_topics = ("política", "politica", "eleição", "eleicao", "concorrente", "insulto")
        for topic in denied_topics:
            if topic in lowered:
                logger.warning("Denied topic '%s' blocked", topic)
                return True, "denied_topic"
        return False, None

    def unmask(self, placeholders_text: str, pii_data: dict[str, list[str]]) -> str:
        text = placeholders_text
        for label, values in pii_data.items():
            placeholder = f"[{label}]"
            for value in values:
                text = text.replace(placeholder, value, 1)
        return text


CONSENT_MESSAGE = (
    "Olá! Sou o assistente SDR da W Levitt, especializado em espaços corporativos. "
    "Para continuar, preciso do seu consentimento LGPD: coletarei nome, e-mail, telefone e CNPJ "
    "para atendimento SDR, com retenção de 90 dias e compartilhamento apenas com corretor/CRM. "
    "Você pode recusar ou revogar respondendo 'não'. Podemos continuar?"
)

REFUSAL_MESSAGE = "Entendido. Se mudar de ideia, envie /start novamente"

FALLBACK_MESSAGE = "Não posso ajudar com isso"
