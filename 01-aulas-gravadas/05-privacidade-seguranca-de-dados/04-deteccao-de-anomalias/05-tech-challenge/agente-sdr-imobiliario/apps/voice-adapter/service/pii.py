from __future__ import annotations

import re

PII_PATTERNS = {
    "NOME": re.compile(r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)+)\b"),
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "TELEFONE": re.compile(r"\+55\s?\d{2}\s?\d{4,5}-?\d{4}"),
    "CNPJ": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
}


class PiiMasker:
    def mask(self, text: str) -> str:
        masked = text
        for label, pattern in PII_PATTERNS.items():
            masked = pattern.sub(f"[{label}]", masked)
        return masked
