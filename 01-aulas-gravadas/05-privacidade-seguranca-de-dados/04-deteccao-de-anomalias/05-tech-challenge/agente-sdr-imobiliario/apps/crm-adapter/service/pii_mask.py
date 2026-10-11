from __future__ import annotations

import re

# Padrões OFICIAIS de masking (espelho do SecurityLayer da u1-core-conversation,
# apps/conversation-router/service/security_layer.py). A cópia é obrigatória:
# a fronteira Lambda proíbe import cross-app (a u1 não expõe internals a consumidores).

# Lista controlada de nomes próprios comuns pt-BR: habilita mascaramento de nome único
# e evita falsos positivos em palavras comuns.
COMMON_FIRST_NAMES = (
    "joão", "joao", "maria", "ana", "pedro", "paulo", "carlos", "bruno", "lucas",
    "fernanda", "júlia", "julia", "rafael", "gabriel", "marcos", "luiza", "beatriz",
    "daniel", "camila", "felipe", "laura", "mariana", "thiago", "andré", "andre",
)

_CAPITAL_TOKEN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\b")

PII_PATTERNS = {
    "NOME": re.compile(r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)+)\b"),
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "TELEFONE": re.compile(r"\+55\s?\d{2}\s?\d{4,5}-?\d{4}"),
    "CNPJ": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
}


def mask_pii(text: str) -> str:
    """Mascara PII com o padrão oficial da u1 (NOME/EMAIL/TELEFONE/CNPJ → [LABEL]).

    Uso exclusivo para logs (ex.: preview de body SQS malformado) — não persiste
    PII e não substitui o fluxo de masking/unmask do produtor.
    """
    masked = text
    for label, pattern in PII_PATTERNS.items():
        masked = pattern.sub(f"[{label}]", masked)
    # Nome único (token capitalizado isolado) só é mascarado se constar da lista controlada.
    for token in set(_CAPITAL_TOKEN_RE.findall(masked)):
        if token.lower() in COMMON_FIRST_NAMES:
            masked = masked.replace(token, "[NOME]")
    return masked
