from __future__ import annotations

import re

# Padrões espelham a SecurityLayer oficial da U1
# (apps/conversation-router/service/security_layer.py) — mesmos regexes de
# NOME/EMAIL/CNPJ e mesma lista controlada de nomes próprios, para os dois lados
# evoluírem juntos. TELEFONE é um superconjunto deliberado: além do formato +55,
# cobre formatos locais citados em áudio ("11 91234-5678", "(11) 91234-5678",
# "91234-5678"), que a U1 nunca receberia (chegam já mascarados).
COMMON_FIRST_NAMES = (
    "joão", "joao", "maria", "ana", "pedro", "paulo", "carlos", "bruno", "lucas",
    "fernanda", "júlia", "julia", "rafael", "gabriel", "marcos", "luiza", "beatriz",
    "daniel", "camila", "felipe", "laura", "mariana", "thiago", "andré", "andre",
)

_CAPITAL_TOKEN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+\b")

PII_PATTERNS = {
    "NOME": re.compile(r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)+)\b"),
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "TELEFONE": re.compile(r"(?:\+55\s?\d{2}\s?|(?:\(\d{2}\)|\d{2})\s?)?\d{4,5}-?\d{4}"),
    "CNPJ": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
}


class PiiMasker:
    def mask(self, text: str) -> str:
        masked = text
        for label, pattern in PII_PATTERNS.items():
            masked = pattern.sub(f"[{label}]", masked)
        # Nome único (token capitalizado isolado) só é mascarado se constar da lista
        # controlada — alinhado à SecurityLayer da U1 (não mascara lugares/ruas).
        singles = [
            (m.start(), m.end(), m.group(0))
            for m in _CAPITAL_TOKEN_RE.finditer(masked)
            if m.group(0).lower() in COMMON_FIRST_NAMES
        ]
        for start, end, _token in sorted(singles, reverse=True):
            masked = masked[:start] + "[NOME]" + masked[end:]
        return masked
