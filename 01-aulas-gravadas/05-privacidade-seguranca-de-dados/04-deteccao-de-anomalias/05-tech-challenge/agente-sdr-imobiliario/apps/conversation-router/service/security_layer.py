from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Lista controlada de nomes próprios comuns pt-BR: habilita mascaramento de nome único
# e leak-check de NOME sem gerar falsos positivos em palavras comuns.
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
        # Nome único (token capitalizado isolado) só é mascarado se constar na lista controlada.
        singles = [
            (m.start(), m.end(), m.group(0))
            for m in _CAPITAL_TOKEN_RE.finditer(masked)
            if m.group(0).lower() in COMMON_FIRST_NAMES
        ]
        if singles:
            extracted.setdefault("NOME", [])
            for start, end, token in sorted(singles, reverse=True):
                masked = masked[:start] + "[NOME]" + masked[end:]
                if token not in extracted["NOME"]:
                    extracted["NOME"].insert(0, token)
        if extracted and self._pii_store is not None and session_id:
            self._pii_store.save(session_id, extracted)
        return masked

    def check_output_leak(self, text: str) -> tuple[bool, str | None]:
        for label in ("EMAIL", "TELEFONE", "CNPJ"):
            if PII_PATTERNS[label].search(text):
                logger.error("PII leakage detected (%s) in LLM output; blocking", label)
                return True, label
        lowered = text.lower()
        for name in COMMON_FIRST_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                logger.error("PII leakage detected (NOME) in LLM output; blocking")
                return True, "NOME"
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


class KmsPiiRegistry:
    """Registro de PII com blob criptografado via KMS no DynamoDB.

    Tabela DynamoDB e chave KMS são provisionados pelo estágio de IaC (deviação registrada no code-summary).
    """

    def __init__(self, dynamodb_client: Any, table_name: str, kms_client: Any, key_id: str | None = None) -> None:
        self._db = dynamodb_client
        self._table = table_name
        self._kms = kms_client
        self._key_id = key_id

    def save(self, session_id: str, extracted: dict[str, list[str]]) -> None:
        merged: dict[str, list[str]] = dict(self.load(session_id))
        for label, values in extracted.items():
            known = merged.setdefault(label, [])
            merged[label] = known + [v for v in values if v not in known]
        blob = self._kms.encrypt(
            KeyId=self._key_id or "alias/sdr-pii",
            Plaintext=json.dumps(merged).encode(),
        )
        self._db.put_item(
            TableName=self._table,
            Item={
                "PK": {"S": f"PII#{session_id}"},
                "SK": {"S": "PII"},
                "ciphertext": {"S": base64.b64encode(blob["CiphertextBlob"]).decode()},
            },
        )

    def load(self, session_id: str) -> dict[str, list[str]]:
        item = self._db.get_item(
            TableName=self._table,
            Key={"PK": {"S": f"PII#{session_id}"}, "SK": {"S": "PII"}},
        ).get("Item")
        if not item:
            return {}
        blob = base64.b64decode(item["ciphertext"]["S"])
        return json.loads(self._kms.decrypt(CiphertextBlob=blob)["Plaintext"].decode())


CONSENT_MESSAGE = (
    "Olá! Sou o assistente SDR da W Levitt, especializado em espaços corporativos. "
    "Para continuar, preciso do seu consentimento LGPD: coletarei nome, e-mail, telefone e CNPJ "
    "para atendimento SDR, com retenção de 90 dias e compartilhamento apenas com corretor/CRM. "
    "Você pode recusar ou revogar respondendo 'não'. Podemos continuar?"
)

REFUSAL_MESSAGE = "Entendido. Se mudar de ideia, envie /start novamente"

FALLBACK_MESSAGE = "Não posso ajudar com isso"
