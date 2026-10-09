from __future__ import annotations

import base64
import json
import logging
import re
import unicodedata
from typing import Any, Callable, Iterable

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
    "CNPJ": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
    "TELEFONE": re.compile(
        # (a) formatos comuns: 11 97991-8262, (11) 97991 8262, 1133334444
        r"(?<!\d)(?:\+?55[\s.-]*)?(?:\(?\d{2}\)?[\s.-]*)(?:9\d{4}|[2-8]\d{3})[\s.-]?\d{4}(?!\d)"
        # (b) celular com o 9 separado por qualquer coisa — como a pessoa digita/dita:
        #     (11) 9-7991-8262, +55 11 9 7991-8262, 11 9 7991 8262
        r"|(?<!\d)(?:\+?55[\s.-]*)?\(?0?\d{2}\)?[\s.-]*9[\s.-]+\d{4}[\s.-]*\d{4}(?!\d)"
        r"|(?<!\d)(?:\+?55[\s.-]*)?\(?0?\d{2}\)?[\s.-]*9[\s.-]*\d{4}[\s.-]+\d{4}(?!\d)"
    ),
}
_PII_PLACEHOLDER_RE = re.compile(r"\[(?:NOME|EMAIL|TELEFONE|CNPJ)(?:_\d+)?\]", re.IGNORECASE)


# --- Telefone FALADO (transcrição de áudio): "onze nove sete nove nove um oito dois seis dois" ----------
# Camada de PII determinística (a LLM nunca vê o número): palavras-número viram dígitos e, se a
# sequência tem formato de telefone brasileiro, é mascarada e guardada como dígitos normalizados.
_UNITS = {"zero": "0", "um": "1", "uma": "1", "dois": "2", "duas": "2", "tres": "3", "quatro": "4",
          "cinco": "5", "seis": "6", "meia": "6", "sete": "7", "oito": "8", "nove": "9"}
_TEENS = {"onze": "11", "doze": "12", "treze": "13", "quatorze": "14", "catorze": "14", "quinze": "15",
          "dezesseis": "16", "dezessete": "17", "dezoito": "18", "dezenove": "19"}
_TENS = {"vinte": "2", "trinta": "3", "quarenta": "4", "cinquenta": "5", "sessenta": "6",
         "setenta": "7", "oitenta": "8", "noventa": "9"}
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ]+|\d+")
_GAP_OK = re.compile(r"[\s,.;\-]*")
_MIN_WORDS = 3  # só dígitos soltos já são cobertos pelas regex; aqui exige palavras de número


def _spoken_fold(word: str) -> str:
    return unicodedata.normalize("NFKD", word.lower()).encode("ascii", "ignore").decode()


def _spoken_is_phone(digits: str) -> bool:
    if len(digits) >= 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) in (12, 13) and digits.startswith("55"):
        digits = digits[2:]
    if len(digits) == 11:
        return re.fullmatch(r"\d{2}9\d{8}", digits) is not None
    if len(digits) == 10:
        return re.fullmatch(r"\d{2}[2-8]\d{7}", digits) is not None
    return False


def find_spoken_phones(text: str) -> list[tuple[int, int, str]]:
    """(início, fim, dígitos) de cada telefone ditado por extenso (ou misturado com dígitos).

    A partir de cada palavra-número lê a sequência seguinte e aceita o MAIOR prefixo com
    formato de telefone, então "...dois seis dois, dois quartos" ainda mascara o número."""
    tokens = [(m.start(), m.end(), _spoken_fold(m.group())) for m in _TOKEN_RE.finditer(text)]
    found: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        digits, words, j, prev_end = "", 0, i, None
        best: tuple[int, int, str, int] | None = None  # (start, end, digitos, índice do próximo token)
        while j < len(tokens):
            s, e, t = tokens[j]
            if prev_end is not None and not _GAP_OK.fullmatch(text[prev_end:s]):
                break
            used, chunk, is_word = 1, None, True
            if t.isdigit() and len(t) <= 4:
                chunk, is_word = t, False
            elif t in _UNITS:
                chunk = _UNITS[t]
            elif t in _TEENS:
                chunk = _TEENS[t]
            elif t in _TENS:
                if (j + 2 < len(tokens) and tokens[j + 1][2] == "e" and tokens[j + 2][2] in _UNITS
                        and _GAP_OK.fullmatch(text[e:tokens[j + 1][0]])
                        and _GAP_OK.fullmatch(text[tokens[j + 1][1]:tokens[j + 2][0]])):
                    chunk, used, e = _TENS[t] + _UNITS[tokens[j + 2][2]], 3, tokens[j + 2][1]
                else:
                    chunk = _TENS[t] + "0"
            if chunk is None:
                break
            digits += chunk
            words += is_word
            prev_end, j = e, j + used
            if words >= _MIN_WORDS and _spoken_is_phone(digits):
                best = (tokens[i][0], e, digits, j)
        if best:
            found.append(best[:3])
            i = best[3]
        else:
            i += 1
    return found


def _mask_spoken_phones(text: str) -> tuple[str, list[str]]:
    out: list[str] = []
    numbers: list[str] = []
    last = 0
    for start, end, digits in find_spoken_phones(text):
        out += [text[last:start], "[TELEFONE]"]
        numbers.append(digits)
        last = end
    out.append(text[last:])
    return "".join(out), numbers


class SecurityLayer:
    def __init__(self, pii_store: Any | None = None) -> None:
        self._pii_store = pii_store

    def mask(self, text: str, session_id: str | None = None) -> str:
        # Telefone falado primeiro: sem isso o padrão de NOME (palavras capitalizadas em
        # sequência) engole "Onze Nove Sete..." e o número vira "nome" em vez de contato.
        masked, spoken_phones = _mask_spoken_phones(text)
        extracted: dict[str, list[str]] = {}
        # Máscara SÓ do que identifica um canal de contato: e-mail, telefone/WhatsApp e CNPJ.
        # Nome NÃO é adivinhado pelo jeito de escrever (palavras capitalizadas): tratava lugar
        # ("Santo André"), empreendimento e o texto colado do bot como pessoa e travava a
        # conversa. O nome do lead vem do perfil do Telegram (handler) e fica no pii_store.
        for label, pattern in PII_PATTERNS.items():
            if label == "NOME":
                continue
            matches = pattern.findall(masked)
            if not matches:
                continue
            extracted[label] = matches
            masked = pattern.sub(f"[{label}]", masked)
        if spoken_phones:
            extracted["TELEFONE"] = spoken_phones + extracted.get("TELEFONE", [])
        if extracted and self._pii_store is not None and session_id:
            self._pii_store.save(session_id, extracted)
        return masked

    def check_output_leak(
        self, text: str, session_pii: dict[str, list[str]] | None = None
    ) -> tuple[bool, str | None]:
        if _PII_PLACEHOLDER_RE.search(text):
            logger.error("Unresolved PII placeholder detected in agent output")
            return True, "PLACEHOLDER"
        for label in ("EMAIL", "TELEFONE", "CNPJ"):
            if PII_PATTERNS[label].search(text):
                logger.error("PII leakage detected (%s) in LLM output; blocking", label)
                return True, label
        if find_spoken_phones(text):
            logger.error("PII leakage detected (TELEFONE por extenso) in LLM output; blocking")
            return True, "TELEFONE"
        # NOME: só bloqueia se for um nome REAL do lead nesta sessão — evita falso
        # positivo em nome de bairro/empreendimento ou uso natural da palavra
        # (checar contra a lista estática COMMON_FIRST_NAMES gerava esse falso positivo).
        lowered = text.lower()
        for name in (session_pii or {}).get("NOME", []):
            if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
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
        # "concorrente" e "insulto" foram removidos: bloqueavam objeções comerciais
        # legítimas ("o de vocês é melhor que o concorrente X?").
        denied_topics = ("política", "politica", "eleição", "eleicao")
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

CONSENT_REASK_MESSAGE = (
    "Antes de seguir, preciso do seu consentimento LGPD para coletar e usar seus dados no "
    "atendimento. Pode confirmar respondendo 'sim'? Se preferir não, responda 'não'."
)

FALLBACK_MESSAGE = "Não posso ajudar com isso"
