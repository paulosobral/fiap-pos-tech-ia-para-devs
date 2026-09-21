from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{2}\)|\d{2})[\s.-]?\d{4,5}[\s.-]?\d{4}")
_LABEL_NAME_RE = re.compile(r"^\s*(?:nome|name|lead|cliente)\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LABEL_EMAIL_RE = re.compile(r"^\s*e-?mail\s*[:\-]\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE | re.MULTILINE)
_LABEL_PHONE_RE = re.compile(r"^\s*(?:tel(?:efone)?|fone|phone|cel(?:ular)?)\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SUBJECT_NAME_RE = re.compile(r"(?:novo lead|contato|lead|interesse)\s*[:\-]\s*([^\-–—,|]+)", re.IGNORECASE)


@dataclass
class ParsedContact:
    name: str | None
    email: str | None
    phone: str | None
    message_text: str
    subject: str | None


class EmailParser(Protocol):
    def parse(self, mail: dict[str, Any]) -> ParsedContact | None: ...


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _digits(candidate: str) -> int:
    return len(re.sub(r"\D", "", candidate))


class HeuristicEmailParser:
    """Parser heurístico de e-mails de contato de portais imobiliários (FR12.2).

    Tolerante a formatos: campos rotulados no corpo (`Nome:`, `E-mail:`,
    `Telefone:`), padrão de assunto de portal (`Novo lead: Fulano`) e
    cabeçalho From. Corpo MIME cru (`ses.mail.content`, base64) é decodificado
    quando presente; sem e-mail extraível o contato é considerado não
    parseável (retorno `None` → drop explícito no orquestrador).
    """

    def parse(self, mail: dict[str, Any]) -> ParsedContact | None:
        common = mail.get("commonHeaders") or {}
        subject = _clean(common.get("subject"))
        body = self._decode_body(mail)
        message_text = body or subject or ""
        email = self._extract_email(mail, common, body)
        if not email:
            return None
        name = self._extract_name(mail, common, subject, body, email)
        phone = self._extract_phone(body)
        return ParsedContact(
            name=name,
            email=email,
            phone=phone,
            message_text=message_text[:MAX_MESSAGE_CHARS],
            subject=subject,
        )

    def _decode_body(self, mail: dict[str, Any]) -> str:
        raw = mail.get("content")
        if not raw:
            return ""
        try:
            message = BytesParser(policy=policy.default).parsebytes(base64.b64decode(raw))
            part = message.get_body(preferencelist=("plain",))
            return str(part.get_content()) if part is not None else ""
        except Exception as exc:
            logger.error("unreadable email content: %s", type(exc).__name__)
            return ""

    def _extract_email(self, mail: dict[str, Any], common: dict[str, Any], body: str) -> str | None:
        labeled = _LABEL_EMAIL_RE.search(body)
        if labeled:
            return labeled.group(1).strip()
        sender = self._from_value(mail, common)
        if sender:
            match = _EMAIL_RE.search(sender)
            if match:
                return match.group(0).strip()
        match = _EMAIL_RE.search(body)
        return match.group(0).strip() if match else None

    def _extract_name(
        self, mail: dict[str, Any], common: dict[str, Any], subject: str | None, body: str, email: str
    ) -> str | None:
        labeled = _LABEL_NAME_RE.search(body)
        if labeled:
            candidate = _clean(labeled.group(1))
            if candidate and "@" not in candidate:
                return candidate
        if subject:
            matched = _SUBJECT_NAME_RE.search(subject)
            if matched:
                candidate = _clean(matched.group(1))
                if candidate:
                    return candidate
        sender = self._from_value(mail, common)
        if sender and "<" in sender:
            display = _clean(sender.split("<", 1)[0].strip(' "\''))
            if display and "@" not in display:
                return display
        return None

    def _extract_phone(self, body: str) -> str | None:
        labeled = _LABEL_PHONE_RE.search(body)
        if labeled:
            candidate = _clean(labeled.group(1))
            if candidate and 10 <= _digits(candidate) <= 13:
                return candidate
            return None
        match = _PHONE_RE.search(body)
        if match and 10 <= _digits(match.group(0)) <= 13:
            return _clean(match.group(0))
        return None

    @staticmethod
    def _from_value(mail: dict[str, Any], common: dict[str, Any]) -> str | None:
        sender = common.get("from")
        if isinstance(sender, list):
            sender = sender[0] if sender else None
        return _clean(sender) or _clean(mail.get("source"))
