"""Cliente LLM (OpenRouter) — classificação de intenção do lead.

Sem dependência externa (stdlib urllib) de propósito: o POC empacota as Lambdas
com zip direto (limite 50MB) e o runtime python3.11 não traz `requests`. A troca
de provedor (OpenRouter -> Bedrock) segue o contrato LiteLLM do PRD: este módulo
é o único ponto que fala com o provider.

Config (env, injetadas no deploy):
  LLM_API_KEY   chave do OpenRouter (Secrets Manager). Vazia/ausente = módulo
                indisponível: o fluxo cai no classificador por regex.
  LLM_MODEL     modelo (default 'anthropic/claude-3.5-haiku').
  LLM_TIMEOUT   timeout da chamada em segundos (default 8).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-3.5-haiku"
VALID_INTENTS = ("purchase", "rent", "investment", "unknown")

_SYSTEM_PROMPT = (
    "Você é o classificador de intenção de um SDR imobiliário corporativo B2B "
    "(imóveis comerciais: escritórios, lajes corporativas, conjuntos comerciais). "
    "Classifique a mensagem do lead quanto à intenção de COMPRA, LOCAÇÃO ou "
    "INVESTIMENTO. Responda APENAS com JSON no formato "
    '{"intent": "purchase"|"rent"|"investment"|"unknown", "confidence": 0.0..1.0}. '
    'Use "unknown" quando a mensagem não indicar nenhuma das três; confiança '
    "baixa (0.1-0.4) quando ambígua."
)

_REPLY_SYSTEM_PROMPT = (
    "Você é o SDR de IA de uma consultoria imobiliária corporativa B2B "
    "(W Levitt), atendendo leads do Telegram em português do Brasil. "
    "Atributos: tom consultivo e humanizado, educação e objetividade; "
    "nunca invente preço, metragem, nome de empreendimento ou prazo que não "
    "esteja no catálogo fornecido. Você recebe (a) a resposta 'oficial' do "
    "sistema (o conteúdo que DEVE ser transmitido), (b) os imóveis "
    "recomendados (só estes podem ser citados) e (c) os dados do lead. "
    "Reescreva a resposta oficial em texto natural de chat, mantendo o "
    "significado, sem bullet listas longas, com até 3 opções e uma pergunta "
    "de avanço no final. Responda SEMPRE de acordo com o contexto citado; "
    "se nenhum imóvel do catálogo atender ao pedido do lead, diga "
    "sinceramente que vai verificar com a equipe e sugerir novas opções em "
    "seguida (não invente). Se a resposta oficial é de recusa/consentimento "
    "LGPD ou encaminhamento ao corretor, mantenha o conteúdo que não deve "
    "ser alterado e apenas polia. NUNCA peça dados sensíveis nem prometa "
    "disponibilidade não citada."
)


def classify_intent(
    message: str,
    api_key: str,
    model: str | None = None,
    url: str = OPENROUTER_URL,
    timeout: float | None = None,
) -> tuple[str, float]:
    """Retorna (intent, confidence) via OpenRouter. Levanta exceção em falha —
    o chamador decide o fallback (regex)."""
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    timeout = timeout or float(os.environ.get("LLM_TIMEOUT", "8"))
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "max_tokens": 40,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "agente-sdr-imobiliario-poc",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    raw = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    intent, confidence = _parse(raw)
    if intent is None or confidence is None:
        raise ValueError("Resposta LLM sem intenção válida")
    logger.info("LLM classify: intent=%s confidence=%.2f", intent, confidence)
    return intent, confidence


def _parse(raw: str) -> tuple[str | None, float | None]:
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    intent = data.get("intent", "unknown")
    if intent not in VALID_INTENTS:
        return None, None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None, None
    return intent, max(0.0, min(1.0, confidence))


def generate_reply(
    message: str,
    canned_response: str,
    lead_info: dict[str, Any],
    properties: list[dict[str, Any]],
    api_key: str,
    model: str | None = None,
    url: str = OPENROUTER_URL,
    timeout: float | None = None,
) -> str:
    """Reescreve a resposta oficial do fluxo em texto natural humanizado (FR-02).

    NUNCA recebe PII real: `message` já vem mascarada da security-layer (PRD 7.3).
    `properties` são os top-k do RAG — só esses podem ser citados (constraint via
    RAG, PRD 8.1/8.2). Levanta exceção em qualquer falha: o chamador mantém a
    resposta oficial (fallback determinístico).
    """
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    timeout = timeout or float(os.environ.get("LLM_TIMEOUT", "8"))
    lead = json.dumps(lead_info, ensure_ascii=False, default=str)[:800]
    props = json.dumps(
        [
            {k: p.get(k) for k in ("title", "type", "class", "region", "area_util",
                                   "price", "price_text", "vagas", "disponibilidade")}
            for p in properties
        ],
        ensure_ascii=False,
        default=str,
    )[:1500]
    user_block = (
        "RESPOSTA OFICIAL DO SISTEMA (transmita o conteúdo, pode melhorar o tom):\n"
        f"{canned_response}\n\n"
        f"DADOS DO LEAD (estruturados, já mascarados):\n{lead or '(vazio)'}\n\n"
        f"IMÓVEIS RECOMENDADOS (SÓ estes podem ser citados):\n{props or '(nenhum)'}\n\n"
        f"ÚLTIMA MENSAGEM DO LEAD:\n{message[:500]}"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
        "max_tokens": 280,
        "temperature": 0.6,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "agente-sdr-imobiliario-poc",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    raw = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if not raw or not raw.strip():
        raise ValueError("Resposta LLM vazia")
    logger.info("LLM reply gerado (%d chars)", len(raw))
    return raw.strip()