"""Cliente LLM via LiteLLM (OpenRouter → Claude 3.5 Haiku) — FR-02/§8.1 do PRD.

Usa LiteLLM como cliente abstraído (LLM_PROVIDER=openrouter|bedrock) conforme PRD §8.1.
Fallback: se litellm não estiver instalado (ex.: testes locais sem deps), usa urllib
puro como transporte — mantém o mesmo contrato e os meslos prompts.

Config (env, injetadas no deploy):
  LLM_API_KEY       chave do OpenRouter (override de dev/testes). Produção: o handler
                    resolve via Secrets Manager (sdr/llm-api-key, env LLM_API_SECRET_ID).
                    Nenhuma das duas = módulo indisponível: o fluxo cai no classificador
                    por regex.
  LLM_MODEL     modelo (default 'anthropic/claude-3.5-haiku').
  LLM_TIMEOUT   timeout da chamada em segundos (default 8).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic/claude-3-haiku"

_CACHED_MODEL: str | None = None
_CACHED_MODEL_TS: float = 0.0
_CACHE_TTL_SECONDS = 60.0


def resolve_model(explicit_model: str | None = None) -> str:
    """Resolve o modelo OpenRouter a ser usado.
    Ordem de precedência:
      1. explicit_model passado diretamente
      2. AWS SSM Parameter Store (/sdr/llm-model ou env LLM_MODEL_SSM_PARAM)
      3. Variável de ambiente LLM_MODEL
      4. DEFAULT_MODEL
    """
    if explicit_model:
        return explicit_model

    global _CACHED_MODEL, _CACHED_MODEL_TS
    import time
    now = time.time()
    if _CACHED_MODEL and (now - _CACHED_MODEL_TS) < _CACHE_TTL_SECONDS:
        return _CACHED_MODEL

    ssm_param_name = os.environ.get("LLM_MODEL_SSM_PARAM", "/sdr/llm-model")
    if ssm_param_name:
        try:
            import boto3
            ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))
            res = ssm.get_parameter(Name=ssm_param_name)
            val = res.get("Parameter", {}).get("Value")
            if val and val.strip():
                _CACHED_MODEL = val.strip()
                _CACHED_MODEL_TS = now
                logger.info("Modelo LLM resolvido via SSM Parameter Store (%s): %s", ssm_param_name, _CACHED_MODEL)
                return _CACHED_MODEL
        except Exception:
            # Fallback silencioso para env ou default se SSM falhar/não existir
            pass

    env_model = os.environ.get("LLM_MODEL")
    if env_model and env_model.strip():
        _CACHED_MODEL = env_model.strip()
        _CACHED_MODEL_TS = now
        return _CACHED_MODEL

    return DEFAULT_MODEL
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

# --- LiteLLM (cliente abstraído conforme PRD §8.1) ---------------------------

try:
    import litellm

    litellm.telemetry = False  # NF-02/§8.8: telemetria desligada
    _HAS_LITELLM = True
except ImportError:  # pragma: no cover — fallback para testes sem deps
    litellm = None  # type: ignore[assignment]
    _HAS_LITELLM = False


def _completion(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, str] | None = None,
) -> str:
    """Chama LiteLLM (openrouter) ou fallback urllib se litellm ausente."""
    if _HAS_LITELLM:
        kwargs: dict[str, Any] = {
            "model": f"openrouter/{model}",
            "messages": messages,
            "api_key": api_key,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": float(os.environ.get("LLM_TIMEOUT", "8")),
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = litellm.completion(**kwargs)
        return resp["choices"][0]["message"]["content"]
    # Fallback urllib (testes locais sem litellm instalado)
    return _urllib_completion(messages, api_key, model, max_tokens, temperature, response_format)


def _urllib_completion(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, str] | None,
) -> str:
    import urllib.request

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "agente-sdr-imobiliario-poc",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(os.environ.get("LLM_TIMEOUT", "8"))) as resp:
        body = json.loads(resp.read().decode())
    return (body.get("choices") or [{}])[0].get("message", {}).get("content", "")


# --- API pública -------------------------------------------------------------

def classify_intent(
    message: str,
    api_key: str,
    model: str | None = None,
    url: str = "",  # mantido por compat; litellm ignora
    timeout: float | None = None,
) -> tuple[str, float]:
    """Retorna (intent, confidence) via LiteLLM/OpenRouter. Levanta exceção em falha —
    o chamador decide o fallback (regex)."""
    model = resolve_model(model)
    if timeout:
        os.environ["LLM_TIMEOUT"] = str(timeout)
    raw = _completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        api_key=api_key,
        model=model,
        max_tokens=40,
        temperature=0,
        response_format={"type": "json_object"},
    )
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
    url: str = "",
    timeout: float | None = None,
) -> str:
    """Reescreve a resposta oficial do fluxo em texto natural humanizado (FR-02).

    NUNCA recebe PII real: `message` já vem mascarada da security-layer (PRD 7.3).
    `properties` são os top-k do RAG — só esses podem ser citados (constraint via
    RAG, PRD 8.1/8.2). Levanta exceção em qualquer falha: o chamador mantém a
    resposta oficial (fallback determinístico).
    """
    model = resolve_model(model)
    if timeout:
        os.environ["LLM_TIMEOUT"] = str(timeout)
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
    raw = _completion(
        messages=[
            {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
        api_key=api_key,
        model=model,
        max_tokens=280,
        temperature=0.6,
    )
    if not raw or not raw.strip():
        raise ValueError("Resposta LLM vazia")
    logger.info("LLM reply gerado (%d chars)", len(raw))
    return raw.strip()
