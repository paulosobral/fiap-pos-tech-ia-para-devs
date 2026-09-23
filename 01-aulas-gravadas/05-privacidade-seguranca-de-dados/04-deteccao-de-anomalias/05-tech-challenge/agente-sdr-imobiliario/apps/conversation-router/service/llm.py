"""Cliente LLM multi-tier via LiteLLM (OpenRouter) — FR-02/ADR-010.

Camadas:
  Tier 1 (Primary):   deepseek/deepseek-chat — barato, 90% do tráfego
  Tier 2 (Fallback):  anthropic/claude-3-haiku — contingência em 429/timeout
  Tier 3 (Complex):   anthropic/claude-3.5-sonnet — negociação avançada/handoff

Config (env, injetadas no deploy):
  LLM_MODEL_PRIMARY         nome do modelo Tier 1 (default deepseek/deepseek-chat)
  LLM_MODEL_FALLBACK        nome do modelo Tier 2 (default anthropic/claude-3-haiku)
  LLM_MODEL_COMPLEX         nome do modelo Tier 3 (default anthropic/claude-3.5-sonnet)
  LLM_MODEL_PRIMARY_SSM     path SSM Parameter Store p/ Tier 1
  LLM_MODEL_FALLBACK_SSM    path SSM Parameter Store p/ Tier 2
  LLM_MODEL_COMPLEX_SSM     path SSM Parameter Store p/ Tier 3
  LLM_API_KEY               chave do OpenRouter
  LLM_TIMEOUT               timeout da chamada em segundos (default 8)
"""
from __future__ import annotations

import json
import logging
import os
import time as _time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PRIMARY = "deepseek/deepseek-chat"
DEFAULT_MODEL_FALLBACK = "anthropic/claude-3-haiku"
DEFAULT_MODEL_COMPLEX = "anthropic/claude-3.5-sonnet"

TIER_PRIMARY = "primary"
TIER_FALLBACK = "fallback"
TIER_COMPLEX = "complex"

_SSM_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 60.0

_SSM_PARAM_BY_TIER = {
    TIER_PRIMARY: "LLM_MODEL_PRIMARY_SSM",
    TIER_FALLBACK: "LLM_MODEL_FALLBACK_SSM",
    TIER_COMPLEX: "LLM_MODEL_COMPLEX_SSM",
}

_ENV_BY_TIER = {
    TIER_PRIMARY: "LLM_MODEL_PRIMARY",
    TIER_FALLBACK: "LLM_MODEL_FALLBACK",
    TIER_COMPLEX: "LLM_MODEL_COMPLEX",
}

_DEFAULT_BY_TIER = {
    TIER_PRIMARY: DEFAULT_MODEL_PRIMARY,
    TIER_FALLBACK: DEFAULT_MODEL_FALLBACK,
    TIER_COMPLEX: DEFAULT_MODEL_COMPLEX,
}


def resolve_model(
    tier: str = TIER_PRIMARY,
    explicit_model: str | None = None,
) -> str:
    if explicit_model:
        return explicit_model

    now = _time.time()
    if tier in _SSM_CACHE:
        cached_val, cached_ts = _SSM_CACHE[tier]
        if (now - cached_ts) < _CACHE_TTL_SECONDS:
            return cached_val

    ssm_env = _SSM_PARAM_BY_TIER.get(tier, "")
    ssm_param = os.environ.get(ssm_env)
    if ssm_param:
        try:
            import boto3
            ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))
            res = ssm.get_parameter(Name=ssm_param)
            val = res.get("Parameter", {}).get("Value")
            if val and val.strip():
                _SSM_CACHE[tier] = (val.strip(), now)
                return val.strip()
        except Exception:
            pass

    env_name = _ENV_BY_TIER.get(tier, "")
    env_model = os.environ.get(env_name)
    if env_model and env_model.strip():
        _SSM_CACHE[tier] = (env_model.strip(), now)
        return env_model.strip()

    return _DEFAULT_BY_TIER.get(tier, DEFAULT_MODEL_PRIMARY)


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
    "SDR imobiliário B2B da W Levitt. Atenda leads do Telegram em português, "
    "tom consultivo, direto. Regras rígidas:\n\n"
    "1. A RESPOSTA OFICIAL contém o que DEVE ser dito. Você APENAS melhora o tom, "
    "NUNCA altera o significado nem adiciona informações novas.\n"
    "2. IMÓVEIS RECOMENDADOS: SÓ cite imóveis que apareçam nesta lista. "
    "Se a lista for '(nenhum)' ou vazia, NUNCA mencione imóvel, preço, metragem, "
    "bairro ou valor — apenas reescreva a resposta oficial.\n"
    "3. NUNCA invente: preço, metragem, bairro, nome de empreendimento, "
    "disponibilidade, ou prazo.\n"
    "4. Máximo 3 frases. A pergunta final DEVE SER COerente com a ação do lead:\n"
    "   - se mostrou opções → pergunte se quer agendar ou ver mais\n"
    "   - se é qualificação → peça o dado faltante\n"
    "   - se é handoff → confirme o encaminhamento\n"
    "   - NUNCA force agendamento quando o lead só quer ver propriedades."
)

# --- LiteLLM (cliente abstraído conforme PRD §8.1) ---------------------------

try:
    import litellm

    litellm.telemetry = False
    _HAS_LITELLM = True
except ImportError:  # pragma: no cover
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
    return _urllib_completion(messages, api_key, model, max_tokens, temperature, response_format)


def _completion_with_fallback(
    messages: list[dict[str, str]],
    api_key: str,
    primary_model: str,
    fallback_model: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, str] | None = None,
) -> str:
    try:
        return _completion(messages, api_key, primary_model, max_tokens, temperature, response_format)
    except Exception:
        logger.warning(
            "Modelo primário %s falhou; tentando fallback %s",
            primary_model, fallback_model, exc_info=True,
        )
        return _completion(messages, api_key, fallback_model, max_tokens, temperature, response_format)


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
    url: str = "",
    timeout: float | None = None,
) -> tuple[str, float]:
    primary = resolve_model(TIER_PRIMARY, explicit_model=model)
    fallback = resolve_model(TIER_FALLBACK)
    if timeout:
        os.environ["LLM_TIMEOUT"] = str(timeout)
    raw = _completion_with_fallback(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        api_key=api_key,
        primary_model=primary,
        fallback_model=fallback,
        max_tokens=40,
        temperature=0,
        response_format={"type": "json_object"},
    )
    intent, confidence = _parse(raw)
    if intent is None or confidence is None:
        raise ValueError("Resposta LLM sem intenção válida")
    logger.info("LLM classify: intent=%s confidence=%.2f model=%s", intent, confidence, primary)
    return intent, confidence


def generate_reply(
    message: str,
    canned_response: str,
    lead_info: dict[str, Any],
    properties: list[dict[str, Any]],
    api_key: str,
    model: str | None = None,
    url: str = "",
    timeout: float | None = None,
    force_complex: bool = False,
) -> str:
    primary = resolve_model(TIER_PRIMARY, explicit_model=model) if not force_complex else resolve_model(TIER_PRIMARY)
    fallback = resolve_model(TIER_FALLBACK)
    complex_model = resolve_model(TIER_COMPLEX)

    if force_complex:
        chosen_model = complex_model
        use_fallback = False
    else:
        chosen_model = primary
        use_fallback = True

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
    if use_fallback:
        raw = _completion_with_fallback(
            messages=[
                {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            api_key=api_key,
            primary_model=chosen_model,
            fallback_model=fallback,
            max_tokens=280,
            temperature=0.6,
        )
    else:
        raw = _completion(
            messages=[
                {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            api_key=api_key,
            model=chosen_model,
            max_tokens=280,
            temperature=0.6,
        )
    if not raw or not raw.strip():
        raise ValueError("Resposta LLM vazia")
    logger.info("LLM reply gerado (%d chars) model=%s", len(raw), chosen_model)
    return raw.strip()


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


# --- Roteamento agentic (ADR-011) --------------------------------------------
# O LLM nunca escolhe o próximo nó do grafo diretamente — apenas classifica a
# ação pretendida do lead num enum fixo. O código (sales_flow.py) mapeia
# (estado_atual, action, gates de negócio) -> próximo nó, sempre determinístico.

VALID_ACTIONS = (
    "provide_info",
    "request_options",
    "refine_search",
    "compare_properties",
    "visit_interest",
    "request_schedule",
    "request_human",
    "decline",
    "unclear",
)

_KNOWN_LEAD_FIELDS = ("area", "region", "budget", "deadline", "people_count", "decision_maker")

_ROUTER_SYSTEM_PROMPT = (
    "Você é o roteador de conversa de um SDR imobiliário B2B. A cada mensagem do lead, "
    "faça duas coisas:\n"
    "1. EXTRAIA os dados de negócio citados na mensagem (só os que aparecerem): "
    "area, region, budget, deadline, people_count (inteiro), decision_maker ('yes'/'no').\n"
    "2. CLASSIFIQUE a ação pretendida do lead, escolhendo UMA destas: "
    f"{', '.join(VALID_ACTIONS)}.\n"
    "   - provide_info: está respondendo com dados (padrão sem pedido claro)\n"
    "   - request_options: quer VER/RECEBER/VER mais opções de imóveis ou detalhes "
    "('cadê as opções', 'me envie mais detalhes', 'tem mais?', 'quero ver', 'manda as opções', 'envie detalhes')\n"
    "   - refine_search: quer AJUSTAR critérios da busca ('mais barato', 'menor', 'outra região', "
    "'sem estacionamento?', filtros novos)\n"
    "   - compare_properties: quer COMPARAR opções já mostradas ('diferença entre 1 e 2', "
    "'compara as duas', 'qual é melhor')\n"
    "   - visit_interest: demonstra INTERESSE em visitar ou em um imóvel específico "
    "('quero visitar', 'gostei da 2', 'essa me interessa') SEM verbo de agendamento explícito\n"
    "   - request_schedule: quer agendar visita APENAS com verbo explícito de agendamento "
    "('agendar', 'marcar visita', 'reserve', 'agende', 'quero marcar')\n"
    "   - request_human: QUER FALAR COM UM CORRETOR/HUMANO AGORA (ex: 'quero um corretor', "
    "'fala com alguém', 'preciso de um humano', 'atendente')\n"
    "   - decline: quer parar/recusar/desistir\n"
    "   - unclear: ambígua, sem ação clara\n"
    "REGRA DE OURO: pedir mais detalhes = request_options; ajustar filtros = refine_search; "
    "visitar/gostei = visit_interest; agendar = verbo explícito (request_schedule). "
    "Pedir um corretor = request_human.\n"
    'Responda APENAS com JSON: {"lead_info": {...}, "action": "<uma das opções>"}.'
)


def extract_and_route(
    message: str,
    lead_info: dict[str, Any],
    current_state: str,
    api_key: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Uma chamada LLM (Tier 1): extrai deltas de lead_info + classifica a ação
    do lead num enum fixo (VALID_ACTIONS). O próximo nó do grafo é decidido em
    código (sales_flow.py), nunca pelo LLM — ver ADR-011.

    Levanta ValueError se a resposta não for JSON válido ou a ação estiver fora
    do enum; o caller deve tratar como falha e cair no fallback determinístico.
    """
    primary = resolve_model(TIER_PRIMARY, explicit_model=model)
    fallback = resolve_model(TIER_FALLBACK)
    context = json.dumps(
        {"lead_info_atual": lead_info, "estado_atual": current_state}, ensure_ascii=False, default=str
    )[:600]
    raw = _completion_with_fallback(
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXTO: {context}\n\nMENSAGEM DO LEAD: {message}"},
        ],
        api_key=api_key,
        primary_model=primary,
        fallback_model=fallback,
        max_tokens=200,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return _parse_extract_and_route(raw)


def _parse_extract_and_route(raw: str) -> dict[str, Any]:
    if not raw:
        raise ValueError("Resposta LLM vazia")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Resposta LLM não é JSON válido") from exc
    action = data.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Ação fora do enum válido: {action!r}")
    extracted = data.get("lead_info")
    if not isinstance(extracted, dict):
        extracted = {}
    clean = {k: v for k, v in extracted.items() if k in _KNOWN_LEAD_FIELDS and v not in (None, "")}
    return {"lead_info": clean, "action": action}
