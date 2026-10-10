"""Nome da assistente: vive no SSM Parameter Store (`/sdr/bot-name`) e é lido com cache.

A ordem é a mesma dos modelos de LLM (ver `llm.resolve_model`): SSM → variável `BOT_NAME` → padrão.
Com TTL curto, trocar o nome no SSM vale em poucos minutos, sem novo deploy.
"""
from __future__ import annotations

import logging
import os
import time as _time

logger = logging.getLogger(__name__)

DEFAULT_BOT_NAME = "Cecília"
SSM_PARAM_ENV = "BOT_NAME_SSM"  # nome do parâmetro SSM (definido no Terraform)
ENV_NAME = "BOT_NAME"
CACHE_TTL_SECONDS = 300

_cache: tuple[str, float] | None = None


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())[:40]


def get_bot_name() -> str:
    global _cache
    now = _time.time()
    if _cache and now - _cache[1] < CACHE_TTL_SECONDS:
        return _cache[0]
    name = ""
    param = os.environ.get(SSM_PARAM_ENV)
    if param:
        try:
            import boto3

            ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))
            name = _clean(ssm.get_parameter(Name=param).get("Parameter", {}).get("Value"))
        except Exception:
            logger.warning("Nome da bot: falha ao ler o SSM %s; usando fallback", param, exc_info=True)
    name = name or _clean(os.environ.get(ENV_NAME)) or DEFAULT_BOT_NAME
    _cache = (name, now)
    return name


def reset_cache() -> None:
    global _cache
    _cache = None
