"""X-Ray nas chamadas AWS (boto3/botocore) das Lambdas.

O Lambda já abre o segmento (`tracing_mode = "Active"` no Terraform); aqui só instrumentamos o
botocore para que DynamoDB, SQS, KMS e Secrets Manager apareçam como subsegmentos. O identificador
do trace segue nas mensagens do SQS, então a Lambda que consome continua o mesmo trace.

NÃO usamos `patch_all()`: ele também instrumenta `requests`/`urllib` e grava a URL das chamadas de
saída no trace, o que vazaria o token do bot (`https://api.telegram.org/bot<TOKEN>/...`).
O botocore só registra operação, tabela e fila, nunca o conteúdo das mensagens.

Sem o `aws-xray-sdk` instalado (testes locais) não faz nada; e observabilidade nunca derruba o handler.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
_enabled = False


def enable() -> bool:
    """Instrumenta o botocore uma única vez. Devolve True se o X-Ray ficou ativo."""
    global _enabled
    if _enabled:
        return True
    # Chamada instrumentada fora de um segmento só registra o erro (o padrão do SDK é levantar exceção).
    os.environ.setdefault("AWS_XRAY_CONTEXT_MISSING", "LOG_ERROR")
    try:
        from aws_xray_sdk.core import patch
    except ImportError:
        return False
    try:
        patch(("botocore",))
    except Exception:  # pragma: no cover - defensivo
        logger.warning("X-Ray: falha ao instrumentar o botocore; seguindo sem tracing", exc_info=True)
        return False
    _enabled = True
    return True
