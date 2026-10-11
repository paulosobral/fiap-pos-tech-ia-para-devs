"""X-Ray no conversation-router (ECS Fargate).

Diferente das Lambdas, o ECS não abre o segmento sozinho: o `server.py` cria um por requisição
(`segment`) e as chamadas à LLM viram subsegmentos (`subsegment`), o que mostra onde o tempo de cada
turno é gasto. As chamadas AWS (botocore) viram subsegmentos; HTTP de saída NÃO é instrumentado, porque a URL do
Telegram carrega o token do bot. O SDK envia os dados ao daemon (sidecar da task, UDP 2000 em 127.0.0.1).

Sem o `aws-xray-sdk` (testes locais) tudo vira no-op; observabilidade nunca derruba o atendimento.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, ContextManager

logger = logging.getLogger(__name__)
_recorder: Any = None
_loaded = False


def _get_recorder() -> Any:
    """Devolve o `xray_recorder` do SDK, ou None quando o SDK não está instalado."""
    global _recorder, _loaded
    if not _loaded:
        _loaded = True
        os.environ.setdefault("AWS_XRAY_CONTEXT_MISSING", "LOG_ERROR")  # fora de segmento só registra o erro
        try:
            from aws_xray_sdk.core import patch, xray_recorder

            # Só botocore. NUNCA patch_all(): ele grava a URL das chamadas de saída no trace e a do
            # Telegram carrega o token do bot (https://api.telegram.org/bot<TOKEN>/...).
            patch(("botocore",))
            _recorder = xray_recorder
        except ImportError:
            _recorder = None
        except Exception:  # pragma: no cover - defensivo
            logger.warning("X-Ray: falha ao iniciar; seguindo sem tracing", exc_info=True)
            _recorder = None
    return _recorder


def enable() -> bool:
    return _get_recorder() is not None


def segment(name: str) -> ContextManager[Any]:
    """Segmento raiz de uma requisição (no-op sem SDK)."""
    recorder = _get_recorder()
    return recorder.in_segment(name) if recorder is not None else contextlib.nullcontext()


def subsegment(name: str, **annotations: Any) -> ContextManager[Any]:
    """Subsegmento dentro do segmento corrente; erros levantados dentro dele ficam marcados no trace."""
    recorder = _get_recorder()
    if recorder is None:
        return contextlib.nullcontext()
    return _annotated_subsegment(recorder, name, annotations)


@contextlib.contextmanager
def _annotated_subsegment(recorder: Any, name: str, annotations: dict[str, Any]):
    with recorder.in_subsegment(name) as sub:
        for key, value in annotations.items():
            _put(sub, key, value)
        yield sub


def annotate(key: str, value: Any) -> None:
    """Anotação pesquisável no segmento/subsegmento corrente (no-op sem SDK ou sem segmento)."""
    recorder = _get_recorder()
    if recorder is None:
        return
    try:
        entity = recorder.current_subsegment() or recorder.current_segment()
    except Exception:
        return
    _put(entity, key, value)


def _put(entity: Any, key: str, value: Any) -> None:
    if entity is None or value is None:
        return
    try:
        entity.put_annotation(key, value if isinstance(value, (str, int, float, bool)) else str(value))
    except Exception:  # pragma: no cover - anotação nunca pode quebrar o fluxo
        pass
