"""SQS Worker para ECS Fargate — consome mensagens de voz da fila SQS.

Este runner substitui o event source mapping SQS->Lambda para o voice-adapter.
Em vez de uma Lambda acionada por SQS, um container ECS Fargate faz long-polling
da fila `sdr-voice-queue` e processa mensagens com faster-whisper real.

O loop:
1. sqs.receive_message (max_messages, wait_time_seconds=20)
2. Para cada mensagem: VoiceAdapter.process_message(body)
3. Sucesso -> sqs.delete_message
4. Retry (TranscriptionError/RouterError) -> não deleta (SQS re-entrega)
5. Drop (AudioConversionError/invalido) -> sqs.delete_message (descarta)
6. Sleep poll_interval segundos, repetir

Variáveis de ambiente:
- VOICE_QUEUE_URL: URL da fila SQS (obrigatória)
- TELEGRAM_BOT_TOKEN: token do bot (obrigatória)
- INTERNAL_SECRET_TOKEN: secret para POST /internal/inbound-text (obrigatória)
- ROUTER_BASE_URL: URL do API Gateway que hospeda o conversation-router (obrigatória)
- SESSIONS_TABLE: tabela DynamoDB de sessões (default: sdr-sessions)
- AWS_REGION: região AWS (default: us-east-1)
- WORKER_POLL_INTERVAL: segundos entre polls (default: 5)
- WORKER_MAX_MESSAGES: mensagens por batch (default: 5)
- WHISPER_MODEL_SIZE: tamanho do modelo (default: small)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import boto3

from service.pii import PiiMasker
from service.router_gateway import HttpRouterGateway
from service.telegram_gateway import TelegramGateway
from service.transcriber import WhisperTranscriber
from service.voice_adapter import VoiceAdapter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"missing required env var {name}")
    return value


def _build_adapter() -> VoiceAdapter:
    import requests

    http = requests.Session()
    telegram = TelegramGateway(http, bot_token=_env("TELEGRAM_BOT_TOKEN"))
    transcriber = WhisperTranscriber(model_size=os.environ.get("WHISPER_MODEL_SIZE", "small"))
    router = HttpRouterGateway(
        http,
        base_url=_env("ROUTER_BASE_URL"),
        secret_token=_env("INTERNAL_SECRET_TOKEN"),
    )
    sessions = SessionLookup(
        boto3.client("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1")),
        os.environ.get("SESSIONS_TABLE", "sdr-sessions"),
    )
    return VoiceAdapter(
        telegram=telegram,
        transcriber=transcriber,
        router=router,
        sessions=sessions,
        masker=PiiMasker(),
    )


class SessionLookup:
    """Adapter DynamoDB para validar sessões (mesma lógica do handler Lambda)."""

    def __init__(self, client: Any, table_name: str) -> None:
        self._client = client
        self._table = table_name

    def get_session(self, session_id: str, telegram_user_id: int) -> dict[str, Any] | None:
        try:
            resp = self._client.get_item(
                TableName=self._table,
                Key={"session_id": {"S": session_id}, "telegram_user_id": {"N": str(telegram_user_id)}},
            )
            item = resp.get("Item")
            if not item:
                return None
            return {"session_id": session_id, "telegram_user_id": telegram_user_id}
        except Exception:
            logger.warning("session lookup failed", exc_info=True)
            return None


def main() -> None:
    queue_url = _env("VOICE_QUEUE_URL")
    region = os.environ.get("AWS_REGION", "us-east-1")
    poll_interval = int(os.environ.get("WORKER_POLL_INTERVAL", "5"))
    max_messages = int(os.environ.get("WORKER_MAX_MESSAGES", "5"))

    sqs = boto3.client("sqs", region_name=region)
    adapter = _build_adapter()

    logger.info("voice-worker started: queue=%s, max_messages=%d, interval=%ds", queue_url, max_messages, poll_interval)

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=20,
            )
            messages = response.get("Messages", [])
            if not messages:
                continue

            for msg in messages:
                receipt_handle = msg["ReceiptHandle"]
                record = {"body": msg["Body"], "messageId": msg["MessageId"]}
                outcome = adapter._process_record(record)

                if outcome == "ok" or outcome == "drop":
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                    logger.info("message %s: %s (deleted)", msg["MessageId"], outcome)
                else:
                    logger.warning("message %s: %s (kept for retry)", msg["MessageId"], outcome)

        except Exception:
            logger.error("poll cycle failed", exc_info=True)
            time.sleep(poll_interval)


if __name__ == "__main__":
    main()
