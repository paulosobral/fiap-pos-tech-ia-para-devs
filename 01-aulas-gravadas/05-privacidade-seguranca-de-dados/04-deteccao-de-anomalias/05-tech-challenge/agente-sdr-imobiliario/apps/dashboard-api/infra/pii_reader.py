import base64
import json
import logging
from typing import Any

from logs import log_event


class PiiReader:
    """Leitor do registro de PII (dono: conversation-router). Decifra com KMS só no momento
    da leitura autenticada do backoffice; nada é persistido nem logado."""

    def __init__(self, dynamodb_client: Any, kms_client: Any, table_name: str = "sdr-pii") -> None:
        self._db = dynamodb_client
        self._kms = kms_client
        self._table = table_name

    def load(self, session_id: str) -> dict[str, list[str]]:
        try:
            item = self._db.get_item(
                TableName=self._table,
                Key={"PK": {"S": f"PII#{session_id}"}, "SK": {"S": "PII"}},
            ).get("Item")
            if not item:
                return {}
            blob = base64.b64decode(item["ciphertext"]["S"])
            return json.loads(self._kms.decrypt(CiphertextBlob=blob)["Plaintext"].decode())
        except Exception as exc:
            log_event("pii_read_failed", level=logging.WARNING, error=str(exc))
            return {}
