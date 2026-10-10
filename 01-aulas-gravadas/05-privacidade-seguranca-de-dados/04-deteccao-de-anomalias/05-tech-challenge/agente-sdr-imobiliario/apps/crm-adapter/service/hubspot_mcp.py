"""CRM real: HubSpot via MCP remoto (https://mcp.hubspot.com), OAuth 2.1 + PKCE.

O refresh token do HubSpot é de USO ÚNICO: cada renovação devolve um novo e invalida o
anterior, então ele vive no Secrets Manager (sdr/hubspot-mcp) e é regravado a cada renovação.
O access token fica em cache na instância da Lambda até perto de expirar.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from service.crm_gateway import CrmError, log_event

MCP_URL = "https://mcp.hubspot.com"
TOKEN_URL = "https://mcp.hubspot.com/oauth/v3/token"
_EXPIRY_MARGIN = 60

# Estágio da esteira (u1) -> "Lead status" padrão do contato no HubSpot (sem propriedade custom).
STAGE_TO_LEAD_STATUS = {
    "novo": "NEW",
    "qualificado": "OPEN",
    "contato-feito": "CONNECTED",
    "visita-agendada": "IN_PROGRESS",
    "handoff": "OPEN_DEAL",
    "ganho": "OPEN_DEAL",
    "perdido": "UNQUALIFIED",
}


class SecretsManagerTokenStore:
    """Credenciais OAuth no Secrets Manager: {"client_id","client_secret","refresh_token"}."""

    def __init__(self, secrets_client: Any, secret_id: str) -> None:
        self._sm = secrets_client
        self._secret_id = secret_id

    def load(self) -> dict[str, str]:
        raw = self._sm.get_secret_value(SecretId=self._secret_id)["SecretString"]
        return json.loads(raw)

    def save(self, creds: dict[str, str]) -> None:
        self._sm.put_secret_value(SecretId=self._secret_id, SecretString=json.dumps(creds))


class HubSpotMcpClient:
    """`call_tool(name, args) -> dict` síncrono sobre o SDK MCP (async, streamable HTTP)."""

    def __init__(
        self,
        store: Any,
        http_post: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._store = store
        self._post = http_post
        self._clock = clock
        self._session_factory = session_factory or self._sdk_session
        self._access_token: str | None = None
        self._expires_at = 0.0

    def _token(self) -> str:
        if self._access_token and self._clock() < self._expires_at - _EXPIRY_MARGIN:
            return self._access_token
        creds = self._store.load()
        post = self._post
        if post is None:
            import requests

            post = requests.post
        resp = post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": creds["refresh_token"],
            },
            timeout=20,
        )
        if resp.status_code != 200:
            log_event("hubspot_token_refresh_failed", level=logging.ERROR, status=resp.status_code)
            raise CrmError("hubspot token refresh failed")
        body = resp.json()
        if body.get("refresh_token"):
            self._store.save({**creds, "refresh_token": body["refresh_token"]})
        self._access_token = body["access_token"]
        self._expires_at = self._clock() + float(body.get("expires_in", 1800))
        return self._access_token

    @staticmethod
    def _sdk_session(access_token: str) -> Any:  # pragma: no cover - rede real
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        class _Ctx:
            async def __aenter__(self_inner):
                http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {access_token}"})
                self_inner._transport = streamable_http_client(MCP_URL, http_client=http)
                streams = await self_inner._transport.__aenter__()
                self_inner._session = ClientSession(streams[0], streams[1])
                await self_inner._session.__aenter__()
                await self_inner._session.initialize()
                return self_inner._session

            async def __aexit__(self_inner, *exc):
                await self_inner._session.__aexit__(*exc)
                await self_inner._transport.__aexit__(*exc)

        return _Ctx()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        token = self._token()

        async def run() -> Any:
            async with self._session_factory(token) as session:
                return await session.call_tool(name, arguments)

        result = asyncio.run(run())
        text = "".join(getattr(c, "text", "") for c in getattr(result, "content", []) or [])
        if getattr(result, "is_error", False):
            raise CrmError(f"hubspot tool {name} error: {text[:200]}")
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise CrmError(f"hubspot tool {name} returned non-json") from exc


# O resumo no campo `message` do contato é lido pelo corretor: códigos em inglês viram português.
_INTENT_PT = {"purchase": "compra", "rent": "locação", "investment": "investimento"}
_URGENCY_PT = {"high": "alta", "medium": "média", "low": "baixa"}


def _split_name(full: str) -> tuple[str, str]:
    parts = (full or "").split(None, 1)
    if not parts:
        return "", ""
    return parts[0], parts[1] if len(parts) > 1 else ""


class HubSpotCrmGateway:
    """CrmGateway sobre o HubSpot MCP: lead = contato (nome, e-mail, telefone, lead status).

    A qualificação (score, urgência, orçamento, prazo, área) vai no campo padrão `message`
    do contato, sem criar propriedades customizadas. Contato é localizado por e-mail ou
    telefone. O estágio já conhecido fica em cache na instância (get_lead).
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._cache: dict[str, dict[str, Any]] = {}

    def upsert_lead(self, lead: dict[str, Any]) -> dict[str, Any]:
        lead_id = str(lead.get("lead_id") or "")
        first, last = _split_name(str(lead.get("name") or ""))
        props = {
            k: v
            for k, v in {
                "firstname": first,
                "lastname": last,
                "email": str(lead.get("email") or ""),
                "phone": str(lead.get("phone") or ""),
                "message": self._summary(lead),
            }.items()
            if v
        }
        contact_id = self._find_contact(lead)
        if contact_id is None:
            res = self._call(
                "manage_crm_objects",
                {
                    "confirmationStatus": "CONFIRMATION_WAIVED_FOR_SESSION",
                    "createRequest": {"objects": [{"objectType": "CONTACT", "properties": props}]},
                },
            )
            contact_id = self._extract_id(res)
        elif props:
            self._call(
                "manage_crm_objects",
                {
                    "confirmationStatus": "CONFIRMATION_WAIVED_FOR_SESSION",
                    "updateRequest": {
                        "objects": [{"objectType": "CONTACT", "objectId": int(contact_id), "properties": props}]
                    },
                },
            )
        record = {"crm_id": str(contact_id or ""), "lead_id": lead_id, "stage": self._cache.get(lead_id, {}).get("stage")}
        self._cache[lead_id] = record
        return record

    def get_lead(self, lead_id: str) -> dict[str, Any] | None:
        return self._cache.get(lead_id)

    def update_stage(self, lead_id: str, stage: str) -> None:
        record = self._cache.get(lead_id)
        if not record or not record.get("crm_id"):
            raise CrmError(f"lead {lead_id} not synced to hubspot yet")
        status = STAGE_TO_LEAD_STATUS.get(stage)
        if status:
            self._call(
                "manage_crm_objects",
                {
                    "confirmationStatus": "CONFIRMATION_WAIVED_FOR_SESSION",
                    "updateRequest": {
                        "objects": [
                            {
                                "objectType": "CONTACT",
                                "objectId": int(record["crm_id"]),
                                "properties": {"hs_lead_status": status},
                            }
                        ]
                    },
                },
            )
        record["stage"] = stage

    def _find_contact(self, lead: dict[str, Any]) -> str | None:
        for prop, value in (("email", lead.get("email")), ("phone", lead.get("phone"))):
            if not value:
                continue
            res = self._call(
                "search_crm_objects",
                {
                    "objectType": "CONTACT",
                    "limit": 1,
                    "properties": ["email", "phone"],
                    "filterGroups": [{"filters": [{"propertyName": prop, "operator": "EQ", "value": str(value)}]}],
                },
            )
            results = res.get("results") or []
            if results:
                return str(results[0]["id"])
        return None

    @staticmethod
    def _extract_id(res: dict[str, Any]) -> str | None:
        """Aceita as formas conhecidas de resposta de criação; id ausente vira erro explícito."""
        stack = [res]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key in ("id", "objectId"):
                    if node.get(key):
                        return str(node[key])
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        raise CrmError("hubspot create returned no object id")

    @staticmethod
    def _summary(lead: dict[str, Any]) -> str:
        parts = [
            ("Score", lead.get("score")),
            ("Urgência", _URGENCY_PT.get(str(lead.get("urgency")), lead.get("urgency"))),
            ("Intenção", _INTENT_PT.get(str(lead.get("intent")), lead.get("intent"))),
            ("Orçamento", lead.get("budget")),
            ("Prazo", lead.get("deadline")),
            ("Área", lead.get("area")),
            ("Região", lead.get("region")),
            ("Imóvel escolhido", lead.get("property")),
        ]
        body = " | ".join(f"{k}: {v}" for k, v in parts if v not in (None, ""))
        return f"SDR W Levitt (lead {lead.get('lead_id')}) — {body}" if body else ""

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._client.call_tool(name, arguments)
        except CrmError:
            raise
        except Exception as exc:
            log_event("hubspot_tool_failed", level=logging.ERROR, tool=name, error=str(exc))
            raise CrmError(f"hubspot tool {name} failed") from exc
