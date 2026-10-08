import contextlib
import json
from types import SimpleNamespace

import pytest

from service.crm_gateway import CrmError
from service.hubspot_mcp import HubSpotCrmGateway, HubSpotMcpClient, SecretsManagerTokenStore


class FakeStore:
    def __init__(self):
        self.creds = {"client_id": "cid", "client_secret": "sec", "refresh_token": "r1"}
        self.saved = []

    def load(self):
        return dict(self.creds)

    def save(self, creds):
        self.creds = creds
        self.saved.append(creds)


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self.result


def make_client(result, post=None, clock=lambda: 1000.0):
    session = FakeSession(result)

    @contextlib.asynccontextmanager
    async def factory(token):
        session.token = token
        yield session

    post = post or (lambda url, data=None, timeout=None: FakeResp(200, {"access_token": "AT", "refresh_token": "r2", "expires_in": 1800}))
    store = FakeStore()
    return HubSpotMcpClient(store, http_post=post, clock=clock, session_factory=factory), store, session


def text_result(payload, is_error=False):
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))], is_error=is_error)


class TestMcpClient:
    def test_refresh_rotates_token_and_calls_tool(self):
        client, store, session = make_client(text_result({"results": []}))
        assert client.call_tool("search_crm_objects", {"objectType": "CONTACT"}) == {"results": []}
        assert session.token == "AT"
        assert store.creds["refresh_token"] == "r2"  # refresh de uso único regravado

    def test_access_token_cached_until_expiry(self):
        calls = []
        clock = [1000.0]

        def post(url, data=None, timeout=None):
            calls.append(data["refresh_token"])
            return FakeResp(200, {"access_token": "AT", "refresh_token": f"r{len(calls) + 1}", "expires_in": 1800})

        client, store, _ = make_client(text_result({}), post=post, clock=lambda: clock[0])
        client.call_tool("x", {})
        client.call_tool("x", {})
        assert len(calls) == 1
        clock[0] += 1800
        client.call_tool("x", {})
        assert calls == ["r1", "r2"]

    def test_refresh_failure_raises(self):
        client, _, _ = make_client(text_result({}), post=lambda *a, **k: FakeResp(400))
        with pytest.raises(CrmError):
            client.call_tool("x", {})

    def test_tool_error_and_non_json(self):
        client, _, _ = make_client(text_result({"m": "boom"}, is_error=True))
        with pytest.raises(CrmError):
            client.call_tool("x", {})
        bad = SimpleNamespace(content=[SimpleNamespace(text="oi")], is_error=False)
        client, _, _ = make_client(bad)
        with pytest.raises(CrmError):
            client.call_tool("x", {})


class ScriptedClient:
    def __init__(self, search_results=None, create_result=None):
        self.calls = []
        self.search_results = search_results or []
        self.create_result = create_result or {"results": [{"succeeded": True, "successDetails": {"id": 777}}]}

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "search_crm_objects":
            return {"results": self.search_results.pop(0) if self.search_results else []}
        return self.create_result


LEAD = {"lead_id": "L1", "name": "Ana Souza", "email": "ana@x.com", "phone": "11999990000",
        "score": 80, "urgency": "high", "intent": "compra", "budget": "5 mil", "area": "200 m²"}


class TestGateway:
    def test_creates_contact_when_not_found(self):
        client = ScriptedClient()
        record = HubSpotCrmGateway(client).upsert_lead(LEAD)
        assert record["crm_id"] == "777"
        name, args = client.calls[-1]
        assert name == "manage_crm_objects"
        obj = args["createRequest"]["objects"][0]
        assert obj["objectType"] == "CONTACT"
        assert obj["properties"]["firstname"] == "Ana" and obj["properties"]["lastname"] == "Souza"
        assert obj["properties"]["email"] == "ana@x.com"
        assert "Score: 80" in obj["properties"]["message"]
        assert args["confirmationStatus"] == "CONFIRMATION_WAIVED_FOR_SESSION"

    def test_updates_existing_contact_found_by_email(self):
        client = ScriptedClient(search_results=[[{"id": 55}]])
        record = HubSpotCrmGateway(client).upsert_lead(LEAD)
        assert record["crm_id"] == "55"
        name, args = client.calls[-1]
        assert args["updateRequest"]["objects"][0]["objectId"] == 55

    def test_falls_back_to_phone_search(self):
        client = ScriptedClient(search_results=[[], [{"id": 9}]])
        assert HubSpotCrmGateway(client).upsert_lead(LEAD)["crm_id"] == "9"
        searches = [c for c in client.calls if c[0] == "search_crm_objects"]
        assert [s[1]["filterGroups"][0]["filters"][0]["propertyName"] for s in searches] == ["email", "phone"]

    def test_update_stage_sets_lead_status_and_caches(self):
        client = ScriptedClient()
        gw = HubSpotCrmGateway(client)
        gw.upsert_lead(LEAD)
        gw.update_stage("L1", "qualificado")
        _, args = client.calls[-1]
        assert args["updateRequest"]["objects"][0]["properties"] == {"hs_lead_status": "OPEN"}
        assert gw.get_lead("L1")["stage"] == "qualificado"

    def test_update_stage_without_sync_fails(self):
        with pytest.raises(CrmError):
            HubSpotCrmGateway(ScriptedClient()).update_stage("L1", "novo")

    def test_create_without_id_is_error(self):
        with pytest.raises(CrmError):
            HubSpotCrmGateway(ScriptedClient(create_result={"results": []})).upsert_lead(LEAD)

    def test_client_failure_becomes_crm_error(self):
        class Boom:
            def call_tool(self, *a):
                raise RuntimeError("net")

        with pytest.raises(CrmError):
            HubSpotCrmGateway(Boom()).upsert_lead(LEAD)


def test_secret_store_roundtrip():
    class Sm:
        def __init__(self):
            self.v = json.dumps({"refresh_token": "a"})

        def get_secret_value(self, SecretId):
            return {"SecretString": self.v}

        def put_secret_value(self, SecretId, SecretString):
            self.v = SecretString

    store = SecretsManagerTokenStore(Sm(), "sdr/hubspot-mcp")
    assert store.load() == {"refresh_token": "a"}
    store.save({"refresh_token": "b"})
    assert store.load()["refresh_token"] == "b"
