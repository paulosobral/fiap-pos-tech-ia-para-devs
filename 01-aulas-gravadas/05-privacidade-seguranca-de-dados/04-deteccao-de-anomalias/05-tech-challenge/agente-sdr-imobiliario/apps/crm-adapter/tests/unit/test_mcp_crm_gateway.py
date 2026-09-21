from unittest.mock import MagicMock

import pytest

import service.crm_gateway as crm_gateway_module
from service.crm_gateway import CrmError, McpCrmGateway, McpUnavailableError, build_hubspot_client


def lead(**overrides):
    base = {
        "lead_id": "lead-1",
        "name": "Ana Ribeiro",
        "email": "ana@empresa.com",
        "phone": "+5511999990000",
        "score": 85,
        "urgency": "alta",
        "intent": "compra",
        "session_id": "s1",
    }
    base.update(overrides)
    return base


def fake_mcp(record=None):
    client = MagicMock()
    client.call_tool.return_value = {"record": record or {"crm_id": "hs-1", "lead_id": "lead-1"}}
    return client


class TestMcpCrmGateway:
    def test_upsert_sends_properties_and_returns_record(self):
        client = fake_mcp()
        gateway = McpCrmGateway(client)
        record = gateway.upsert_lead(lead())
        assert record == {"crm_id": "hs-1", "lead_id": "lead-1"}
        name, arguments = client.call_tool.call_args[0]
        assert name == "crm_upsert_lead"
        assert arguments["properties"]["email"] == "ana@empresa.com"

    def test_get_lead_returns_record(self):
        client = fake_mcp({"crm_id": "hs-1"})
        gateway = McpCrmGateway(client)
        assert gateway.get_lead("lead-1") == {"crm_id": "hs-1"}

    def test_get_lead_missing_returns_none(self):
        client = fake_mcp()
        client.call_tool.return_value = {"record": None}
        gateway = McpCrmGateway(client)
        assert gateway.get_lead("ghost") is None

    def test_update_stage_calls_tool(self):
        client = fake_mcp()
        gateway = McpCrmGateway(client)
        gateway.update_stage("lead-1", "qualificado")
        client.call_tool.assert_called_once_with(
            "crm_update_stage", {"lead_id": "lead-1", "stage": "qualificado"}
        )

    def test_tool_failure_raises_crm_error(self):
        client = fake_mcp()
        client.call_tool.side_effect = RuntimeError("mcp down")
        gateway = McpCrmGateway(client)
        with pytest.raises(CrmError):
            gateway.upsert_lead(lead())

    def test_non_object_result_raises_crm_error(self):
        client = fake_mcp()
        client.call_tool.return_value = "garbage"
        gateway = McpCrmGateway(client)
        with pytest.raises(CrmError):
            gateway.update_stage("lead-1", "novo")


class TestMcpAvailability:
    def test_build_client_raises_without_sdk(self, monkeypatch):
        monkeypatch.setattr(crm_gateway_module, "_HAS_MCP", False)
        with pytest.raises(McpUnavailableError):
            build_hubspot_client()

    def test_build_client_with_sdk_present(self, monkeypatch):
        monkeypatch.setattr(crm_gateway_module, "_HAS_MCP", True)
        assert build_hubspot_client() is crm_gateway_module.ClientSession