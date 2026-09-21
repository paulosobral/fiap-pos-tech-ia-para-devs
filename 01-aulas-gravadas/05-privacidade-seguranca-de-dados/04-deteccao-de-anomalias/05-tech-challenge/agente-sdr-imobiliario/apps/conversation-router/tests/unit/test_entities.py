import re

from service.entities import (
    VALID_STATES,
    Conversation,
    Lead,
    conversation_ttl_seconds,
    looks_like_phone,
)


class TestLead:
    def test_to_item_required_fields(self):
        lead = Lead(lead_id="l1", telegram_user_id=42)
        item = lead.to_item()
        assert item["lead_id"] == "l1"
        assert item["telegram_user_id"] == 42
        assert item["status"] == "new"
        assert item["created_at"]

    def test_to_item_omits_none_optionals(self):
        item = Lead(lead_id="l1", telegram_user_id=1).to_item()
        assert "score" not in item
        assert "intent" not in item

    def test_roundtrip_from_item(self):
        original = Lead(lead_id="l2", telegram_user_id=7, score=85, intent="rent", status="qualified")
        restored = Lead.from_item(original.to_item())
        assert restored.score == 85
        assert restored.intent == "rent"
        assert restored.status == "qualified"

    def test_route_roundtrip_from_item(self):
        original = Lead(lead_id="l3", telegram_user_id=3, route="diretor")
        restored = Lead.from_item(original.to_item())
        assert restored.route == "diretor"
        assert "route" not in Lead(lead_id="l4", telegram_user_id=4).to_item()


class TestConversation:
    def test_ttl_is_90_days(self):
        assert conversation_ttl_seconds() == 7_776_000

    def test_to_item_has_ttl(self):
        conv = Conversation(session_id="s1", lead_id="l1")
        assert conv.to_item()["ttl"] == 7_776_000
        assert conv.to_item()["consent_recorded"] is False

    def test_roundtrip_from_item(self):
        original = Conversation(session_id="s1", lead_id="l1", current_state="qualification", pii_masked=True)
        restored = Conversation.from_item(original.to_item())
        assert restored.current_state == "qualification"
        assert restored.pii_masked is True

    def test_valid_states_include_all_flow_states(self):
        assert set(VALID_STATES) == {
            "greeting", "elicitation", "intent", "qualification",
            "recommendation", "scheduling", "handoff", "followup",
        }

    def test_looks_like_phone(self):
        assert looks_like_phone("+55 11 91234-5678")
        assert not looks_like_phone("sem telefone aqui")
