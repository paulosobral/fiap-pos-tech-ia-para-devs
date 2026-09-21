from service.message_builder import FollowupMessageBuilder


class TestFollowupMessageBuilder:
    def test_day_two_message_is_gentle_reopen(self):
        message = FollowupMessageBuilder().build({"step": 2})
        assert "Retomando" in message
        assert "espaços corporativos" in message

    def test_day_two_message_uses_known_intent(self):
        message = FollowupMessageBuilder().build({"step": 2, "intent": "locação"})
        assert "à locação" in message

    def test_day_five_message_offers_options(self):
        message = FollowupMessageBuilder().build({"step": 5, "intent": "compra"})
        assert "à compra" in message
        assert "novidades" in message

    def test_day_nine_message_is_last_call(self):
        message = FollowupMessageBuilder().build({"step": 9, "intent": "investimento"})
        assert "Última mensagem" in message
        assert "para investimento" in message

    def test_unknown_step_and_intent_fall_back_gracefully(self):
        builder = FollowupMessageBuilder()
        assert "Última mensagem" in builder.build({"step": 7})
        assert "opções compatíveis" in builder.build({"step": 5, "intent": "desconhecida"})
        assert "Retomando" in builder.build({})

    def test_message_never_contains_pii_from_summary(self):
        summary = {
            "step": 2,
            "intent": "locação",
            "state": "qualification",
            "name": "Ana Souza",
            "email": "ana@empresa.com",
            "phone": "11988887777",
        }
        message = FollowupMessageBuilder().build(summary)
        assert "Ana" not in message
        assert "ana@empresa.com" not in message
        assert "11988887777" not in message
