from service.flow.lead_qualifier import SCORE_THRESHOLD, LeadQualifier


class TestLeadQualifier:
    def setup_method(self):
        self.q = LeadQualifier()

    def full_info(self):
        return {
            "area": "1000 m²",
            "region": "Berrini",
            "budget": "R$ 50k/mês",
            "deadline": "3 meses",
            "people_count": 50,
            "decision_maker": "yes",
        }

    def test_complete_info_scores_high(self):
        result = self.q.calculate_score(self.full_info())
        assert result["score"] == 100
        assert result["missing"] == []

    def test_missing_fields_listed(self):
        info = {"area": "500 m²"}
        result = self.q.calculate_score(info)
        assert set(result["missing"]) == {"region", "budget", "deadline", "people_count", "decision_maker"}

    def test_score_below_threshold_not_qualified(self):
        assert not self.q.is_qualified(50)

    def test_score_at_threshold_qualified(self):
        assert self.q.is_qualified(SCORE_THRESHOLD)

    def test_explanation_natural_language(self):
        result = self.q.calculate_score(self.full_info())
        text = self.q.explain(result)
        assert text.startswith("Seu score é 100")
        assert "informações completas" in text

    def test_short_deadline_adds_urgency(self):
        result = self.q.calculate_score({**self.full_info()})
        assert "prazo curto" in result["factors"]

    def test_route_rotation_up_to_500(self):
        rotation = ["ana", "bruno"]
        assert self.q.route(400, rotation, "diretor") == "ana"
        assert self.q.route(450, rotation, "diretor") == "bruno"

    def test_route_specialist_above_500(self):
        assert self.q.route(800, ["ana", "bruno"], "diretor") == "diretor"
