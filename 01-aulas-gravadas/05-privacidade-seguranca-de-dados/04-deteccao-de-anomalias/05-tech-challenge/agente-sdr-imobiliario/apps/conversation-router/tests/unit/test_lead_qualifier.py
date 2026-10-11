import pytest

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

    def test_budget_low_value_scores_less_than_threshold(self):
        result = self.q.calculate_score({**self.full_info(), "budget": "R$ 60"})
        assert "orçamento definido e alto" not in result["factors"]
        assert result["score"] == 85

    def test_budget_unit_variants_normalized_to_high(self):
        for budget in ("R$ 50k", "50 mil", "60.000", "1,2 milhão", "50000"):
            result = self.q.calculate_score({**self.full_info(), "budget": budget})
            assert "orçamento definido e alto" in result["factors"], budget
            assert result["score"] == 100

    def test_budget_value_millions_scale(self):
        assert self.q._budget_value("R$ 1.500.000") == pytest.approx(1_500_000)
        assert self.q._budget_value("1,5 milhão") == pytest.approx(1_500_000)
        assert self.q._budget_value("1,5 milhões") == pytest.approx(1_500_000)

    def test_budget_value_thousands_and_mil(self):
        assert self.q._budget_value("R$ 1.500") == pytest.approx(1_500)
        assert self.q._budget_value("R$ 60 mil") == pytest.approx(60_000)
        assert self.q._budget_value("R$ 1.500.000") > self.q._budget_value("R$ 1.500")

    def test_deadline_months_weeks_and_days_normalized(self):
        assert self.q._deadline_months("2 semanas") == pytest.approx(14 / 30)
        assert self.q._deadline_months("15 dias") == pytest.approx(15 / 30)
        assert self.q._deadline_months("3 meses") == pytest.approx(3)

    def test_short_deadline_in_weeks_counts_as_urgent(self):
        result = self.q.calculate_score({**self.full_info(), "deadline": "2 semanas"})
        assert "prazo curto" in result["factors"]
        assert result["score"] == 100

    def test_short_deadline_in_days_counts_as_urgent(self):
        result = self.q.calculate_score({**self.full_info(), "deadline": "15 dias"})
        assert "prazo curto" in result["factors"]
        assert result["score"] == 100

    def test_urgency_labels_follow_score_factors(self):
        assert self.q.urgency({"deadline": "2 meses", "budget": "R$ 1.500.000"}) == "high"
        assert self.q.urgency({"deadline": "6 meses", "budget": "R$ 1.500.000"}) == "medium"
        assert self.q.urgency({"deadline": "2 meses", "budget": "R$ 60"}) == "medium"
        assert self.q.urgency({"deadline": "6 meses", "budget": "R$ 60"}) == "low"
        assert self.q.urgency({}) == "low"

    def test_route_rotation_up_to_500(self):
        rotation = ["ana", "bruno"]
        assert self.q.route(400, rotation, "diretor") == "ana"
        assert self.q.route(450, rotation, "diretor") == "bruno"

    def test_route_specialist_above_500(self):
        assert self.q.route(800, ["ana", "bruno"], "diretor") == "diretor"
