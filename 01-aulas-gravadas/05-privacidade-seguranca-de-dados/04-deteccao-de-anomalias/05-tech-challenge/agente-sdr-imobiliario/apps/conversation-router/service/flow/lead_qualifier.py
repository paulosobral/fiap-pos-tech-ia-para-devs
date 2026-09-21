from __future__ import annotations

from typing import Any

SCORE_THRESHOLD = 70
AREA_THRESHOLD_M2 = 500


class LeadQualifier:
    def calculate_score(self, info: dict[str, Any]) -> dict[str, Any]:
        missing = self.missing_fields(info)
        score = 0
        factors: list[str] = []

        if not missing:
            score += 20
            factors.append("informações completas")
        else:
            score += max(0, 20 - 5 * len(missing))

        if info.get("decision_maker") == "yes":
            score += 20
            factors.append("decisor definido")

        deadline_score = self._deadline_score(info.get("deadline"))
        score += deadline_score
        if deadline_score >= 20:
            factors.append("prazo curto")

        budget_score = self._budget_score(info.get("budget"))
        score += budget_score
        if budget_score >= 20:
            factors.append("orçamento definido e alto")

        score = min(score, 100)
        return {"score": score, "factors": factors, "missing": missing}

    def missing_fields(self, info: dict[str, Any]) -> list[str]:
        required = ("area", "region", "budget", "deadline", "people_count", "decision_maker")
        return [field for field in required if not info.get(field)]

    def explain(self, result: dict[str, Any]) -> str:
        parts = ", ".join(result["factors"]) if result["factors"] else "há informações pendentes"
        return f"Seu score é {result['score']} porque {parts}."

    def is_qualified(self, score: float) -> bool:
        return score >= SCORE_THRESHOLD

    def route(self, area_m2: float | None, rotation: list[str], specialist: str) -> str:
        if area_m2 is None or area_m2 <= AREA_THRESHOLD_M2:
            if not rotation:
                return specialist
            idx = self._rotation_idx % len(rotation)
            chosen = rotation[idx]
            self._rotation_idx += 1
            return chosen
        return specialist

    _rotation_idx = 0

    def _deadline_score(self, deadline: str | None) -> int:
        if not deadline:
            return 0
        months = self._deadline_months(deadline)
        if months is not None and months <= 3:
            return 30
        return 10

    def _budget_score(self, budget: str | None) -> int:
        if not budget:
            return 0
        import re

        match = re.search(r"(\d+)", str(budget).replace(".", "").replace(",", ""))
        if match and int(match.group(1)) >= 50:
            return 30
        return 15

    @staticmethod
    def _deadline_months(deadline: str) -> float | None:
        import re

        match = re.search(r"(\d+)\s*(mês|meses|month)", str(deadline).lower())
        if match:
            return float(match.group(1))
        return None
