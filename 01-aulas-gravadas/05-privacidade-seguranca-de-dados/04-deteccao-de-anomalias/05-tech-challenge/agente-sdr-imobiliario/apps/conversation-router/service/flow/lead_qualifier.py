from __future__ import annotations

import re
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

    def urgency(self, info: dict[str, Any]) -> str:
        """Rótulo low/medium/high para o handoff do CRM (entities.md/Contrato 4).

        Reutiliza os MESMOS fatores do score: prazo curto (R3: ≤ 3 meses, com
        semanas/dias normalizados) e ticket alto (orçamento ≥ 50k).
        high = os dois; medium = exatamente um; low = nenhum.
        """
        urgent_deadline = self._deadline_score(info.get("deadline")) >= 30
        high_budget = self._budget_score(info.get("budget")) >= 30
        if urgent_deadline and high_budget:
            return "high"
        if urgent_deadline or high_budget:
            return "medium"
        return "low"

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

    def _budget_value(self, budget: str) -> float | None:
        # pt-BR: "mil"/"k" = ×1000, "milhão" = ×1e6; pontos em \d{1,3}(\.\d{3})+ são separador de milhar.
        raw = str(budget).lower()
        multiplier = 1.0
        if re.search(r"milh", raw):
            multiplier = 1_000_000
        elif re.search(r"(\d\s*k\b|\bk\b|\bmil\b)", raw):
            multiplier = 1_000
        digits = re.sub(r"[^\d.,]", "", raw)
        if not digits:
            return None
        if "," in digits and "." in digits:
            digits = digits.replace(".", "").replace(",", ".")
        elif "," in digits:
            digits = digits.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", digits):
            digits = digits.replace(".", "")
        try:
            return float(digits) * multiplier
        except ValueError:
            return None

    def _budget_score(self, budget: str | None) -> int:
        if not budget:
            return 0
        value = self._budget_value(budget)
        if value is not None and value >= 50_000:
            return 30
        return 15

    @staticmethod
    def _deadline_months(deadline: str) -> float | None:
        """Normaliza prazo para meses: "2 semanas" ≈ 0,47, "15 dias" = 0,5 (base 30 dias/mês)."""
        match = re.search(r"(\d+)\s*(m[êe]s(?:es)?|month|semanas?|dias?)", str(deadline).lower())
        if not match:
            return None
        value = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("sem"):
            return value * 7 / 30
        if unit.startswith("dia"):
            return value / 30
        return float(value)
