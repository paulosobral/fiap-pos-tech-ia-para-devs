from __future__ import annotations

from typing import Any

INTENT_PHRASES = {
    "compra": "à compra",
    "locacao": "à locação",
    "locação": "à locação",
    "investimento": "para investimento",
}


class FollowupMessageBuilder:
    """FR8.2 — mensagem de retomada com contexto resumido, PII-safe.

    Recebe apenas um resumo derivado da conversa (`step`, `intent`, `state`) —
    nunca as mensagens brutas nem o contato do lead — então nenhuma PII pode
    vazar para a mensagem de follow-up.
    """

    def build(self, summary: dict[str, Any]) -> str:
        try:
            step = int(summary.get("step") or 2)
        except (TypeError, ValueError):
            step = 2
        intent = str(summary.get("intent") or "").strip().lower()
        phrase = INTENT_PHRASES.get(intent)
        if step <= 2:
            if phrase:
                return (
                    f"Olá! Retomando seu atendimento {phrase} de espaço corporativo — "
                    "quer continuar de onde paramos?"
                )
            return "Olá! Retomando nosso atendimento sobre espaços corporativos — quer continuar de onde paramos?"
        if step <= 5:
            if phrase:
                return (
                    f"Oi! Ainda temos opções compatíveis com sua busca {phrase} de espaço "
                    "corporativo. Quer que eu mostre as novidades?"
                )
            return "Oi! Ainda temos opções compatíveis com o que você procura. Quer que eu mostre as novidades?"
        if phrase:
            return (
                f"Última mensagem por aqui: se sua busca {phrase} de espaço corporativo "
                "ainda vale a pena, me avise que retomo o atendimento na hora."
            )
        return (
            "Última mensagem por aqui: se ainda tiver interesse em espaços corporativos, "
            "me avise que retomo o atendimento na hora."
        )
