"""
explainability.py
=================
Wraps any LLM response with source attribution so users always know
which protocol or document the assistant drew from.
"""

from __future__ import annotations


_SOURCE_HEADER = "\n\n---\n📚 **Fontes consultadas:**\n"


def attach_sources(response: str, sources: list[str]) -> str:
    """
    Append a formatted sources block to `response`.
    If no sources, note that the response was generated without retrieval context.
    """
    if not sources:
        return (
            response
            + "\n\n---\n"
            + "ℹ️ *Resposta gerada sem correspondência direta nos protocolos internos. "
            "Baseada no treinamento geral do modelo.*"
        )

    source_block = _SOURCE_HEADER
    for i, src in enumerate(sources, 1):
        source_block += f"{i}. {src}\n"
    return response + source_block


def build_explainability_footer(
    agent_steps: list[str],
    urgency_level: str,
    human_approval_required: bool,
) -> str:
    """
    Build a transparency footer describing which agents ran and their conclusions.
    """
    lines = ["---", "🔍 **Rastreabilidade do fluxo:**"]
    for step in agent_steps:
        lines.append(f"- {step}")

    urgency_labels = {"low": "Baixa 🟢", "medium": "Moderada 🟡", "high": "Alta 🔴", "": "Não classificada"}
    lines.append(f"- Urgência: {urgency_labels.get(urgency_level, urgency_level)}")

    if human_approval_required:
        lines.append(
            "- ✋ **Aprovação médica necessária** antes de implementar sugestões farmacológicas."
        )

    return "\n".join(lines)
