"""
pharmacy_agent.py
=================
Suggests pharmacological options based on the differential diagnosis.

SAFETY GUARANTEE: Every response ALWAYS includes the mandatory disclaimer
that prescription requires physician validation. The LLM output is post-processed
to enforce this regardless of what the model generates.
"""

from __future__ import annotations

from assistant.rag_chain import ask
from langgraph_flow.state import PatientState
from security.guardrails import enforce_prescription_disclaimer

_MANDATORY_DISCLAIMER = (
    "\n\n⚠️ AVISO OBRIGATÓRIO: As sugestões farmacológicas acima são orientativas e baseadas em "
    "protocolos gerais. A prescrição, ajuste de dose e administração de qualquer medicamento "
    "REQUEREM validação, prescrição e assinatura do médico responsável pelo paciente. "
    "Este assistente NÃO substitui o julgamento clínico médico."
)


def pharmacy_node(state: PatientState, rag_chain) -> dict:
    if state.get("diagnosis_low_evidence", False):
        return {
            "suggested_treatment": (
                "Sugestões farmacológicas não geradas porque o diagnóstico está com baixa confiança "
                "documental. Reavalie sinais clínicos e exames antes de qualquer conduta medicamentosa."
            ) + _MANDATORY_DISCLAIMER,
            "sources": [],
            "human_approval_required": True,
            "agent_steps": [
                "[Farmácia] Etapa bloqueada por baixa evidência diagnóstica."
            ],
        }

    diagnosis = state.get("differential_diagnosis", "")
    patient_info = state.get("patient_info", {})
    allergies = patient_info.get("allergies", "Nenhuma conhecida")
    conditions = patient_info.get("conditions", "")
    meds = state.get("active_medications", [])
    current_meds_str = ", ".join(m["name"] for m in meds) if meds else "nenhuma"

    question = (
        f"Com base no seguinte diagnóstico diferencial:\n{diagnosis[:600]}\n\n"
        f"Quais são as opções farmacológicas recomendadas pelos protocolos? "
        f"Considere: alergias conhecidas do paciente ({allergies}), "
        f"condições concomitantes ({conditions}), "
        f"medicações em uso ({current_meds_str}). "
        "Liste classes de medicamentos e exemplos, indicando possíveis interações."
    )

    result = ask(rag_chain, question, use_history=False)
    raw_answer = result["answer"]
    sources = result["sources"]

    # Força o disclaimer de segurança independentemente da saída da LLM.
    safe_answer = enforce_prescription_disclaimer(raw_answer) + _MANDATORY_DISCLAIMER

    step_msg = (
        f"[Farmácia] Sugestões farmacológicas geradas. "
        "Requer validação médica antes de qualquer prescrição."
    )

    return {
        "suggested_treatment": safe_answer,
        "sources": sources,
        "human_approval_required": True,
        "agent_steps": [step_msg],
    }
