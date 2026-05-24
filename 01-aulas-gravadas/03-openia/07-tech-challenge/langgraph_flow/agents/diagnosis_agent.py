"""
diagnosis_agent.py
==================
Gera um diagnóstico diferencial usando a cadeia RAG.
O contexto do paciente (condições, exames realizados, medicamentos) é injetado no prompt.
"""

from __future__ import annotations

from assistant.rag_chain import ask
from langgraph_flow.state import PatientState


_SAFE_FALLBACK_DIAGNOSIS = (
    "Evidência insuficiente nos protocolos recuperados para montar um diagnóstico diferencial "
    "confiável no momento.\n"
    "- Hipótese 1: Não conclusiva com base documental atual.\n"
    "  Evidências: insuficientes no contexto recuperado.\n"
    "  Exames para confirmar: avaliação clínica presencial e exames direcionados conforme protocolo.\n"
    "- Limitações de evidência: ausência de recuperação semântica forte para os sintomas informados."
)


def _looks_invalid_diagnosis(answer: str) -> bool:
    upper = answer.upper()
    red_flags = [
        "NÃO ENCONTRAR",
        "CUIDADOS INADEQUADOS",
        "SINTOMAS DE CID",
        "HIPÓTESES CONFIRMADAS",
    ]
    return any(flag in upper for flag in red_flags)


def _build_patient_context(state: PatientState) -> str:
    info = state.get("patient_info", {})
    lines = [
        f"Paciente: {info.get('name', 'Desconhecido')}, {info.get('age', '?')} anos, sexo {info.get('sex', '?')}.",
        f"Condições conhecidas: {info.get('conditions', 'Nenhuma informada')}.",
        f"Alergias: {info.get('allergies', 'Nenhuma conhecida')}.",
    ]

    completed = state.get("completed_exams", [])
    if completed:
        exam_lines = [
            f"  - {e['exam_name']}: {e.get('result', 'sem resultado')} ({e.get('completed_date', '')})"
            for e in completed[:6]
        ]
        lines.append("Exames recentes:\n" + "\n".join(exam_lines))

    meds = state.get("active_medications", [])
    if meds:
        med_lines = [f"  - {m['name']} {m.get('dose', '')} {m.get('frequency', '')}" for m in meds]
        lines.append("Medicações em uso:\n" + "\n".join(med_lines))

    prev_diags = state.get("diagnoses", [])
    if prev_diags:
        diag_lines = [f"  - {d['description']} (CID: {d.get('cid10', '?')})" for d in prev_diags[:4]]
        lines.append("Diagnósticos anteriores:\n" + "\n".join(diag_lines))

    return "\n".join(lines)


def diagnosis_node(state: PatientState, rag_chain) -> dict:
    symptoms = state.get("symptoms", "")
    patient_context = _build_patient_context(state)

    question = (
        f"Sintomas relatados: '{symptoms}'.\n\n"
        "Tarefa: elaborar diagnóstico diferencial de forma conservadora, estritamente baseado "
        "nos protocolos recuperados e no contexto clínico informado.\n"
        "Regras obrigatórias:\n"
        "1) Não invente doenças, exames, doses ou termos.\n"
        "2) Se evidência for insuficiente, declare explicitamente limitação de evidência.\n"
        "3) Liste no máximo 3 hipóteses prováveis, em ordem de plausibilidade.\n"
        "4) Para cada hipótese, inclua sinais de suporte e exames de confirmação recomendados.\n"
        "5) Não inclua prescrição medicamentosa nesta etapa.\n"
        "Formato:\n"
        "- Hipótese 1: ...\n"
        "  Evidências: ...\n"
        "  Exames para confirmar: ...\n"
        "- Hipótese 2: ...\n"
        "- Hipótese 3: ...\n"
        "- Limitações de evidência: ..."
    )

    result = ask(rag_chain, question, patient_context=patient_context, use_history=False)
    answer = result["answer"].strip()
    sources = result["sources"]
    low_evidence = bool(result.get("low_evidence", False) or not sources)

    if low_evidence or _looks_invalid_diagnosis(answer):
        answer = _SAFE_FALLBACK_DIAGNOSIS
        low_evidence = True

    if low_evidence:
        step_msg = (
            "[Diagnóstico] Evidência insuficiente para hipótese confiável. "
            "Fluxo farmacológico bloqueado até revisão médica."
        )
    else:
        step_msg = (
            f"[Diagnóstico] Diagnóstico diferencial gerado com base nos sintomas e "
            f"{len(sources)} protocolo(s) relevante(s)."
        )

    return {
        "differential_diagnosis": answer,
        "diagnosis_low_evidence": low_evidence,
        "human_approval_required": low_evidence,
        "sources": sources,
        "agent_steps": [step_msg],
    }
