"""
diagnosis_agent.py
==================
Generates a differential diagnosis using the RAG chain.
Patient context (conditions, completed exams, medications) is injected into the prompt.
"""

from __future__ import annotations

import json

from assistant.rag_chain import ask
from langgraph_flow.state import PatientState


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
        f"Com base nos sintomas relatados — '{symptoms}' — e no histórico clínico do paciente, "
        "elabore um diagnóstico diferencial detalhado com as hipóteses mais prováveis, "
        "condutas recomendadas e exames que poderiam confirmar cada hipótese."
    )

    result = ask(rag_chain, question, patient_context=patient_context)
    answer = result["answer"]
    sources = result["sources"]

    step_msg = (
        f"[Diagnóstico] Diagnóstico diferencial gerado com base nos sintomas e "
        f"{len(sources)} protocolo(s) relevante(s)."
    )

    return {
        "differential_diagnosis": answer,
        "sources": sources,
        "agent_steps": [step_msg],
    }
