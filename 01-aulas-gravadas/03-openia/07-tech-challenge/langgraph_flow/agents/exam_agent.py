"""
exam_agent.py
=============
Fetches pending and completed exams for the patient from the SQLite database.
"""

from __future__ import annotations

from assistant.patient_db import get_completed_exams, get_pending_exams
from langgraph_flow.state import PatientState


def exam_node(state: PatientState) -> dict:
    patient_id = state["patient_id"]

    pending = get_pending_exams(patient_id)
    completed = get_completed_exams(patient_id)

    if pending:
        pending_names = ", ".join(e["exam_name"] for e in pending)
        step_msg = f"[Exames] {len(pending)} exame(s) pendente(s): {pending_names}."
    else:
        step_msg = "[Exames] Nenhum exame pendente."

    if completed:
        completed_summary = "; ".join(
            f"{e['exam_name']}: {e['result'] or 'sem resultado'}"
            for e in completed[:5]
        )
        step_msg += f" Exames concluídos (últimos {min(5, len(completed))}): {completed_summary}."

    return {
        "pending_exams": pending,
        "completed_exams": completed,
        "agent_steps": [step_msg],
    }
