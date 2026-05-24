"""
alert_agent.py
==============
Emits structured alerts for high-urgency cases.
Writes to the audit log and returns alert messages to be displayed in the UI.
"""

from __future__ import annotations

import datetime

from langgraph_flow.state import PatientState
from security.audit_logger import get_audit_logger

_logger = get_audit_logger()


def alert_node(state: PatientState) -> dict:
    patient_id = state["patient_id"]
    urgency = state.get("urgency_level", "unknown")
    symptoms = state.get("symptoms", "")
    patient_info = state.get("patient_info", {})
    patient_name = patient_info.get("name", f"ID {patient_id}")

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    alert_message = (
        f"🚨 ALERTA MÉDICO — {timestamp}\n"
        f"Paciente: {patient_name} (ID {patient_id})\n"
        f"Urgência: {urgency.upper()}\n"
        f"Sintomas relatados: {symptoms[:300]}\n"
        f"Ação recomendada: Avaliação médica IMEDIATA. Acione equipe de plantão."
    )

    # Registro estruturado no log de auditoria.
    _logger.warning(
        "HIGH_URGENCY_ALERT",
        extra={
            "event": "high_urgency_alert",
            "patient_id": patient_id,
            "patient_name": patient_name,
            "urgency_level": urgency,
            "symptoms": symptoms[:300],
            "timestamp": timestamp,
        },
    )

    step_msg = (
        f"[Alerta] ALERTA DE URGÊNCIA ALTA emitido para {patient_name}. "
        "Equipe médica notificada no log de auditoria."
    )

    return {
        "alerts": [alert_message],
        "agent_steps": [step_msg],
    }
