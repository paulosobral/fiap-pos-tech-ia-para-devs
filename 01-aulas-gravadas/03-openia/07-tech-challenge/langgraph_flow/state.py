"""
state.py
========
Esquema de estado compartilhado para o fluxo médico LangGraph.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

import operator


class PatientState(TypedDict):
    patient_id: int
    patient_info: dict[str, Any]
    symptoms: str
    urgency_level: str           # "low" | "medium" | "high"
    pending_exams: list[dict]
    completed_exams: list[dict]
    active_medications: list[dict]
    diagnoses: list[dict]
    differential_diagnosis: str
    diagnosis_low_evidence: bool
    suggested_treatment: str
    alerts: Annotated[list[str], operator.add]
    human_approval_required: bool
    sources: Annotated[list[str], operator.add]
    agent_steps: Annotated[list[str], operator.add]
    final_response: str
