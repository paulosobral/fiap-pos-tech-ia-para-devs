"""
triage_agent.py
===============
Classifies patient urgency (low / medium / high) based on reported symptoms.
Uses keyword heuristics + the RAG chain for context enrichment.
"""

from __future__ import annotations

import re

from langgraph_flow.state import PatientState

_HIGH_URGENCY_KEYWORDS = [
    "parada cardíaca", "parada respiratória", "iam", "infarto",
    "avc", "acidente vascular", "choque", "sepse grave", "séptico",
    "convulsão em andamento", "glasgow < 8", "glasgow menor",
    "dispneia grave", "saturação < 88", "saturação menor",
    "hemorragia maciça", "sangramento ativo",
    "pressão arterial < 80", "pressão menor que 80",
    "inconsciente", "coma",
]

_MEDIUM_URGENCY_KEYWORDS = [
    "dor torácica", "dispneia moderada", "confusão mental", "síncope",
    "febre alta", "temperatura > 39", "glicemia > 300", "hipoglicemia",
    "dor abdominal intensa", "vômitos persistentes", "cefaleia intensa",
    "edema agudo", "pressão alta", "hipertensão grave",
]


def _classify_urgency(symptoms: str) -> str:
    symptoms_lower = symptoms.lower()
    for kw in _HIGH_URGENCY_KEYWORDS:
        if kw in symptoms_lower:
            return "high"
    for kw in _MEDIUM_URGENCY_KEYWORDS:
        if kw in symptoms_lower:
            return "medium"
    return "low"


def triage_node(state: PatientState) -> dict:
    symptoms = state.get("symptoms", "")
    urgency = _classify_urgency(symptoms)

    urgency_labels = {"low": "Baixa", "medium": "Moderada", "high": "Alta"}
    step_msg = (
        f"[Triagem] Sintomas analisados: '{symptoms[:100]}'. "
        f"Urgência classificada como: {urgency_labels[urgency]}."
    )

    alert = []
    if urgency == "high":
        alert = [f"ALERTA TRIAGEM: Paciente com urgência ALTA. Sintomas: {symptoms[:200]}"]

    return {
        "urgency_level": urgency,
        "agent_steps": [step_msg],
        "alerts": alert,
    }
