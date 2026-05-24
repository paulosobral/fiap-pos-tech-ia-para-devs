"""
graph.py
========
StateGraph do LangGraph — fluxo multiagente do assistente médico.

Fluxo:
    triage_node
        ├─ urgency=high ──→ alert_node ──→ exam_node
        └─ urgency=low/medium ──────────→ exam_node
                                               │
                                          diagnosis_node
                                               │
                                     [ponto de interrupção humana]
                                               │
                                          pharmacy_node
                                               │
                                             END

Uso:
    from langgraph_flow.graph import build_graph
    graph = build_graph(rag_chain)
    result = graph.invoke(initial_state)
"""

from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from assistant.patient_db import (
    get_active_medications,
    get_diagnoses,
    get_patient,
)
from langgraph_flow.state import PatientState
from langgraph_flow.agents.triage_agent import triage_node
from langgraph_flow.agents.exam_agent import exam_node
from langgraph_flow.agents.diagnosis_agent import diagnosis_node
from langgraph_flow.agents.pharmacy_agent import pharmacy_node
from langgraph_flow.agents.alert_agent import alert_node
from security.audit_logger import log_agent_step


# ── Wrappers que injetam rag_chain nos agentes que precisam ──────────────────

def _diagnosis_node(state: PatientState, rag_chain) -> dict:
    result = diagnosis_node(state, rag_chain)
    log_agent_step("diagnosis_node", state["patient_id"], result.get("agent_steps", []))
    return result


def _pharmacy_node(state: PatientState, rag_chain) -> dict:
    result = pharmacy_node(state, rag_chain)
    log_agent_step("pharmacy_node", state["patient_id"], result.get("agent_steps", []))
    return result


def _triage_node(state: PatientState) -> dict:
    result = triage_node(state)
    log_agent_step("triage_node", state["patient_id"], result.get("agent_steps", []))
    return result


def _exam_node(state: PatientState) -> dict:
    result = exam_node(state)
    log_agent_step("exam_node", state["patient_id"], result.get("agent_steps", []))
    return result


def _alert_node(state: PatientState) -> dict:
    result = alert_node(state)
    log_agent_step("alert_node", state["patient_id"], result.get("agent_steps", []))
    return result


# ── Roteamento ────────────────────────────────────────────────────────────────

def _route_after_triage(state: PatientState) -> Literal["alert_node", "exam_node"]:
    if state.get("urgency_level") == "high":
        return "alert_node"
    return "exam_node"


# ── Builder do grafo ──────────────────────────────────────────────────────────

def build_graph(rag_chain):
    """Constrói e compila o StateGraph. Retorna um app LangGraph compilado."""
    graph = StateGraph(PatientState)

    # Registra nós.
    graph.add_node("triage_node", _triage_node)
    graph.add_node("alert_node", _alert_node)
    graph.add_node("exam_node", _exam_node)
    graph.add_node("diagnosis_node", partial(_diagnosis_node, rag_chain=rag_chain))
    graph.add_node("pharmacy_node", partial(_pharmacy_node, rag_chain=rag_chain))

    # Ponto de entrada.
    graph.set_entry_point("triage_node")

    # Arestas.
    graph.add_conditional_edges(
        "triage_node",
        _route_after_triage,
        {"alert_node": "alert_node", "exam_node": "exam_node"},
    )
    graph.add_edge("alert_node", "exam_node")
    graph.add_edge("exam_node", "diagnosis_node")
    # Ponto de interrupção humana antes de pharmacy.
    graph.add_edge("diagnosis_node", "pharmacy_node")
    graph.add_edge("pharmacy_node", END)

    return graph.compile(interrupt_before=["pharmacy_node"], checkpointer=MemorySaver())


def build_initial_state(patient_id: int, symptoms: str) -> PatientState:
    """Constrói o PatientState inicial a partir do patient_id e descrição de sintomas."""
    patient_info = get_patient(patient_id) or {}
    active_medications = get_active_medications(patient_id)
    diagnoses = get_diagnoses(patient_id)

    return PatientState(
        patient_id=patient_id,
        patient_info=patient_info,
        symptoms=symptoms,
        urgency_level="",
        pending_exams=[],
        completed_exams=[],
        active_medications=active_medications,
        diagnoses=diagnoses,
        differential_diagnosis="",
        diagnosis_low_evidence=False,
        suggested_treatment="",
        alerts=[],
        human_approval_required=False,
        sources=[],
        agent_steps=[],
        final_response="",
    )
