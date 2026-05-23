"""
app.py
======
Streamlit interface for the Medical AI Assistant.

Run:
    streamlit run app.py

Tabs:
  1. Assistente Médico  — chat interface invoking the LangGraph flow
  2. Auditoria          — view recent audit log entries
"""

from __future__ import annotations

# ── SSL compatibility patch (must precede ALL other imports) ──────────────────
# uv Python's bundled OpenSSL fails to initialise ssl.SSLContext on Fedora when
# the system openssl.cnf contains FIPS/legacy directives the bundled build can't
# parse. Fix: disable the system config file and add a Python-level fallback so
# aiohttp (imported transitively via unsloth→datasets) doesn't crash at load time.
import os
import ssl as _ssl

os.environ["OPENSSL_CONF"] = "/dev/null"   # force-override system openssl.cnf
os.environ["OPENSSL_MODULES"] = ""         # prevent FIPS provider auto-load

_orig_create_default_context = _ssl.create_default_context

def _safe_create_default_context(*args, **kwargs):
    try:
        return _orig_create_default_context(*args, **kwargs)
    except _ssl.SSLError:
        # App only loads local models — no external TLS connections needed.
        return _ssl._create_unverified_context()

_ssl.create_default_context = _safe_create_default_context
# ─────────────────────────────────────────────────────────────────────────────

import time
from pathlib import Path

import streamlit as st

from assistant.patient_db import get_all_patients, get_patient, init_db
from assistant.rag_chain import build_rag_chain
from assistant.vector_store import build_vector_store
from langgraph_flow.graph import build_graph, build_initial_state
from security.audit_logger import load_recent_logs, log_query
from security.explainability import attach_sources, build_explainability_footer
from security.guardrails import InputValidationError, sanitise_for_display, validate_input

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente Médico IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise DB & vector store on first run ─────────────────────────────────
@st.cache_resource
def _init_resources():
    init_db()
    build_vector_store()
    return True


@st.cache_resource
def _load_rag_chain():
    chain, memory = build_rag_chain(use_adapter=True)
    return chain, memory


@st.cache_resource
def _load_graph(_rag_chain):
    return build_graph(_rag_chain)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[int, dict]:
    st.sidebar.title("🏥 Assistente Médico IA")
    st.sidebar.caption("Suporte clínico baseado em protocolos hospitalares")
    st.sidebar.divider()

    patients = get_all_patients()
    patient_options = {f"[{p['id']}] {p['name']} ({p['age']}a, {p['sex']})": p["id"] for p in patients}
    selected_label = st.sidebar.selectbox("Selecionar Paciente", list(patient_options.keys()))
    patient_id = patient_options[selected_label]
    patient_info = get_patient(patient_id) or {}

    st.sidebar.divider()
    st.sidebar.subheader("Dados do Paciente")
    st.sidebar.write(f"**Nome:** {patient_info.get('name', '—')}")
    st.sidebar.write(f"**Idade:** {patient_info.get('age', '—')} anos")
    st.sidebar.write(f"**Sexo:** {patient_info.get('sex', '—')}")
    st.sidebar.write(f"**Tipo Sanguíneo:** {patient_info.get('blood_type', '—')}")
    st.sidebar.write(f"**Alergias:** {patient_info.get('allergies', '—')}")
    st.sidebar.write(f"**Condições:** {patient_info.get('conditions', '—')}")

    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ Este assistente é de suporte clínico. "
        "Todas as sugestões requerem validação médica."
    )

    return patient_id, patient_info


# ── Chat tab ──────────────────────────────────────────────────────────────────
def render_chat_tab(patient_id: int, patient_info: dict, graph, rag_chain) -> None:
    st.header("💬 Assistente Clínico")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "graph_state" not in st.session_state:
        st.session_state.graph_state = None

    # Display conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col1, col2 = st.columns([4, 1])
    with col1:
        query_type = st.radio(
            "Modo de consulta",
            ["Fluxo completo (LangGraph)", "Pergunta rápida (RAG direto)"],
            horizontal=True,
            label_visibility="collapsed",
        )

    user_input = st.chat_input("Descreva os sintomas ou faça uma pergunta clínica …")

    if not user_input:
        return

    # Validate input
    try:
        safe_input = validate_input(user_input)
    except InputValidationError as e:
        st.error(str(e))
        return

    with st.chat_message("user"):
        st.markdown(safe_input)
    st.session_state.chat_history.append({"role": "user", "content": safe_input})

    t_start = time.monotonic()

    with st.chat_message("assistant"):
        if query_type == "Fluxo completo (LangGraph)":
            _run_langgraph_flow(safe_input, patient_id, patient_info, graph, t_start)
        else:
            _run_rag_direct(safe_input, patient_id, patient_info, rag_chain, t_start)


def _run_langgraph_flow(
    symptoms: str, patient_id: int, patient_info: dict, graph, t_start: float
) -> None:
    import uuid
    if "langgraph_thread_id" not in st.session_state:
        st.session_state.langgraph_thread_id = str(uuid.uuid4())
    thread_id = st.session_state.langgraph_thread_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = build_initial_state(patient_id, symptoms)
    final_state = None
    all_output = ""

    with st.status("Executando fluxo multi-agente …", expanded=True) as status:
        # Run graph up to interrupt point (before pharmacy)
        for event in graph.stream(initial_state, config, stream_mode="values"):
            final_state = event

            agent_steps = final_state.get("agent_steps", [])
            if agent_steps:
                latest_step = agent_steps[-1]
                st.write(sanitise_for_display(latest_step))

            # Display alerts immediately
            for alert in final_state.get("alerts", []):
                st.error(sanitise_for_display(alert))

        status.update(label="Aguardando aprovação médica …", state="running")

        # Check if interrupted at pharmacy node
        if final_state and final_state.get("human_approval_required") is False:
            # Resume pharmacy node (None input = resume from checkpoint)
            for event in graph.stream(None, config, stream_mode="values"):
                final_state = event
                agent_steps = final_state.get("agent_steps", [])
                if agent_steps:
                    st.write(sanitise_for_display(agent_steps[-1]))

        # Rotate thread_id so next invocation starts fresh
        st.session_state.langgraph_thread_id = str(uuid.uuid4())
        status.update(label="Fluxo concluído.", state="complete")

    if not final_state:
        st.warning("Nenhum resultado gerado.")
        return

    diagnosis = sanitise_for_display(final_state.get("differential_diagnosis", ""))
    treatment = sanitise_for_display(final_state.get("suggested_treatment", ""))
    sources = final_state.get("sources", [])
    agent_steps = final_state.get("agent_steps", [])
    urgency = final_state.get("urgency_level", "")
    human_approval = final_state.get("human_approval_required", False)

    # Build full response
    if diagnosis:
        st.subheader("🩺 Diagnóstico Diferencial")
        st.markdown(attach_sources(diagnosis, sources))

    if treatment:
        st.subheader("💊 Sugestões Farmacológicas")
        st.markdown(treatment)

    footer = build_explainability_footer(agent_steps, urgency, human_approval)
    st.markdown(footer)

    all_output = "\n\n".join(filter(None, [diagnosis, treatment]))
    latency_ms = (time.monotonic() - t_start) * 1000

    log_query(
        patient_id=patient_id,
        query=symptoms,
        response=all_output,
        sources=sources,
        agent_steps=agent_steps,
        latency_ms=latency_ms,
        urgency_level=urgency,
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": all_output[:1000] + " …"}
    )


def _run_rag_direct(
    question: str, patient_id: int, patient_info: dict, rag_chain, t_start: float
) -> None:
    from assistant.rag_chain import ask

    patient_ctx = (
        f"Paciente: {patient_info.get('name')}, {patient_info.get('age')}a, "
        f"condições: {patient_info.get('conditions', 'nenhuma')}, "
        f"alergias: {patient_info.get('allergies', 'nenhuma conhecida')}."
    )

    with st.spinner("Consultando base de protocolos …"):
        result = ask(rag_chain, question, patient_context=patient_ctx)

    answer = sanitise_for_display(result["answer"])
    sources = result["sources"]
    latency_ms = (time.monotonic() - t_start) * 1000

    full_response = attach_sources(answer, sources)
    st.markdown(full_response)

    log_query(
        patient_id=patient_id,
        query=question,
        response=answer,
        sources=sources,
        agent_steps=["Consulta RAG direta"],
        latency_ms=latency_ms,
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response[:1000] + " …"}
    )


# ── Audit tab ─────────────────────────────────────────────────────────────────
def render_audit_tab() -> None:
    st.header("📋 Log de Auditoria")
    n_entries = st.slider("Número de entradas recentes", min_value=5, max_value=100, value=20)
    logs = load_recent_logs(n=n_entries)

    if not logs:
        st.info("Nenhuma entrada no log ainda.")
        return

    st.caption(f"Exibindo {len(logs)} entradas mais recentes (ordem inversa).")
    for entry in logs:
        with st.expander(
            f"[{entry.get('timestamp', '?')}] {entry.get('event', '?')} — Paciente {entry.get('patient_id', '?')}",
            expanded=False,
        ):
            st.json(entry)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_resources()
    rag_chain, _ = _load_rag_chain()
    graph = _load_graph(rag_chain)

    patient_id, patient_info = render_sidebar()

    tab_chat, tab_audit = st.tabs(["💬 Assistente", "📋 Auditoria"])

    with tab_chat:
        render_chat_tab(patient_id, patient_info, graph, rag_chain)

    with tab_audit:
        render_audit_tab()


if __name__ == "__main__":
    main()
