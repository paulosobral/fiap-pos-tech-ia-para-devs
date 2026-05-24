"""
app.py
======
Interface Streamlit para o Assistente de IA Médica.

Execução:
        streamlit run app.py

Abas:
    1. Assistente Médico  — interface de chat que aciona o fluxo LangGraph
    2. Auditoria          — visualiza entradas recentes do log de auditoria
    3. Glossário & Testes — glossário de abreviações médicas e casos de teste com respostas de referência
"""

from __future__ import annotations

# ── Patch de compatibilidade SSL (deve vir antes de TODOS os imports) ─────────
# O OpenSSL embarcado no Python do uv pode falhar ao inicializar ssl.SSLContext
# no Fedora quando o openssl.cnf do sistema traz diretivas FIPS/legacy que esse
# build não consegue interpretar. Solução: desativar o arquivo de configuração
# do sistema e aplicar um fallback em Python para evitar falha de importação no aiohttp
# (importado de forma transitiva via unsloth→datasets).
import os
import ssl as _ssl

os.environ["OPENSSL_CONF"] = "/dev/null"   # força sobrescrever o openssl.cnf do sistema
os.environ["OPENSSL_MODULES"] = ""         # evita autoload do provider FIPS

_orig_create_default_context = _ssl.create_default_context

def _safe_create_default_context(*args, **kwargs):
    try:
        return _orig_create_default_context(*args, **kwargs)
    except _ssl.SSLError:
        # O app carrega apenas modelos locais; não depende de TLS externo.
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

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente Médico IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inicializa DB e vector store na primeira execução ─────────────────────────
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


# ── Barra lateral ──────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[int, dict]:
    st.sidebar.title("🏥 Assistente Médico IA")
    st.sidebar.caption("Suporte clínico baseado em protocolos hospitalares")
    st.sidebar.divider()

    patients = get_all_patients()
    # Primeira entrada = sentinela "sem paciente" (id=0) para consultas gerais.
    patient_options: dict[str, int] = {"[—] Consulta Geral (sem paciente)": 0}
    patient_options.update({f"[{p['id']}] {p['name']} ({p['age']}a, {p['sex']})": p["id"] for p in patients})
    selected_label = st.sidebar.selectbox("Selecionar Paciente", list(patient_options.keys()))
    patient_id = patient_options[selected_label]
    patient_info = get_patient(patient_id) or {}

    st.sidebar.divider()
    st.sidebar.subheader("Dados do Paciente")
    if patient_id == 0:
        st.sidebar.info("Nenhum paciente selecionado.\nAs respostas serão baseadas apenas nos protocolos e no modelo.")
    else:
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


# ── Aba de chat ────────────────────────────────────────────────────────────────
def render_chat_tab(patient_id: int, patient_info: dict, graph, rag_chain) -> None:
    st.header("💬 Assistente Clínico")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "graph_state" not in st.session_state:
        st.session_state.graph_state = None

    if "last_patient_id" not in st.session_state:
        st.session_state.last_patient_id = patient_id

    if st.session_state.last_patient_id != patient_id:
        st.session_state.chat_history = []
        if hasattr(rag_chain, "clear_history"):
            rag_chain.clear_history()
        st.session_state.last_patient_id = patient_id

    # Exibe o histórico da conversa.
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

    # Valida o input.
    try:
        safe_input = validate_input(user_input)
    except InputValidationError as e:
        st.error(str(e))
        return

    with st.chat_message("user"):
        st.markdown(safe_input)
    st.session_state.chat_history.append({"role": "user", "content": safe_input})

    t_start = time.monotonic()

    # Evita contaminação entre consultas no fluxo clínico.
    if hasattr(rag_chain, "clear_history"):
        rag_chain.clear_history()

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
        # Executa o grafo até o ponto de interrupção (antes de pharmacy).
        for event in graph.stream(initial_state, config, stream_mode="values"):
            final_state = event

            agent_steps = final_state.get("agent_steps", [])
            if agent_steps:
                latest_step = agent_steps[-1]
                st.write(sanitise_for_display(latest_step))

            # Exibe alertas imediatamente.
            for alert in final_state.get("alerts", []):
                st.error(sanitise_for_display(alert))

        status.update(label="Aguardando aprovação médica …", state="running")

        # Verifica se houve interrupção no nó pharmacy.
        if final_state and final_state.get("human_approval_required") is False:
            # Retoma o nó pharmacy (input None = retoma do checkpoint).
            for event in graph.stream(None, config, stream_mode="values"):
                final_state = event
                agent_steps = final_state.get("agent_steps", [])
                if agent_steps:
                    st.write(sanitise_for_display(agent_steps[-1]))

        # Gera novo thread_id para a próxima execução começar limpa.
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

    # Monta a resposta completa.
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


# ── Aba de auditoria ──────────────────────────────────────────────────────────
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


# ── Aba de referência ─────────────────────────────────────────────────────────
_GLOSSARY = [
    ("HAS",            "Hipertensão Arterial Sistêmica",             "Pressão alta crônica."),
    ("DM2",            "Diabetes Mellitus tipo 2",                   "Açúcar alto no sangue por resistência à insulina."),
    ("ICC",            "Insuficiência Cardíaca Congestiva",          "Coração não bombeia sangue em quantidade suficiente."),
    ("AVC",            "Acidente Vascular Cerebral",                 "\"Derrame\" — interrupção do fluxo sanguíneo cerebral."),
    ("PAC",            "Pneumonia Adquirida na Comunidade",          "Pneumonia contraída fora do ambiente hospitalar."),
    ("IAM",            "Infarto Agudo do Miocárdio",                 "Ataque cardíaco — obstrução de artéria coronária."),
    ("DPOC",           "Doença Pulmonar Obstrutiva Crônica",         "Enfisema / bronquite crônica: dificuldade respiratória permanente."),
    ("SEPSE",          "—",                                         "Resposta inflamatória sistêmica grave a uma infecção."),
    ("HbA1c",          "Hemoglobina Glicada",                        "Média do açúcar no sangue dos últimos ~3 meses."),
    ("BNP",            "Peptídeo Natriurético Cerebral",             "Marcador laboratorial de sobrecarga e estresse cardíaco."),
    ("SpO₂",           "Saturação periférica de oxigênio",           "Percentual de O₂ no sangue medido pelo oxímetro de pulso."),
    ("PA",             "Pressão Arterial",                           "Força do sangue nas artérias. Ex.: 120/80 mmHg."),
    ("FC",             "Frequência Cardíaca",                        "Número de batimentos cardíacos por minuto."),
    ("FR",             "Frequência Respiratória",                    "Número de respirações por minuto."),
    ("ECG",            "Eletrocardiograma",                          "Exame que registra a atividade elétrica do coração."),
    ("INR",            "International Normalized Ratio",             "Medida da coagulação; importante em uso de anticoagulantes."),
    ("TFG",            "Taxa de Filtração Glomerular",               "Medida da função renal em mL/min."),
    ("NYHA",           "New York Heart Association",                 "Classificação funcional da ICC de I (leve) a IV (grave)."),
    ("IECA",           "Inib. da Enzima Conversora de Angiotensina", "Classe de anti-hipertensivos (ex.: enalapril, captopril)."),
    ("BRA",            "Bloqueador do Receptor de Angiotensina",     "Anti-hipertensivo alternativo ao IECA (ex.: losartana)."),
    ("BCC",            "Bloqueador dos Canais de Cálcio",            "Anti-hipertensivo vasodilatador (ex.: anlodipino)."),
    ("CHA₂DS₂-VASc",  "Score de risco tromboembólico",              "Pontuação que estima risco de AVC em fibrilação atrial."),
    ("DPN",            "Dispneia Paroxística Noturna",               "Falta de ar que acorda o paciente à noite; critério maior de ICC."),
    ("MMII",           "Membros Inferiores",                         "Pernas. \"Edema de MMII\" = inchaço nas pernas."),
    ("ATB",            "Antibiótico",                               "Medicamento usado para tratar infecções bacterianas."),
]

_TEST_CASES = [
    {
        "label": "🫀 HAS — Urgência Hipertensiva",
        "patient_profile": (
            "Paciente 58a, masculino, HAS há 10 anos em uso de losartana 50mg + anlodipino 5mg. "
            "Chegou ao pronto-atendimento com cefaleia intensa na nuca e PA 190/110 mmHg. "
            "Sem déficits neurológicos, sem dor torácica, sem alteração visual."
        ),
        "suggested_query": (
            "Paciente com HAS, cefaleia intensa occipital, PA 190/110 mmHg, sem sinais de lesão "
            "aguda de órgão-alvo. Qual a conduta para urgência hipertensiva?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Distinção urgência vs. emergência hipertensiva**
- **Urgência:** PA elevada sem lesão aguda de órgão-alvo (LOA). Redução gradual em 24–48h, manejo ambulatorial.
- **Emergência:** PA elevada **com** LOA (AVC, IAM, dissecção aórtica, edema agudo de pulmão). Exige internação e medicação IV.

**Conduta na urgência (este caso)**
1. Repouso, ambiente calmo, confirmar aferição com manguito adequado.
2. **Captopril 25 mg VO** (não sublingual — absorção oral é segura e eficaz) ou **clonidina 0,1 mg VO** repetível em 1h.
3. Meta: reduzir PA em ~25% nas primeiras 2–6h; não normalizar rapidamente (risco de isquemia cerebral).
4. **Evitar:** nifedipina SL (quedas bruscas de PA) e nitroprussiato sem LOA.
5. Investigar causa desencadeante: adesão ao tratamento, ingesta de sal, AINE, descontinuação de clonidina.
6. Reavaliação em 24–48h; considerar ajuste do esquema anti-hipertensivo crônico.

**Exames complementares úteis:** ECG, creatinina, EAS, fundo de olho se disponível.
        """,
    },
    {
        "label": "🍬 DM2 — Descompensação Glicêmica",
        "patient_profile": (
            "Paciente 52a, feminina, DM2 há 8 anos em uso de metformina 850mg 12/12h. "
            "Glicemia capilar 320 mg/dL, HbA1c 11%, sem náuseas, sem vômitos, sem hálito cetônico. "
            "Refere ter suspendido medicação por 10 dias por dificuldade financeira."
        ),
        "suggested_query": (
            "Paciente DM2, glicemia 320 mg/dL, HbA1c 11%, sem sinais de cetoacidose. "
            "Como manejar a descompensação glicêmica e retomar o controle?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Avaliação inicial**
- Descartar cetoacidose (glicemia >250 + pH <7,3 + cetonemia) e ESHH (hiperosmolar).
- Investigar gatilho: não-adesão (este caso), infecção, dieta, outros medicamentos (corticoides, diuréticos tiazídicos).

**Manejo imediato**
1. Retomar metformina se TFG ≥ 45 mL/min (contraindicada em TFG <30).
2. Hidratação oral adequada.
3. Se glicemia >300 mg/dL persistente: adicionar insulina NPH basal 0,1–0,2 UI/kg/noite como ponte até re-compensação.
4. Orientar monitorização domiciliar 2–4x/dia.

**Metas glicêmicas (ADA 2024)**
- Jejum: 80–130 mg/dL | Pós-prandial 2h: < 180 mg/dL | HbA1c < 7% (individualizar em idosos → < 8%).

**Próximos passos**
- Avaliar intensificação do esquema: adicionar iSGLT2 (dapagliflozina) ou arGLP-1 (semaglutida) se sem contraindicação, considerando benefício cardiovascular e renal.
- Solicitar: TFG, perfil lipídico, microalbuminúria, fundo de olho (rastreio de complicações).
- Reforço de educação em diabetes e acesso a medicação (programa Farmácia Popular).
        """,
    },
    {
        "label": "🫁 ICC — Critérios de Framingham e Estadiamento NYHA",
        "patient_profile": (
            "Paciente 68a, masculino, HAS + DM2 + ICC diagnosticada há 3 anos. "
            "Relata dispneia ao deitar (ortopneia 2 travesseiros), edema de MMII ++/4, "
            "fadiga intensa ao caminhar 50m. BNP 890 pg/mL. Faz uso de carvedilol + enalapril + furosemida."
        ),
        "suggested_query": (
            "Paciente com ICC, ortopneia, edema de MMII ++ e dispneia com atividade leve. "
            "Quais os critérios de Framingham para diagnóstico de ICC e como classificar pela NYHA?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Critérios de Framingham para ICC**
Diagnóstico requer ≥ 2 critérios **maiores** OU 1 maior + 2 **menores**.

| Maiores | Menores |
|---|---|
| Dispneia paroxística noturna (DPN) | Edema de MMII bilateral |
| Distensão venosa jugular | Dispneia aos esforços |
| Estertores pulmonares | Hepatomegalia |
| Cardiomegalia na Rx | Derrame pleural |
| Edema agudo de pulmão | Capacidade vital reduzida 1/3 |
| Galope B3 | FC > 120 bpm |
| Refluxo hepatojugular | |
| Perda > 4,5 kg com tratamento | |

*Este paciente preenche ≥ 2 maiores → diagnóstico confirmado.*

**Classificação NYHA**
- **I:** Sem limitação — atividades físicas habituais sem sintomas.
- **II:** Leve limitação — sintomas em atividades moderadas.
- **III:** Limitação importante — sintomas em atividades leves (< 50m). ← **Este paciente**
- **IV:** Sintomas em repouso ou mínimo esforço.

**Manejo na descompensação**
Avaliar a tríade: congestão (diurético), débito (betabloqueador/IECA) e ritmo (ECG). 
Considerar reajuste de furosemida IV se refratário a VO. BNP > 400 pg/mL = marcador de risco elevado.
        """,
    },
    {
        "label": "🦠 Sepse — Diagnóstico e Bundle",
        "patient_profile": (
            "Paciente 45a, masculino, internado há 2 dias por pneumonia. "
            "Evolui com febre 39,2°C, FC 118 bpm, FR 24 rpm, PA 98/62 mmHg, "
            "confusão mental (novo), leucocitose 18.000 com desvio à esquerda, lactato 2,8 mmol/L."
        ),
        "suggested_query": (
            "Febre 39°C, FC 118, FR 24, PA 98/62, confusão mental nova, lactato 2,8. "
            "Quais os critérios de sepse grave e qual o bundle de atendimento?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Definições (Sepsis-3, 2016)**
- **Sepse:** disfunção orgânica ameaçadora à vida causada por resposta desregulada à infecção.
  - Critério prático: suspeita de infecção + **SOFA ≥ 2** (ou qSOFA ≥ 2 no extra-hospitalar).
- **Choque séptico:** sepse + hipotensão (PAM < 65) refratária a fluidos + lactato > 2 mmol/L.

*Este paciente: infecção documentada + confusão (SNC) + hipotensão + lactato 2,8 → **choque séptico**.*

**Bundle Hora-1 (Surviving Sepsis Campaign)**
1. ✅ Coletar **hemoculturas** (2 pares) **antes** do antibiótico.
2. ✅ Iniciar **antibiótico de largo espectro ≤ 1h** (ex.: piperacilina-tazobactam IV).
3. ✅ Dosar **lactato** sérico (já 2,8 — repetir em 2h para avaliar clareamento).
4. ✅ **Cristaloide 30 mL/kg IV** em 3h se hipotensão ou lactato > 4.
5. ✅ Iniciar **vasopressor** (noradrenalina) se PAM < 65 após ressuscitação → meta PAM ≥ 65.
6. ✅ **Oxigênio** suplementar — meta SpO₂ ≥ 94%.

**Monitoramento:** débito urinário (> 0,5 mL/kg/h), lactato seriado, PCR, procalcitonina.
        """,
    },
    {
        "label": "🧠 AVC — Trombólise com Alteplase",
        "patient_profile": (
            "Paciente 61a, feminina, hipertensa. Apresentou hemiplegia direita súbita e afasia "
            "motora há 2 horas. PA 162/94 mmHg. TC de crânio: sem hemorragia. "
            "Sem anticoagulação, sem cirurgia recente, plaquetas 210k, glicemia 112 mg/dL."
        ),
        "suggested_query": (
            "Paciente com AVC isquêmico, hemiplegia direita, afasia, início há 2h, "
            "TC sem hemorragia, PA 162/94. Critérios para trombólise com alteplase e como conduzir?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Janela terapêutica:** até **4,5 horas** do início dos sintomas (ou último bem conhecido).

**Critérios de inclusão:**
- AVC isquêmico com déficit neurológico mensurável (NIHSS ≥ 4 tipicamente).
- TC de crânio sem hemorragia ← ✅
- Idade ≥ 18 anos ← ✅
- Início dos sintomas < 4,5h ← ✅ (2h)

**Principais exclusões:**
- PA > 185/110 mmHg não controlada → **neste caso PA 162/94: elegível sem necessidade de controle prévio**.
- Glicemia < 50 ou > 400 mg/dL → ✅ (112)
- Anticoagulante com INR > 1,7 ou DOAC nas últimas 48h → ✅ (sem uso)
- Cirurgia de grande porte < 14 dias, trauma craniano < 3 meses → ✅
- Plaquetas < 100.000 → ✅ (210k)

**Dose rt-PA (alteplase):**
> 0,9 mg/kg IV (máximo 90 mg)
> 10% em bolus 1 min → 90% restantes em infusão de 60 min

**Cuidados pós-trombólise:**
- Monitorar PA a cada 15 min na 1ª hora (meta < 180/105).
- Sem anticoagulação ou antiplaquetário por 24h.
- TC de controle em 24h.
- Unidade de AVC ou UTI neurológica.
        """,
    },
    {
        "label": "🫁 PAC — Antibioticoterapia Ambulatorial",
        "patient_profile": (
            "Paciente 38a, masculino, sem comorbidades, não fumante. "
            "Tosse produtiva há 5 dias, febre 38,5°C, dor pleurítica. "
            "Rx tórax: infiltrado lobar em base direita. CURB-65 = 0 (ambulatorial)."
        ),
        "suggested_query": (
            "Paciente com PAC leve (CURB-65 = 0), sem comorbidades, ambulatorial. "
            "Qual antibiótico de primeira linha e por quanto tempo?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Estratificação de gravidade (CURB-65)**
| Critério | Pontos |
|---|---|
| Confusão | 1 |
| Ureia > 50 mg/dL | 1 |
| FR ≥ 30 rpm | 1 |
| PA sistólica < 90 ou diastólica ≤ 60 | 1 |
| Idade ≥ 65 anos | 1 |

Escore 0–1 → tratamento ambulatorial | 2 → internar | ≥ 3 → UTI a considerar.

**Antibioticoterapia — PAC ambulatorial sem comorbidades:**

| Agente coberto | Opção preferencial | Duração |
|---|---|---|
| Típico (*S. pneumoniae*) | Amoxicilina 1g VO 8/8h | 5–7 dias |
| Atípico (*Mycoplasma*, *Chlamydophila*) | Azitromicina 500mg/dia | 5 dias |
| Empírico (incerteza) | Amoxicilina + Azitromicina | 5–7 dias |

**Com comorbidades** (DM, DRC, ICC, imunossupressão):
→ Amoxicilina-clavulanato 875/125mg 12/12h **+** azitromicina.

**Hospitalizado:**
→ Ceftriaxona 1g IV/dia + azitromicina OU levofloxacino 750mg/dia (se alergia a beta-lactâmico).

**Critérios de falha** (reavaliar em 48–72h): piora clínica, sem melhora de febre → ampliar cobertura ou internar.
        """,
    },
]


def render_reference_tab() -> None:
    st.header("📚 Glossário & Casos de Teste")
    st.caption(
        "Referência rápida para quem não é da área da saúde e casos clínicos prontos "
        "para testar o modelo, com respostas de referência geradas por **Claude Sonnet 4.6**."
    )

    # ── Glossário ─────────────────────────────────────────────────────────────
    st.subheader("📖 Glossário de Siglas Médicas")
    st.info(
        "As siglas abaixo aparecem nos prontuários dos pacientes sintéticos e nas respostas do modelo. "
        "Use esta tabela como referência durante os testes."
    )

    import pandas as pd
    df = pd.DataFrame(
        _GLOSSARY,
        columns=["Sigla", "Nome completo", "O que significa (para leigos)"],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Casos de teste ────────────────────────────────────────────────────────
    st.subheader("🧪 Casos de Teste Sugeridos")
    st.info(
        "Para cada caso abaixo você encontra: o perfil do paciente a selecionar (ou usar consulta geral), "
        "a pergunta sugerida para colar no chat e a resposta esperada segundo **Claude Sonnet 4.6**. "
        "Compare com o que o modelo fine-tunado responde para avaliar qualidade e alucinações."
    )

    for case in _TEST_CASES:
        with st.expander(case["label"], expanded=False):
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Perfil do paciente**")
                st.markdown(case["patient_profile"])
                st.markdown("---")
                st.markdown("**Pergunta sugerida — copie e cole no chat:**")
                st.code(case["suggested_query"], language=None)

            with col_r:
                st.markdown(case["reference_answer"])


# ── Principal ─────────────────────────────────────────────────────────────────
def main() -> None:
    _init_resources()
    rag_chain, _ = _load_rag_chain()
    graph = _load_graph(rag_chain)

    patient_id, patient_info = render_sidebar()

    tab_chat, tab_audit, tab_ref = st.tabs(["💬 Assistente", "📋 Auditoria", "📚 Glossário & Testes"])

    with tab_chat:
        render_chat_tab(patient_id, patient_info, graph, rag_chain)

    with tab_audit:
        render_audit_tab()

    with tab_ref:
        render_reference_tab()


if __name__ == "__main__":
    main()
