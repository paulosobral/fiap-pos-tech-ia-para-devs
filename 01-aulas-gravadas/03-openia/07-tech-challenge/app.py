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
    ("TSH",            "Hormônio Estimulante da Tireoide",           "Controla a produção de T4/T3; elevado no hipotireoidismo."),
    ("T4L",            "Tiroxina Livre",                             "Hormônio tireoidiano ativo; baixo no hipotireoidismo primário."),
    ("LT4",            "Levotiroxina",                               "Hormônio tireoidiano sintético; tratamento do hipotireoidismo."),
    ("Hipotireoidismo", "—",                                         "Glândula tireoide produz hormônios em quantidade insuficiente."),
    # ── Condições adicionais dos pacientes ────────────────────────────────
    ("FA",              "Fibrilação Atrial",                          "Arritmia em que o átrio bate de forma irregular; aumenta risco de AVC."),
    ("DRC",             "Doença Renal Crônica",                       "Perda progressiva da função dos rins; estadiada de 1 (leve) a 5 (diálise)."),
    ("DM1",             "Diabetes Mellitus tipo 1",                   "Diabetes autoimune; pâncreas não produz insulina; requer insulina."),
    ("DLP",             "Dislipidemia",                               "Alteração no colesterol ou triglicérides; fator de risco cardiovascular."),
    ("LES",             "Lúpus Eritematoso Sistêmico",                "Doença autoimune que pode afetar pele, articulações, rins e outros órgãos."),
    ("HFrEF",           "IC com Fração de Ejeção Reduzida",           "IC com FE < 40%; disfunção sistólica do ventrículo esquerdo."),
    ("HFpEF",           "IC com Fração de Ejeção Preservada",         "IC com FE ≥ 50%; disfunção diastólica."),
    ("AAS",             "Ácido Acetilsalicílico",                     "Aspirina; antiagregante plaquetário usado em doenças cardiovasculares."),
    ("TARV",            "Terapia Antirretroviral",                    "Tratamento do HIV com combinação de antivirais; permite vida praticamente normal."),
    ("HIV",             "Vírus da Imunodeficiência Humana",           "Vírus que compromete o sistema imune; controlado cronicamente com TARV."),
    ("Asma",            "—",                                         "Doença inflamatória das vias aéreas com broncoespasmo reversível."),
    ("Rinite alérgica", "—",                                         "Inflamação nasal alérgica; frequentemente associada à asma."),
    ("Parkinson",       "—",                                         "Doença neurodegenerativa: tremor, rigidez e lentidão de movimentos."),
    ("Alzheimer",       "—",                                         "Demência progressiva mais comum; perda de memória e função cognitiva."),
    ("Epilepsia",       "—",                                         "Distúrbio neurológico com crises convulsivas recorrentes."),
    ("Enxaqueca",       "—",                                         "Cefaleia pulsátil intensa, frequentemente com náusea e fotofobia."),
    ("Depressão",       "—",                                         "Transtorno de humor com tristeza persistente, anedonia e prejuízo funcional."),
    ("Ansiedade",       "—",                                         "Transtorno de ansiedade generalizada; preocupação excessiva e sintomas físicos."),
    ("Gota",            "—",                                         "Artrite por depósito de cristais de ácido úrico; dor intensa em articulações."),
    ("Obesidade",       "—",                                         "IMC ≥ 30 kg/m²; fator de risco para DM2, HAS, DLP e doenças cardiovasculares."),
    ("Tabagismo",       "—",                                         "Dependência de tabaco; principal fator de risco evitável de doenças crônicas."),
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
    {
        "label": "🦋 Hipotireoidismo — Diagnóstico e Levotiroxina",
        "patient_profile": (
            "Paciente 42a, feminina, sem comorbidades prévias. Refere cansaço progressivo há 4 meses, "
            "ganho de 6 kg sem mudança de dieta, constipação, pele seca, intolerância ao frio e "
            "raciocínio mais lento. TSH 18,2 mUI/L (ref. 0,4–4,0), T4L 0,6 ng/dL (ref. 0,8–1,8)."
        ),
        "suggested_query": (
            "Paciente com TSH 18,2 e T4L 0,6, fadiga, ganho de peso, constipação, intolerância ao frio. "
            "Como confirmar o diagnóstico de hipotireoidismo primário e como iniciar levotiroxina?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Diagnóstico**
- **Hipotireoidismo primário confirmado:** TSH elevado (18,2) + T4L baixo (0,6) — padrão clássico.
- Classificação: **manifesto** (sintomático + T4L abaixo da referência).
- Subclínico seria: TSH elevado + T4L normal + pouco ou nenhum sintoma.

**Investigar causa (tireoidite de Hashimoto é a mais comum)**
- Solicitar **anti-TPO** (anticorpo antitireoperoxidase) — positivo em ~95% dos Hashimoto.
- USG de tireoide: útil se palpar nódulo ou bócio.

**Tratamento — Levotiroxina (LT4)**
| Perfil | Dose inicial |
|---|---|
| Adulto jovem saudável | 1,6 µg/kg/dia (dose plena) |
| Idoso ou cardiopata | 12,5–25 µg/dia, com aumento gradual a cada 4–6 semanas |
| Este caso (42a, sem comorbidades) | ~75–100 µg/dia em dose única matinal |

**Regras de administração:**
- Tomar em **jejum** (30–60 min antes do café ou 4h após cálcio/ferro/antácido).
- Dose única matinal; consistência é mais importante que horário exato.

**Monitoramento:**
- Repetir TSH + T4L em **6–8 semanas** após início ou ajuste.
- Meta: TSH 0,5–2,5 mUI/L (individualizar em idosos e grávidas).
- Sintomas melhoram em 2–4 semanas; normalização laboratorial em 6–12 semanas.

**Gestação:** necessidade de LT4 aumenta ~30% — ajustar imediatamente.
        """,
    },
    {
        "label": "💨 Asma — Crise Aguda e Escalonamento de Tratamento",
        "patient_profile": (
            "Paciente 34a, feminina, asma desde a infância. Refere crises noturnas 2–3x/semana, "
            "uso diário de salbutamol (β2 de curta ação). Acordou às 3h com dispneia intensa, "
            "sibilos difusos, SpO₂ 93%, FC 110 bpm, FR 28 rpm. Não usou CI nos últimos 15 dias."
        ),
        "suggested_query": (
            "Paciente asmática com crise noturna grave, SpO₂ 93%, sibilos, FR 28, uso excessivo de "
            "salbutamol e sem corticoide inalatório. Como manejar a crise e escalonar o tratamento?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Classificação da crise (GINA 2024)**
| Grau | SpO₂ | Fala | Musculatura acessória |
|---|---|---|---|
| Leve/Moderada | ≥ 95% | Frases completas | Não |
| Grave | 91–94% | Palavras | Sim |
| Risco de vida | < 91% | Impossível | Paradoxal |

*Este caso: grave (SpO₂ 93%, FR 28, sibilos difusos).*

**Manejo imediato da crise grave**
1. **O₂** suplementar → meta SpO₂ ≥ 94–95%.
2. **Salbutamol 2,5 mg** nebulizado (ou 4–8 jatos em espaçador) a cada 20 min × 3 na 1ª hora.
3. **Ipratrópio 0,5 mg** nebulizado nas primeiras 3 doses (broncoespasmo grave).
4. **Prednisolona 40–50 mg VO** (ou metilprednisolona 60–125 mg IV) — início em até 1h.
5. Reavaliar após 1h: se SpO₂ < 92% ou sem melhora → hospitalizar; se ≥ 95% e melhora clínica → alta com corticoide VO 5–7 dias.

**Escalonamento do tratamento de manutenção (GINA)**
| Degrau | Tratamento |
|---|---|
| 1 | Formoterol+beclometasona em baixa dose conforme necessário |
| 2 | CI baixa dose diário + SABA de resgate |
| 3 | CI baixa dose + LABA (ex.: salmeterol) |
| 4 | CI média/alta dose + LABA |
| 5 | Adicionar anti-IgE / anti-IL5 / corticoide oral |

*Uso ≥ 3×/semana de SABA = asma não controlada → subir degrau.*

**Educação:** técnica inalatória, plano de ação escrito, evitar gatilhos (ácaros, fumaça, AINE em sensíveis).
        """,
    },
    {
        "label": "❤️ FA — Anticoagulação e Controle de Frequência",
        "patient_profile": (
            "Paciente 75a, masculino, FA persistente diagnosticada há 2 anos, HAS, DPOC. "
            "Em uso de warfarina (INR atual 2,8), carvedilol 12,5 mg 2x/dia. "
            "INR no limite superior; questiona se pode trocar para DOAC e "
            "pergunta sobre a meta de frequência cardíaca."
        ),
        "suggested_query": (
            "Paciente com FA, 75 anos, HAS, DPOC, INR 2,8 em warfarina. "
            "Posso trocar para DOAC? Qual a meta de FC e como calcular o CHA₂DS₂-VASc?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Risco tromboembólico — CHA₂DS₂-VASc**
| Critério | Pontos |
|---|---|
| C — ICC / HFrEF | 1 |
| H — HAS | 1 ← ✅ |
| A₂ — Idade ≥ 75 anos | **2** ← ✅ |
| D — DM2 | 1 |
| S₂ — AVC/AIT prévio | **2** |
| V — Doença vascular | 1 |
| A — Idade 65–74 anos | 1 |
| Sc — Sexo feminino | 1 |

*Este paciente: HAS (1) + Idade ≥ 75 (2) = **3 pontos** → anticoagulação fortemente indicada.*

**Warfarina vs DOAC**
- **DOAC** (rivaroxabana, apixabana, dabigatrana) são preferidos em FA não valvar: dose fixa, sem monitoramento de INR, menos interações.
- **Trocar:** sim, é seguro. Estratégia: suspender warfarina quando INR < 2,0 e iniciar DOAC no mesmo dia.
- **Atenção DPOC:** sem contraindicação específica para DOAC. Verificar TFG antes (dabigatrana requer TFG ≥ 30; apixabana é mais segura em DRC).

**Meta de frequência cardíaca**
- FA com função ventricular preservada: **FC < 110 bpm em repouso** (lenient control, RACE II).
- FA sintomática ou com IC: meta mais estrita **< 80 bpm**.
- Carvedilol é adequado; alternativas: bisoprolol, diltiazem (se sem IC sistólica).

**Reversão de ritmo:** avaliar cardioversão elétrica se sintomático e < 48h de início; anticoagular por ≥ 3 semanas antes se > 48h ou tempo incerto.
        """,
    },
    {
        "label": "🫘 DRC — Estadiamento e Proteção Renal",
        "patient_profile": (
            "Paciente 61a, masculino, DRC estágio 3 (TFGe 42 mL/min/1,73m²), DM2 e HAS. "
            "Potássio sérico pendente. Faz uso de enalapril 10 mg + metformina 850 mg 2x/dia. "
            "Pergunta se pode continuar a metformina e como proteger os rins."
        ),
        "suggested_query": (
            "Paciente com DRC estágio 3, TFGe 42, DM2 e HAS em uso de enalapril e metformina. "
            "Posso manter a metformina? Como estadiar a DRC e quais são as medidas de neuroproteção renal?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Estadiamento DRC (KDIGO)**
| Estágio | TFGe (mL/min/1,73m²) | Descrição |
|---|---|---|
| 1 | ≥ 90 | Normal com marcador de lesão |
| 2 | 60–89 | Levemente reduzida |
| **3a** | **45–59** | **Leve a moderada** |
| **3b** | **30–44** | **Moderada a grave** ← *este paciente (42)* |
| 4 | 15–29 | Grave |
| 5 | < 15 | Falência renal (diálise) |

**Metformina na DRC**
- TFGe ≥ 45: **manter** com monitoramento.
- TFGe 30–44: **reduzir dose** e monitorar mais frequentemente. ← *este caso: cautela, dose reduzida*
- TFGe < 30: **contraindicada** (risco de acidose láctica).
- **Conclusão:** TFGe 42 → zona cinza; manter com dose reduzida (850 mg 1x/dia) e monitorar TFGe a cada 3 meses.

**Medidas de nefroproteção**
1. **Controle pressórico** — meta PA < 130/80 mmHg em DRC + DM; manter IECA/BRA (enalapril é adequado, reduz proteinúria).
2. **Controle glicêmico** — HbA1c < 7–8% (menos estrito em DRC avançada).
3. **Evitar nefrotóxicos** — AINE, contraste iodado sem hidratação prévia, aminoglicosídeos.
4. **iSGLT2** (dapagliflozina, empagliflozina) — indicados em DRC+DM com TFGe ≥ 25; reduzem progressão renal independentemente do efeito glicêmico.
5. **Dieta:** restrição de sódio (< 5 g/dia); ajustar proteína e potássio conforme estágio.
6. **Monitoramento:** TFGe + potássio + microalbuminúria a cada 3 meses.
        """,
    },
    {
        "label": "🩸 DLP — Dislipidemia e Risco Cardiovascular",
        "patient_profile": (
            "Paciente 60a, feminina, DM2, DLP e obesidade (IMC 33). "
            "LDL 148 mg/dL, HDL 38 mg/dL, TG 210 mg/dL. "
            "Alerta de alergia a estatinas (miopatia). Sem doença cardiovascular prévia. "
            "Pergunta qual o alvo de LDL e como tratar sem estatina."
        ),
        "suggested_query": (
            "Paciente DM2 + DLP + obesidade, LDL 148, HDL 38, TG 210, alergia a estatinas por miopatia. "
            "Qual o alvo de LDL e como tratar a dislipidemia sem estatina?"
        ),
        "reference_answer": """
**Referência — Claude Sonnet 4.6**

**Estratificação de risco cardiovascular**
- DM2 + DLP + obesidade + idade > 40a, sem evento prévio → **Alto risco cardiovascular**.
- Com evento prévio (IAM/AVC) → Muito alto risco.

**Metas de LDL (SBC 2020 / ESC 2021)**
| Risco | LDL-alvo |
|---|---|
| Baixo | < 130 mg/dL |
| Moderado | < 100 mg/dL |
| **Alto** | **< 70 mg/dL** ← este caso |
| Muito alto | < 55 mg/dL |

*LDL atual 148 → redução necessária de ~53% para atingir a meta de 70.*

**Tratamento sem estatina (alergia documentada)**
1. **Ezetimiba 10 mg/dia** — inibe absorção intestinal de colesterol; reduz LDL ~18–20%; 1ª linha quando estatina é contraindicada.
2. **Inibidor de PCSK9** (evolocumabe, alirocumabe) — reduz LDL 50–60%; indicado quando ezetimiba isolada não atinge meta; custo elevado.
3. **Bempedoico** (rosuvastatina bempedoica) — alternativa para intolerantes a estatinas; reduz LDL ~18%.
4. **Para TG 210** (moderadamente elevado): restrição de carboidratos simples + álcool; considerar fibratos se TG > 500 mg/dL (risco pancreatite).

**Medidas não farmacológicas (sempre associar)**
- Dieta mediterrânea (↓ gordura saturada, ↑ fibra solúvel).
- Atividade física aeróbica 150 min/semana.
- Perda de 5–10% do peso → reduz TG e sobe HDL.
- Tratar DM2: controle glicêmico melhora perfil lipídico.

**Reavaliar lipidograma** em 3 meses após início do tratamento.
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
    busca = st.text_input("🔍 Buscar no glossário", placeholder="Digite sigla ou palavra-chave...")
    if busca:
        mask = df.apply(lambda col: col.str.contains(busca, case=False, na=False)).any(axis=1)
        df = df[mask]
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
