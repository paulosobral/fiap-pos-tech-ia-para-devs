"""
vector_store.py
===============
Builds and manages a FAISS vector store over the medical protocol documents.
The index is persisted to disk so it is rebuilt only when protocols change.

Usage:
    from assistant.vector_store import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("protocolo sepse")
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4

# ── Protocol corpus (loaded at import time) ───────────────────────────────────
# In a production system these would be read from files/database.
_PROTOCOL_DOCS: list[dict[str, str]] = [
    {
        "source": "Protocolo Sepse Grave",
        "content": (
            "Protocolo Sepse Grave — Hospital:\n"
            "1. Reconhecimento precoce: aplique o escore SOFA. SOFA ≥ 2 com suspeita infecciosa = sepse.\n"
            "2. Bundle 1h (Surviving Sepsis Campaign):\n"
            "   a) Dosar lactato sérico.\n"
            "   b) Hemoculturas (2 pares) antes de antibióticos.\n"
            "   c) Antibióticos de amplo espectro IV.\n"
            "   d) Cristaloide 30 mL/kg IV se hipotensão ou lactato ≥ 4 mmol/L.\n"
            "   e) Vasopressores se PAM < 65 mmHg após reposição.\n"
            "3. Reavaliar resposta em 1-2h; registrar evolução em prontuário."
        ),
    },
    {
        "source": "Protocolo Controle Glicêmico UTI",
        "content": (
            "Protocolo Controle Glicêmico em UTI:\n"
            "Meta glicêmica: 140–180 mg/dL (ADA/AACE para pacientes críticos).\n"
            "1. Iniciar insulinoterapia IV quando glicemia > 180 mg/dL.\n"
            "2. Usar protocolo de infusão contínua: iniciar com 0,05–0,1 U/kg/h.\n"
            "3. Monitorar glicemia capilar a cada 1-2h até estabilização, depois a cada 4h.\n"
            "4. Evitar hipoglicemia (< 70 mg/dL).\n"
            "5. Transição para SC ao reintroduzir dieta."
        ),
    },
    {
        "source": "Protocolo Dor Torácica Aguda",
        "content": (
            "Fluxo Dor Torácica Aguda na Emergência:\n"
            "1. Triagem imediata: ECG em 10 min da chegada.\n"
            "2. Sinais de alerta: elevação de ST → ativar protocolo IAM-CSST.\n"
            "3. Exames: troponina I de alta sensibilidade (0h e 1h), CK-MB, RX tórax.\n"
            "4. Medidas MONA: morfina, oxigênio, nitratos, AAS.\n"
            "5. Estratificação de risco: score HEART ou TIMI."
        ),
    },
    {
        "source": "Protocolo ICC — Critérios Framingham",
        "content": (
            "Insuficiência Cardíaca Congestiva — Critérios de Framingham:\n"
            "Necessário ≥ 2 critérios maiores OU 1 maior + 2 menores.\n"
            "Maiores: dispneia paroxística noturna, ortopneia, turgência jugular, crepitações, "
            "cardiomegalia, edema agudo de pulmão, galope B3, refluxo hepatojugular.\n"
            "Menores: edema bilateral MMII, tosse noturna, dispneia esforço, hepatomegalia, "
            "derrame pleural, FC > 120 bpm.\n"
            "Exames: BNP/NT-proBNP, ecocardiograma, RX tórax."
        ),
    },
    {
        "source": "Protocolo PAC — Pneumonia Adquirida na Comunidade",
        "content": (
            "Pneumonia Adquirida na Comunidade (PAC) — Critérios de Internação:\n"
            "CURB-65: Confusão, Ureia > 43, FR ≥ 30, PAS < 90, Idade ≥ 65.\n"
            "Score 0-1: ambulatorial; 2: internação; ≥ 3: avaliar UTI.\n"
            "Tratamento PAC leve (ambulatorial):\n"
            "  Amoxicilina 500 mg VO 8/8h × 5-7 dias OU Azitromicina 500 mg/dia × 5 dias.\n"
            "Critérios de UTI (ATS/IDSA): ventilação mecânica ou choque séptico."
        ),
    },
    {
        "source": "Protocolo AVC — Acidente Vascular Cerebral",
        "content": (
            "AVC Isquêmico — Protocolo Emergência:\n"
            "1. TC crânio sem contraste imediatamente.\n"
            "2. Janela trombolítica: rtPA IV se ≤ 4,5h do início, sem contraindicações.\n"
            "3. Contraindicações rtPA: AVC ou TCE nos últimos 3 meses, cirurgia intracraniana, "
            "hemorragia prévia, glicemia < 50 ou > 400, PA > 185/110 não controlável.\n"
            "4. Trombectomia mecânica: considerar até 24h com imagem de perfusão favorável.\n"
            "AVC Hemorrágico:\n"
            "  Controlar PA: alvo < 140 mmHg sistólica. Reverter anticoagulação."
        ),
    },
    {
        "source": "Protocolo Hipertensão Arterial — Urgência e Emergência",
        "content": (
            "Hipertensão Arterial:\n"
            "Urgência (PA ≥ 180/120 sem LOA): Captopril 25-50 mg SL/VO ou Clonidina 0,1-0,2 mg VO.\n"
            "Meta: redução 25% PAM em 2-6h.\n"
            "Emergência (PA ≥ 180/120 COM LOA — encefalopatia, EAP, IAM, dissecção aórtica):\n"
            "  Nitroprussiato sódico IV ou Labetalol IV. UTI obrigatória.\n"
            "  Meta: redução 25% PAM em 1h (exceto AVC isquêmico: manter PA < 185/110 se rtPA)."
        ),
    },
    {
        "source": "FAQ — Diabetes Mellitus Tipo 2",
        "content": (
            "Diabetes Mellitus Tipo 2 — FAQ Clínico:\n"
            "Meta HbA1c: < 7% geral; < 8% idosos frágeis.\n"
            "Iniciar insulina: HbA1c > 10%, falha de 2-3 antidiabéticos orais, crise metabólica.\n"
            "Metformina: suspender com TFGe < 30 mL/min/1,73m².\n"
            "SGLT2 (empagliflozina, dapagliflozina): benefício cardiovascular em ICC com FE reduzida.\n"
            "GLP-1 (semaglutida, liraglutida): redução de eventos cardiovasculares em DM2 + DCV estabelecida."
        ),
    },
    {
        "source": "FAQ — Interpretação de Hemograma",
        "content": (
            "Hemograma — Interpretação Clínica:\n"
            "Leucocitose com neutrofilia: infecção bacteriana, estresse, corticosteroides.\n"
            "Leucocitose com linfocitose: infecção viral, leucemia linfocítica crônica.\n"
            "Leucopenia: infecção viral grave, aplasia medular, quimioterapia.\n"
            "Anemia microcítica hipocrômica: deficiência de ferro (mais comum), talassemia, doença crônica.\n"
            "Anemia macrocítica: deficiência de B12 ou folato, hipotireoidismo, álcool.\n"
            "Trombocitopenia < 50.000: risco de sangramento; < 10.000: risco de sangramento espontâneo."
        ),
    },
    {
        "source": "FAQ — Escore CHA2DS2-VASc e Anticoagulação em FA",
        "content": (
            "Fibrilação Atrial — Escore CHA₂DS₂-VASc:\n"
            "C=ICC, H=HAS, A₂=Idade≥75(×2), D=DM, S₂=AVC/TIA(×2), V=DCV, A=Idade 65-74, Sc=Feminino.\n"
            "Score ≥ 2 (homens) ou ≥ 3 (mulheres): anticoagulação recomendada.\n"
            "Preferir NOACs: apixabana, rivaroxabana, dabigatrana (sobre varfarina).\n"
            "Contraindicação absoluta: estenose mitral reumática grave → varfarina."
        ),
    },
]


def _build_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs: list[Document] = []
    for item in _PROTOCOL_DOCS:
        chunks = splitter.split_text(item["content"])
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": item["source"]}))
    return docs


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(force_rebuild: bool = False) -> FAISS:
    faiss_file = INDEX_DIR / "index.faiss"

    if faiss_file.exists() and not force_rebuild:
        print("[vector_store] Loading existing FAISS index …")
        embeddings = _get_embeddings()
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    print("[vector_store] Building FAISS index …")
    docs = _build_documents()
    embeddings = _get_embeddings()
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(INDEX_DIR))
    print(f"[vector_store] Index built with {len(docs)} chunks and saved to {INDEX_DIR}")
    return store


def get_retriever(top_k: int = TOP_K):
    store = build_vector_store()
    return store.as_retriever(search_kwargs={"k": top_k})
