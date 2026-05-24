"""
prepare_dataset.py
==================
Fase 1 — Pipeline de preparação de dados:
  1. Baixar PubMedQA (pqa_labeled) do HuggingFace
  2. Gerar registros hospitalares sintéticos (protocolos, FAQs, modelos de laudo)
  3. Anonimizar com regex
  4. Converter para o formato de instrução Alpaca
  5. Salvar processed/medical_train.jsonl  +  processed/medical_test.jsonl
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_SYNTHETIC = BASE_DIR / "data" / "synthetic"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

for _p in (DATA_RAW, DATA_SYNTHETIC, DATA_PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# ── Helpers de anonimização ───────────────────────────────────────────────────
_ANON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF]"),
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DATA]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATA]"),
    # Heurística ampla de nomes: duas ou mais palavras em Title Case após Dr./Dra./Paciente.
    (re.compile(r"\b(?:Dr\.|Dra\.|Paciente|Patient)\s+[A-ZÁÉÍÓÚÃÕÂÊÔÀÜ][a-záéíóúãõâêôàü]+(?:\s+[A-ZÁÉÍÓÚÃÕÂÊÔÀÜ][a-záéíóúãõâêôàü]+)*"), "[NOME]"),
    (re.compile(r"\bHospital\s+[A-ZÁÉÍÓÚÃÕÂÊÔÀÜ][\w\s]*"), "[HOSPITAL]"),
    (re.compile(r"\b\d{5}-\d{3}\b"), "[CEP]"),
    (re.compile(r"\b(?:\d{2}\s?)?\(?\d{2}\)?\s?\d{4,5}-\d{4}\b"), "[TELEFONE]"),
]


def anonymise(text: str) -> str:
    for pattern, replacement in _ANON_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Formatador Alpaca ─────────────────────────────────────────────────────────
ALPACA_TEMPLATE = (
    "Below is a clinical question. Write an evidence-based medical response.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

EOS_TOKEN = "<|end_of_text|>"  # sobrescrito pelo train.py com tokenizer.eos_token


def to_alpaca(instruction: str, input_ctx: str, output: str) -> dict[str, str]:
    return {
        "instruction": instruction.strip(),
        "input": input_ctx.strip(),
        "output": output.strip(),
        "text": ALPACA_TEMPLATE.format(
            instruction=instruction.strip(),
            input=input_ctx.strip(),
            output=output.strip(),
        )
        + EOS_TOKEN,
    }


# ── PubMedQA ──────────────────────────────────────────────────────────────────

def load_pubmedqa(split_ratio: float = 0.9) -> tuple[list[dict], list[dict]]:
    print("Downloading PubMedQA pqa_labeled …")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    train_records: list[dict] = []
    test_records: list[dict] = []

    for i, item in enumerate(ds):
        question: str = item["question"]
        context_list: list[str] = item["context"]["contexts"]
        long_answer: str = item["long_answer"]
        final_decision: str = item.get("final_decision", "")

        context_text = "\n".join(context_list[:3])  # mantém as 3 sentenças de contexto mais relevantes
        instruction = f"Answer the following clinical question based on PubMed evidence."
        input_ctx = f"Question: {question}\n\nContext:\n{anonymise(context_text)}"
        output = (
            f"Answer: {final_decision.upper()}. {anonymise(long_answer)}"
            if final_decision
            else anonymise(long_answer)
        )

        record = to_alpaca(instruction, input_ctx, output)
        if i / len(ds) < split_ratio:
            train_records.append(record)
        else:
            test_records.append(record)

    print(f"  PubMedQA → {len(train_records)} train / {len(test_records)} test")
    return train_records, test_records


# ── Dados médicos sintéticos ──────────────────────────────────────────────────

_PROTOCOLS: list[dict] = [
    {
        "instruction": "Descreva o protocolo hospitalar para manejo de sepse grave.",
        "input": "",
        "output": (
            "Protocolo Sepse Grave — [HOSPITAL]:\n"
            "1. Reconhecimento precoce: aplique o escore SOFA. SOFA ≥ 2 com suspeita infecciosa = sepse.\n"
            "2. Bundle 1h (Surviving Sepsis Campaign):\n"
            "   a) Dosar lactato sérico.\n"
            "   b) Hemoculturas (2 pares) antes de antibióticos.\n"
            "   c) Antibióticos de amplo espectro IV (ex.: Piperacilina/tazobactam 4,5g a cada 6h).\n"
            "   d) Cristaloide 30 mL/kg IV se hipotensão ou lactato ≥ 4 mmol/L.\n"
            "   e) Vasopressores se PAM < 65 mmHg após reposição.\n"
            "3. Reavaliar resposta em 1-2h; registrar evolução em prontuário.\n"
            "NOTA: Toda prescrição requer validação e assinatura do médico responsável."
        ),
    },
    {
        "instruction": "Qual o protocolo para controle glicêmico em pacientes internados na UTI?",
        "input": "Paciente diabético tipo 2, internado por pneumonia, glicemia 280 mg/dL.",
        "output": (
            "Protocolo Controle Glicêmico em UTI — [HOSPITAL]:\n"
            "Meta glicêmica: 140–180 mg/dL (ADA/AACE para pacientes críticos).\n"
            "1. Iniciar insulinoterapia IV quando glicemia > 180 mg/dL.\n"
            "2. Usar protocolo de infusão contínua: iniciar com 0,05–0,1 U/kg/h.\n"
            "3. Monitorar glicemia capilar a cada 1-2h até estabilização, depois a cada 4h.\n"
            "4. Evitar hipoglicemia (< 70 mg/dL): suspender insulina e oferecer dextrose 50% 20 mL IV.\n"
            "5. Transição para SC ao reintroduzir dieta: calcular dose basal-bolus.\n"
            "NOTA: Prescrição de insulina deve ser validada pelo médico assistente."
        ),
    },
    {
        "instruction": "Descreva o fluxo de atendimento de dor torácica aguda na emergência.",
        "input": "",
        "output": (
            "Fluxo Dor Torácica Aguda — [HOSPITAL]:\n"
            "1. Triagem imediata: ECG em 10 min da chegada.\n"
            "2. Sinais de alerta: elevação de ST → ativar protocolo IAM-CSST; notificar cardiologia.\n"
            "3. Exames iniciais: troponina I de alta sensibilidade (0h e 1h), CK-MB, RX tórax, O2Sat.\n"
            "4. Medidas imediatas (MONA):\n"
            "   - Morfina 2-4 mg IV se dor refratária (avaliar).\n"
            "   - Oxigênio se SatO2 < 90%.\n"
            "   - Nitratos: NTG SL 0,4 mg (contraindicado se PAS < 90 ou uso de sildenafil).\n"
            "   - AAS 200 mg VO (se sem contraindicação).\n"
            "5. Estratificação de risco: score HEART ou TIMI.\n"
            "NOTA: Toda conduta terapêutica requer prescrição e validação médica."
        ),
    },
    {
        "instruction": "Quais os critérios diagnósticos para insuficiência cardíaca congestiva (ICC)?",
        "input": "",
        "output": (
            "Critérios de Framingham para ICC (necessário ≥ 2 critérios maiores ou 1 maior + 2 menores):\n"
            "Critérios Maiores: dispneia paroxística noturna, ortopneia, turgência jugular, crepitações "
            "pulmonares, cardiomegalia ao RX, edema agudo de pulmão, galope B3, refluxo hepatojugular, "
            "perda de peso > 4,5 kg em 5 dias com tratamento.\n"
            "Critérios Menores: edema bilateral de MMII, tosse noturna, dispneia aos esforços, "
            "hepatomegalia, derrame pleural, FC > 120 bpm, perda de peso > 4,5 kg em 5 dias.\n"
            "Exames complementares: BNP/NT-proBNP, ecocardiograma (FEVE), RX tórax."
        ),
    },
    {
        "instruction": "Como interpretar um resultado de hemograma com leucocitose?",
        "input": "Leucócitos totais 18.000/μL, neutrófilos 85%, linfócitos 10%, sem bastonetes.",
        "output": (
            "Leucocitose com neutrofilia (neutrófilos 85% = ~15.300/μL) sugere:\n"
            "1. Processo infeccioso bacteriano agudo (causa mais comum).\n"
            "2. Estresse fisiológico, uso de corticosteroides ou catecolaminas.\n"
            "3. Leucemias mieloides (considerar se persistente sem causa aparente).\n"
            "Conduta orientativa:\n"
            "- Correlacionar com quadro clínico e sinais de SIRS/sepse.\n"
            "- Solicitar hemocultura se febre + foco suspeito.\n"
            "- Avaliar VHS, PCR e procalcitonina para graduar resposta inflamatória.\n"
            "NOTA: Interpretação final e prescrição devem ser realizadas pelo médico."
        ),
    },
    {
        "instruction": "Quais os critérios de Rome IV para síndrome do intestino irritável (SII)?",
        "input": "",
        "output": (
            "Critérios de Roma IV para SII:\n"
            "Dor abdominal recorrente (≥ 1 dia/semana nos últimos 3 meses), associada a ≥ 2 dos seguintes:\n"
            "1. Relacionada à defecação.\n"
            "2. Associada a mudança na frequência das fezes.\n"
            "3. Associada a mudança na forma/consistência das fezes.\n"
            "Subtypes: SII-C (constipação), SII-D (diarréia), SII-M (misto), SII-U (inespecífico).\n"
            "Sinais de alarme que exigem investigação adicional: sangue nas fezes, perda de peso "
            "involuntária, anemia, febre, história familiar de câncer colorretal."
        ),
    },
    {
        "instruction": "Qual a conduta inicial para hipertensão arterial sistêmica em urgência (PA ≥ 180/120 sem lesão de órgão-alvo)?",
        "input": "Paciente assintomático, PA 185/115 mmHg, sem cefaleia, sem déficit neurológico.",
        "output": (
            "Urgência Hipertensiva (PA ≥ 180/120 sem LOA):\n"
            "1. Afastar emergência: avaliar sintomas neurológicos, cardíacos e renais.\n"
            "2. Repouso 15-30 min em ambiente calmo; repetir PA.\n"
            "3. Via oral preferível:\n"
            "   - Captopril 25-50 mg SL ou VO (onset 15-30 min).\n"
            "   - Clonidina 0,1-0,2 mg VO como alternativa.\n"
            "4. Meta: redução de 25% da PAM em 2-6h (não normalização imediata).\n"
            "5. Acompanhamento ambulatorial com médico de referência.\n"
            "NOTA: Prescrição e ajuste de dose devem ser validados pelo médico."
        ),
    },
    {
        "instruction": "Descreva o modelo de laudo de ecocardiograma normal.",
        "input": "Exame solicitado por [NOME], realizado em [DATA].",
        "output": (
            "LAUDO DE ECOCARDIOGRAMA TRANSTORÁCICO\n"
            "Paciente: [NOME] | Data: [DATA] | Indicação: Avaliação cardiológica.\n\n"
            "Técnica: Modo M, 2D, Doppler pulsátil, contínuo e colorido.\n\n"
            "Resultados:\n"
            "- Átrio esquerdo: 36 mm (normal).\n"
            "- Ventrículo esquerdo: DDVE 48 mm, DSVE 31 mm, septo 9 mm, parede posterior 9 mm.\n"
            "- Fração de ejeção (Simpson): 65% (normal ≥ 55%).\n"
            "- Valvas: mitral, tricúspide, aórtica e pulmonar sem alterações estruturais ou "
            "hemodinâmicas significativas.\n"
            "- Derrame pericárdico: ausente.\n"
            "- Função diastólica: preservada (E/A 1,2, TDE 180 ms).\n\n"
            "Conclusão: Ecocardiograma transtorácico dentro dos limites da normalidade.\n"
            "Assinatura: Médico responsável [CRM xxxxx]."
        ),
    },
    {
        "instruction": "Quais perguntas frequentes sobre diabetes mellitus tipo 2 os médicos costumam ter?",
        "input": "",
        "output": (
            "FAQ — Diabetes Mellitus Tipo 2:\n"
            "P: Qual a meta de HbA1c para adultos sem comorbidades?\n"
            "R: < 7% (ADA 2024). Individualizar: < 8% para idosos frágeis ou comorbidades graves.\n\n"
            "P: Quando iniciar insulina em DM2?\n"
            "R: HbA1c > 10%, hiperglicemia sintomática grave, falha de 2-3 antidiabéticos orais, "
            "ou situações de crise (cirurgia, UTI).\n\n"
            "P: Metformina é contraindicada em qual nível de TFGe?\n"
            "R: Suspender com TFGe < 30 mL/min/1,73m². Reduzir dose com TFGe 30-45.\n\n"
            "P: Isglt2 pode ser usado em insuficiência cardíaca?\n"
            "R: Sim. Empagliflozina e dapagliflozina têm benefício cardiovascular comprovado "
            "(EMPA-REG, DAPA-HF).\n\n"
            "NOTA: Decisão terapêutica individual requer avaliação médica."
        ),
    },
    {
        "instruction": "Descreva o modelo de prescrição hospitalar para pneumonia adquirida na comunidade (PAC) leve.",
        "input": "Paciente adulto, sem comorbidades, sem critérios de internação (PSI I-II).",
        "output": (
            "MODELO DE PRESCRIÇÃO — PAC LEVE (Ambulatorial):\n"
            "1. Amoxicilina 500 mg VO 8/8h × 5-7 dias.\n"
            "   OU Azitromicina 500 mg VO/dia × 5 dias (se atípicos suspeitos).\n"
            "2. Dipirona sódica 1g VO 6/6h se febre/dor.\n"
            "3. Hidratação oral adequada (≥ 2L/dia).\n"
            "4. Repouso relativo.\n"
            "5. Retorno se piora clínica, dispneia, cianose ou não melhora em 48-72h.\n"
            "6. Reavaliação em 48h.\n\n"
            "NOTA: Esta é um modelo orientativo. Toda prescrição deve ser assinada "
            "pelo médico responsável após avaliação individual do paciente."
        ),
    },
]

_FAQ_TEMPLATES: list[dict] = [
    {
        "instruction": "Explique a diferença entre AVC isquêmico e hemorrágico.",
        "input": "",
        "output": (
            "AVC Isquêmico: oclusão arterial por trombo ou êmbolo → isquemia focal.\n"
            "Corresponde a ~85% dos AVCs. TC sem contraste: normal nas primeiras horas.\n"
            "Tratamento: trombolítico (rtPA IV) se < 4,5h do início, sem contraindicações.\n\n"
            "AVC Hemorrágico: ruptura de vaso → sangramento intraparenquimatoso ou subaracnóideo.\n"
            "TC sem contraste: hiperdensidade imediata.\n"
            "Tratamento: controle pressórico rigoroso, reversão de anticoagulação, "
            "neurointensivismo; cirurgia em casos selecionados.\n"
            "NOTA: Diagnóstico e tratamento exigem avaliação neurológica e de imagem urgentes."
        ),
    },
    {
        "instruction": "Quando solicitar troponina de alta sensibilidade?",
        "input": "Paciente com dor torácica atípica de 2h de duração.",
        "output": (
            "Troponina de alta sensibilidade (hsTn) — indicações:\n"
            "1. Suspeita de SCA (IAM sem supra ou angina instável).\n"
            "2. Dor torácica aguda sem causa clara.\n"
            "Protocolo 0h/1h (ESC 2023):\n"
            "- Colher na chegada (T0) e 1h depois (T1).\n"
            "- Regra de exclusão: hsTn T0 < LoD (limite de detecção).\n"
            "- Regra de inclusão: hsTn T0 muito elevada ou delta T1-T0 > 6 ng/L (kit-dependente).\n"
            "Causas não-coronárias de elevação: miocardite, PE, IRA, sepse.\n"
            "NOTA: Interpretação requer correlação clínica pelo médico."
        ),
    },
    {
        "instruction": "Como calcular o escore de risco CHADS2-VASc para FA?",
        "input": "Paciente masculino, 68 anos, HAS, sem DM, sem AVC prévio, sem ICC.",
        "output": (
            "CHA₂DS₂-VASc:\n"
            "C — ICC: 0 | H — HAS: +1 | A₂ — Idade ≥ 75: 0 | D — DM: 0\n"
            "S₂ — AVC/TIA: 0 | V — Doença vascular: 0 | A — Idade 65-74: +1 | Sc — Sexo feminino: 0\n"
            "Total: 2 pontos → risco moderado-alto → anticoagulação oral recomendada.\n"
            "Preferir NOACs (apixabana, rivaroxabana, dabigatrana) sobre varfarina.\n"
            "NOTA: Decisão de anticoagular deve ser tomada pelo médico considerando risco de sangramento."
        ),
    },
    {
        "instruction": "Quais os critérios de internação para pneumonia adquirida na comunidade?",
        "input": "",
        "output": (
            "Escores para decisão de internação em PAC:\n"
            "PSI (Pneumonia Severity Index): Classes III-V → internar; I-II → ambulatorial.\n"
            "CURB-65 (1 ponto cada):\n"
            "- Confusão mental | Ureia > 43 mg/dL | FR ≥ 30 ipm | PAS < 90 ou PAD ≤ 60 | Idade ≥ 65\n"
            "Score 0-1: ambulatorial; 2: internação clínica; ≥ 3: avaliar UTI.\n"
            "Critérios de UTI (ATS/IDSA): 1 critério maior (ventilação mecânica ou choque séptico) "
            "ou ≥ 3 critérios menores."
        ),
    },
    {
        "instruction": "Quais os efeitos adversos mais comuns dos inibidores da ECA?",
        "input": "",
        "output": (
            "Efeitos adversos dos iECA (ex.: enalapril, captopril, ramipril):\n"
            "1. Tosse seca (10-15%): mecanismo → acúmulo de bradicinina. Trocar por BRA se intratável.\n"
            "2. Hipercalemia: monitorar K+ especialmente com TFGe < 45 ou uso de diuréticos poupadores.\n"
            "3. Hipotensão de primeira dose: iniciar com dose baixa, orientar paciente.\n"
            "4. Angioedema (raro, < 1%): emergência médica; contraindicação absoluta permanente.\n"
            "5. Piora de função renal: aumento de creatinina até 30% é aceitável; > 30% → suspender.\n"
            "Contraindicações absolutas: gravidez (teratogênico), angioedema prévio por iECA, estenose "
            "bilateral de artéria renal."
        ),
    },
]


def generate_synthetic_records() -> list[dict]:
    records = []
    for item in _PROTOCOLS + _FAQ_TEMPLATES:
        records.append(
            to_alpaca(item["instruction"], item.get("input", ""), item["output"])
        )
    return records


# ── MedQuAD ───────────────────────────────────────────────────────────────────

def load_medquad(split_ratio: float = 0.9, max_records: int | None = None) -> tuple[list[dict], list[dict]]:
    """Load MedQuAD from HuggingFace (lavita/MedQuAD).

    Fields used: Question, Answer, qtype (used to build instruction).
    Records with empty Answer are skipped.
    """
    print("Downloading MedQuAD …")
    ds = load_dataset("lavita/MedQuAD", split="train")

    train_records: list[dict] = []
    test_records: list[dict] = []

    items = list(ds)
    if max_records is not None:
        random.shuffle(items)
        items = items[:max_records]

    for i, item in enumerate(items):
        question: str = (item.get("question") or item.get("Question") or "").strip()
        answer: str = (item.get("answer") or item.get("Answer") or "").strip()
        qtype: str = (item.get("qtype") or "").strip()

        if not question or not answer:
            continue

        instruction = (
            f"Answer the following medical question ({qtype})."
            if qtype
            else "Answer the following medical question."
        )
        input_ctx = f"Question: {anonymise(question)}"
        output = anonymise(answer)

        record = to_alpaca(instruction, input_ctx, output)
        if i / len(items) < split_ratio:
            train_records.append(record)
        else:
            test_records.append(record)

    print(f"  MedQuAD → {len(train_records)} train / {len(test_records)} test")
    return train_records, test_records


# ── Principal ─────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)

    pubmed_train, pubmed_test = load_pubmedqa()
    medquad_train, medquad_test = load_medquad()
    synthetic = generate_synthetic_records()

    train_records = pubmed_train + medquad_train + synthetic
    random.shuffle(train_records)

    # Salva os datasets processados.
    train_path = DATA_PROCESSED / "medical_train.jsonl"
    test_path = DATA_PROCESSED / "medical_test.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    test_records = pubmed_test + medquad_test
    random.shuffle(test_records)

    with test_path.open("w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Também salva os registros sintéticos separadamente para referência.
    synthetic_path = DATA_SYNTHETIC / "synthetic_records.jsonl"
    with synthetic_path.open("w", encoding="utf-8") as f:
        for rec in synthetic:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"\nDataset saved:\n"
        f"  Train: {len(train_records)} records "
        f"(PubMedQA={len(pubmed_train)}, MedQuAD={len(medquad_train)}, Synthetic={len(synthetic)}) "
        f"→ {train_path}\n"
        f"  Test:  {len(test_records)} records "
        f"(PubMedQA={len(pubmed_test)}, MedQuAD={len(medquad_test)}) "
        f"→ {test_path}\n"
        f"  Synthetic: {len(synthetic)} records → {synthetic_path}"
    )


if __name__ == "__main__":
    main()
