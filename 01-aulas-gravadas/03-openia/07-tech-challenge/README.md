# Assistente Médico IA — Tech Challenge Fase 3

Assistente virtual médico com LLM fine-tuned, RAG sobre protocolos hospitalares e fluxo multi-agente via LangGraph.

---

## Quick Start

```bash
# 1. Instalar dependências (GPU local recomendada)
pip install -r requirements.txt
python -m spacy download pt_core_news_sm

# 2. Preparar dataset de treinamento
python fine_tuning/prepare_dataset.py

# 3. Fine-tuning (requer GPU com ≥ 16 GB VRAM)
python fine_tuning/train.py

# 4. Avaliar o modelo
python fine_tuning/eval_model.py

# 5. Iniciar o assistente
streamlit run app.py
```

---

## Pré-requisitos

| Componente | Versão mínima |
|---|---|
| Python | 3.10+ |
| CUDA | 11.8+ (para fine-tuning) |
| VRAM | ≥ 16 GB (fine-tuning); ≥ 8 GB (inferência 4-bit) |
| GPU | NVIDIA (Unsloth é CUDA-only) |

> O app funciona sem fine-tuning: define `use_adapter=False` em `assistant/llm_loader.py` para usar o modelo base como fallback.

---

## Estrutura do Projeto

```
07-tech-challenge/
├── fine_tuning/
│   ├── prepare_dataset.py   # Baixa PubMedQA, gera dados sintéticos, formata Alpaca
│   ├── train.py             # Fine-tuning: Unsloth + LoRA + SFTTrainer
│   ├── eval_model.py        # ROUGE-L, BLEU, exemplos qualitativos
│   └── output/              # Adapter LoRA salvo aqui após treino
├── assistant/
│   ├── llm_loader.py        # Carrega modelo (adapter ou fallback)
│   ├── vector_store.py      # FAISS sobre protocolos hospitalares
│   ├── patient_db.py        # SQLite com 20 pacientes sintéticos
│   └── rag_chain.py         # ConversationalRetrievalChain (LangChain)
├── langgraph_flow/
│   ├── state.py             # PatientState (TypedDict compartilhado)
│   ├── graph.py             # StateGraph + roteamento condicional
│   └── agents/
│       ├── triage_agent.py    # Classifica urgência (low/medium/high)
│       ├── exam_agent.py      # Busca exames pendentes/concluídos
│       ├── diagnosis_agent.py # Diagnóstico diferencial via RAG
│       ├── pharmacy_agent.py  # Sugestões farmacológicas (com guardrail)
│       └── alert_agent.py     # Emite alertas de urgência alta
├── security/
│   ├── guardrails.py        # Valida input, bloqueia injection, enforcement de disclaimer
│   ├── audit_logger.py      # Log JSON estruturado (data/logs/audit.jsonl)
│   └── explainability.py    # Citação de fontes + footer de rastreabilidade
├── data/
│   ├── raw/                 # PubMedQA bruto
│   ├── synthetic/           # Protocolos/laudos/FAQs gerados
│   ├── processed/           # medical_train.jsonl, medical_test.jsonl
│   ├── faiss_index/         # Índice vetorial persistido
│   ├── patient_records.db   # SQLite de pacientes sintéticos
│   └── logs/audit.jsonl     # Log de auditoria
├── app.py                   # Interface Streamlit
└── requirements.txt
```

---

## Arquitetura

### Fluxo LangGraph

```mermaid
graph TD
    A([Sintomas do Paciente]) --> B[triage_node\nClassifica urgência]
    B -->|urgência = high| C[alert_node\nEmite alerta + log]
    B -->|urgência = low/medium| D[exam_node\nBusca exames no SQLite]
    C --> D
    D --> E[diagnosis_node\nDiagnóstico diferencial RAG+LLM]
    E --> F{Interrupt\nAprovação médica}
    F --> G[pharmacy_node\nSugestões farmacológicas]
    G --> H([END])
```

### Pipeline RAG (LangChain)

```
Pergunta → guardrails.validate_input()
         → ConversationalRetrievalChain
               ├─ FAISS retriever (protocolos hospitalares)
               └─ HuggingFacePipeline (LLaMA-3-8B fine-tuned)
         → explainability.attach_sources()
         → audit_logger.log_query()
         → Resposta com fontes citadas
```

---

## Dados

| Dataset | Origem | Registros | Uso |
|---|---|---|---|
| PubMedQA `pqa_labeled` | HuggingFace | ~1.000 | Treino/teste |
| Protocolos sintéticos | Gerado em `prepare_dataset.py` | 15 | Treino |
| FAQs sintéticos | Gerado em `prepare_dataset.py` | 5 | Treino |
| Pacientes sintéticos | `patient_db.py` | 20 | Runtime (SQLite) |

Todos os dados são **públicos ou sintéticos** — nenhum dado real de paciente é utilizado. Os dados sintéticos passam por anonimização com regex antes do treinamento.

---

## Fine-tuning

| Parâmetro | Valor |
|---|---|
| Modelo base | `unsloth/llama-3-8b-bnb-4bit` |
| Método | LoRA (`r=16`, `alpha=16`) |
| Épocas | 3 |
| Batch size | 2 (+ grad. accum. 4 = efetivo 8) |
| Learning rate | 2e-4 (cosine decay) |
| Max seq length | 2048 tokens |
| Formato | Alpaca (`### Instruction / ### Input / ### Response`) |
| Output | `fine_tuning/output/lora_model/` |

---

## Segurança

- **Sem prescrição direta**: toda resposta com conteúdo farmacológico inclui disclaimer obrigatório.
- **Guardrails de input**: bloqueio de prompt injection (regex), limite de 2.000 chars.
- **Auditoria**: cada query, step de agente e alerta é gravado em `data/logs/audit.jsonl`.
- **Explainability**: fontes dos protocolos consultados são exibidas em toda resposta.
- **Human-in-the-loop**: o fluxo LangGraph tem um ponto de interrupção antes das sugestões farmacológicas.

---

## Interface Streamlit

Acesse em `http://localhost:8501` após rodar `streamlit run app.py`.

| Aba | Função |
|---|---|
| 💬 Assistente | Seleção de paciente, chat com LangGraph ou RAG direto |
| 📋 Auditoria | Visualização dos últimos N registros do log JSON |

---

## Avaliação do Modelo

Após o fine-tuning, execute:

```bash
python fine_tuning/eval_model.py
```

Resultados salvos em `fine_tuning/output/eval_results.json`.

Métricas: ROUGE-1, ROUGE-2, ROUGE-L, BLEU (calculados sobre o split de teste do PubMedQA).

---

## Licença

Este projeto foi desenvolvido para fins acadêmicos no curso FIAP Pós Tech — IA para Devs, Fase 3.
Dados de treinamento: PubMedQA (MIT License). Modelo base: LLaMA 3 (Meta Llama 3 Community License).
