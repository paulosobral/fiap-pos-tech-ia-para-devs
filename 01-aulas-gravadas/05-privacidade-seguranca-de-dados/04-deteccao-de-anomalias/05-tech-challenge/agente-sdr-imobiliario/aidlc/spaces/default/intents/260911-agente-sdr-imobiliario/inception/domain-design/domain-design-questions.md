# Domain Design Questions — Agente SDR Imobiliário B2B

## Q1: Separação entre Conversation Router e SDR Agent

O PRD descreve `conversation-router` como um componente separado de `sdr-agent`. Como deve ser a responsabilidade de cada um?

- **Conversation Router**: Apenas roteia mensagens para o fluxo apropriado (SDR, Follow-up, Anomaly)? Ou também processa comandos?
- **SDR Agent**: Gerencia o fluxo conversacional principal (atendimento, qualificação, RAG, handoff)? Ou também roteia?

**Resposta do PRD (§7.2)**:
- **Conversation Router**: Valida, recupera estado da sessão (DynamoDB), chama a engine de fluxo (sales-flow)
- **SDR Agent**: Geração de resposta via OpenRouter (Claude 3.5 Haiku) usando LiteLLM; tools de RAG lookup e scheduling
- **Note**: Os síncronos rodam como módulos/bibliotecas dentro da Lambda conversation-router — sem chamada Lambda→Lambda síncrona

[Answer]: A

---

## Q2: Separação entre Lead Qualifier e SDR Agent

O PRD descreve `lead-qualifier` como um componente separado. Como deve ser a responsabilidade de cada um?

- **Lead Qualifier**: Apenas calcula score e urgência? Ou também aplica questionário adaptativo?
- **SDR Agent**: Coleta informações do lead? Ou apenas conversa?

**Resposta do PRD (§7.2)**:
- **Lead Qualifier**: Extrai estrutura (JSON) e classifica intenção + urgência + budget; grava na ficha do lead
- **SDR Agent**: Geração de resposta via LLM; coleta informações via conversa natural (não questionário rígido)
- **Note**: Lead Qualifier é um módulo interno de sales-flow (LangGraph), não Lambda separada

[Answer]: B

---

## Q3: Separação entre Properties RAG e SDR Agent

O PRD descreve `properties-rag` como um componente separado. Como deve ser a responsabilidade de cada um?

- **Properties RAG**: Apenas busca imóveis no índice FAISS? Ou também decide quando buscar?
- **SDR Agent**: Decide quando buscar imóveis? Ou apenas recebe resultados?

**Resposta do PRD (§7.2)**:
- **Properties RAG**: Vetoriza a base sintética de imóveis (S3) + embeddings; na POC usa FAISS local (índice ~200 docs carrega em memória lambda)
- **SDR Agent**: Usa tools de RAG lookup; decide quando buscar baseado no fluxo conversacional
- **Note**: Properties RAG é um módulo interno de sales-flow, carrega índice em memória

[Answer]: A

---

## Q4: Separação entre Scheduler e Handoff

O PRD descreve `scheduler` e `handoff` como componentes separados. Como deve ser a responsabilidade de cada um?

- **Scheduler**: Apenas valida disponibilidade + grava compromisso? Ou também emite ICS?
- **Handoff**: Apenas gera resumo? Ou também envia mensagem interna?

**Resposta do PRD (§7.2)**:
- **Scheduler**: Valida data/hora, grava compromisso, emite convite .ics e notify corretor
- **Handoff**: Resumo Markdown (+ mensagem interna) para corretor no canal do time
- **Note**: Ambos são módulos internos de sales-flow, não Lambdas separadas

[Answer]: A

---

## Q5: Separação entre Anomaly Detector e o Resto

O PRD descreve `anomaly-detector` como um componente separado. Como deve ser a responsabilidade?

- **Anomaly Detector**: Job diário independente que processa conversas passadas? Ou componente que monitora em tempo real?
- **Dependência**: Chama quais componentes? Apenas lê conversas do banco? Ou também interage com SDR Agent?

**Resposta do PRD (§7.2, §8.5)**:
- **Anomaly Detector**: Job diário (EventBridge, chama Batch Lambda) que extrai features por conversa e aplica Isolation Forest + PCA + Autoencoder
- **Dependência**: Apenas lê conversas do DynamoDB; emite alerta no dashboard e bloqueio de propostas suspeitas
- **Note**: É Lambda própria, acionada por EventBridge (não módulo interno)

[Answer]: A

---

## Q6: Separação entre Dashboard e Componentes de Negócio

O PRD descreve `dash` como um componente separado. Como deve ser a arquitetura?

- **Dashboard**: Apenas UI Streamlit que chama APIs? Ou também contém lógica de negócio?
- **APIs**: Dashboard chama APIs de quais componentes? SDR Agent? Ou componentes dedicados (KPIService, AnomalyService)?

**Resposta do PRD (§7.2, §10)**:
- **Dashboard**: App Streamlit (1 página) no Community Cloud consumindo GET /api/kpis (métricas de negócio)
- **APIs**: Dashboard chama dash-api (Lambda) que agrega KPIs do DynamoDB + CloudWatch
- **Note**: Dashboard não guarda dados locais; dash-api é Lambda própria

[Answer]: A

---

## Q7: Propriedade de Entidades

Quais entidades o sistema precisa? Qual componente deve ser o dono de cada?

**Entidades candidatas**:
- Lead (dados do lead, score, urgência, intenção)
- Conversa (mensagens, contexto, sessão)
- Imóvel (dados do imóvel, índice FAISS)
- Agendamento (compromisso, ICS)
- Anomalia (features, confiança, status)
- Handoff (resumo, status)

**Resposta do PRD (§7.1, §7.3)**:
- **Lead + Conversa**: Conversation Router (DynamoDB sessões + leads — TTL 90d)
- **Imóvel**: Properties RAG (S3 catálogos imóveis + índice FAISS)
- **Agendamento**: Scheduler (calendário simulado)
- **Anomalia**: Anomaly Detector (job diário lê conversas, emite alertas)
- **Handoff**: Handoff (resumo Markdown para corretor)

[Answer]: A

---

## Q8: Interação entre Componentes

Qual estilo de interação entre componentes?

- **Sync vs Async**: Chamadas síncronas (HTTP) ou assíncronas (SQS/EventBridge)?
- **Event-driven**: Componentes emitem eventos e outros reagem? Ou chamadas diretas?

**Resposta do PRD (§7.1, §7.2)**:
- **Síncrono**: Núcleo síncrono roda como módulos internos de uma única Lambda (conversation-router) — sem Lambda→Lambda síncrono (antipattern)
- **Assíncrono**: Transcrição de áudio e sincronização com CRM via SQS (com DLQ); cadência de follow-up via Step Functions; job diário de anomalias via EventBridge
- **Event-driven**: Assíncronos usam SQS/EventBridge; síncronos usam chamadas internas

[Answer]: B

---

## Summary Confirmation

**Resumo consolidado das respostas (baseado no PRD §7.1-7.3)**:
- **Q1 (Conversation Router vs SDR Agent)**: A - Conversation Router apenas roteia, SDR Agent gerencia todo o fluxo conversacional (módulos internos de uma Lambda)
- **Q2 (Lead Qualifier vs SDR Agent)**: B - Lead Qualifier apenas calcula score, SDR Agent coleta informações + conversa (módulo interno de sales-flow)
- **Q3 (Properties RAG vs SDR Agent)**: A - Properties RAG apenas busca, SDR Agent decide quando buscar (módulo interno com índice em memória)
- **Q4 (Scheduler vs Handoff)**: A - Scheduler valida + grava + emite ICS, Handoff apenas gera resumo (módulos internos de sales-flow)
- **Q5 (Anomaly Detector)**: A - Job diário independente que lê conversas do banco e emite alertas (Lambda própria acionada por EventBridge)
- **Q6 (Dashboard vs componentes de negócio)**: A - Dashboard apenas UI, chama APIs dedicadas (dash-api Lambda)
- **Q7 (Propriedade de entidades)**: A - Lead/Conversa: Conversation Router, Imóvel: Properties RAG, Agendamento: Scheduler, Anomalia: Anomaly Detector, Handoff: Handoff
- **Q8 (Interação entre componentes)**: B - Assíncrono (SQS/EventBridge) para follow-up e anomaly, síncrono para o restante (módulos internos de uma Lambda)