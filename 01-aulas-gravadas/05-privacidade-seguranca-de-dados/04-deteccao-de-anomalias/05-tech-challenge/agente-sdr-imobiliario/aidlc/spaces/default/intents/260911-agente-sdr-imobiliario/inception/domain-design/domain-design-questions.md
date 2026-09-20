# Domain Design Questions — Agente SDR Imobiliário B2B

## Q1: Separação entre Conversation Router e SDR Agent

O PRD descreve `conversation-router` como um componente separado de `sdr-agent`. Como deve ser a responsabilidade de cada um?

- **Conversation Router**: Apenas roteia mensagens para o fluxo apropriado (SDR, Follow-up, Anomaly)? Ou também processa comandos?
- **SDR Agent**: Gerencia o fluxo conversacional principal (atendimento, qualificação, RAG, handoff)? Ou também roteia?

**Opções**:
- A) Conversation Router apenas roteia, SDR Agent gerencia todo o fluxo conversacional
- B) Conversation Router roteia + processa comandos simples, SDR Agent gerencia lógica complexa
- C) Conversation Router e SDR Agent são um único componente (ConversationOrchestrator)
- D) Outro (especifique)

---

## Q2: Separação entre Lead Qualifier e SDR Agent

O PRD descreve `lead-qualifier` como um componente separado. Como deve ser a responsabilidade de cada um?

- **Lead Qualifier**: Apenas calcula score e urgência? Ou também aplica questionário adaptativo?
- **SDR Agent**: Coleta informações do lead? Ou apenas conversa?

**Opções**:
- A) Lead Qualifier calcula score + aplica questionário, SDR Agent apenas conversa
- B) Lead Qualifier apenas calcula score, SDR Agent coleta informações + conversa
- C) Lead Qualifier e SDR Agent são um único componente (QualificationAgent)
- D) Outro (especifique)

---

## Q3: Separação entre Properties RAG e SDR Agent

O PRD descreve `properties-rag` como um componente separado. Como deve ser a responsabilidade de cada um?

- **Properties RAG**: Apenas busca imóveis no índice FAISS? Ou também decide quando buscar?
- **SDR Agent**: Decide quando buscar imóveis? Ou apenas recebe resultados?

**Opções**:
- A) Properties RAG apenas busca, SDR Agent decide quando buscar
- B) Properties RAG decide quando buscar + busca, SDR Agent apenas recebe resultados
- C) Properties RAG e SDR Agent são um único componente (PropertySearchAgent)
- D) Outro (especifique)

---

## Q4: Separação entre Scheduler e Handoff

O PRD descreve `scheduler` e `handoff` como componentes separados. Como deve ser a responsabilidade de cada um?

- **Scheduler**: Apenas valida disponibilidade + grava compromisso? Ou também emite ICS?
- **Handoff**: Apenas gera resumo? Ou também envia mensagem interna?

**Opções**:
- A) Scheduler valida + grava + emite ICS, Handoff apenas gera resumo
- B) Scheduler apenas valida + grava, Handoff gera resumo + emite ICS + envia mensagem
- C) Scheduler e Handoff são um único componente (SchedulingAgent)
- D) Outro (especifique)

---

## Q5: Separação entre Anomaly Detector e o Resto

O PRD descreve `anomaly-detector` como um componente separado. Como deve ser a responsabilidade?

- **Anomaly Detector**: Job diário independente que processa conversas passadas? Ou componente que monitora em tempo real?
- **Dependência**: Chama quais componentes? Apenas lê conversas do banco? Ou também interage com SDR Agent?

**Opções**:
- A) Job diário independente que lê conversas do banco e emite alertas
- B) Monitor em tempo real que interage com SDR Agent
- C) Anomaly Detector é parte do SDR Agent (não separado)
- D) Outro (especifique)

---

## Q6: Separação entre Dashboard e Componentes de Negócio

O PRD descreve `dash` como um componente separado. Como deve ser a arquitetura?

- **Dashboard**: Apenas UI Streamlit que chama APIs? Ou também contém lógica de negócio?
- **APIs**: Dashboard chama APIs de quais componentes? SDR Agent? Ou componentes dedicados (KPIService, AnomalyService)?

**Opções**:
- A) Dashboard apenas UI, chama APIs dedicadas (KPIService, AnomalyService, LeadService)
- B) Dashboard UI + lógica de agregação, chama componentes de negócio diretamente
- C) Dashboard chama SDR Agent para tudo (gateway pattern)
- D) Outro (especifique)

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

**Opções**:
- A) Lead: SDR Agent, Conversa: Conversation Router, Imóvel: Properties RAG, Agendamento: Scheduler, Anomalia: Anomaly Detector, Handoff: Handoff
- B) Lead: LeadRepository, Conversa: ConversationRepository, Imóvel: PropertyRepository, Agendamento: SchedulingRepository, Anomalia: AnomalyRepository, Handoff: HandoffRepository (repositories separados)
- C) Tudo em um único componente (MonolithAgent)
- D) Outro (especifique)

---

## Q8: Interação entre Componentes

Qual estilo de interação entre componentes?

- **Sync vs Async**: Chamadas síncronas (HTTP) ou assíncronas (SQS/EventBridge)?
- **Event-driven**: Componentes emitem eventos e outros reagem? Ou chamadas diretas?

**Opções**:
- A) Síncrono (HTTP) para tudo, simples para POC
- B) Assíncrono (SQS/EventBridge) para follow-up e anomaly, síncrono para o restante
- C) Event-driven para tudo (SQS/EventBridge)
- D) Outro (especifique)