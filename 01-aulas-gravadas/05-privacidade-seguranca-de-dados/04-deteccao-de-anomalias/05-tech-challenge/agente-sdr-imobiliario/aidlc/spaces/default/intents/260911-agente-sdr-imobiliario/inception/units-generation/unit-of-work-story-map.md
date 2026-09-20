# Unit of Work Story Map — Agente SDR Imobiliário B2B

> Estágio Units Generation (Inception). Fonte: stories.md, components.md, unit-of-work.md.

---

## Story to Unit Mapping

| Story ID | Story Title | Implementing Unit | Unit Directory | Notes |
|-----------|-------------|-------------------|---------------|-------|
| US1.1 | Iniciar Conversa | U1 | u1-core-conversation | ConversationRouter + SecurityLayer |
| US1.2 | Conversar com Texto Livre | U1 | u1-core-conversation | ConversationRouter + SDR Agent |
| US1.3 | Enviar Mensagem de Voz | U2 | u2-async-voice | VoiceAdapter |
| US2.1 | Detectar Intenção | U1 | u1-core-conversation | LeadQualifier + SDR Agent |
| US2.2 | Coletar Informações Básicas | U1 | u1-core-conversation | LeadQualifier + SDR Agent |
| US3.1 | Calcular Score de Prontidão | U1 | u1-core-conversation | LeadQualifier |
| US3.2 | Qualificar para Handoff | U1 | u1-core-conversation | LeadQualifier + Handoff |
| US4.1 | Buscar Imóveis Relevantes | U1 | u1-core-conversation | PropertiesRAG |
| US4.2 | Apresentar Opções | U1 | u1-core-conversation | SDR Agent |
| US5.1 | Agendar Visita | U1 | u1-core-conversation | Scheduler |
| US6.1 | Gerar Handoff | U1 | u1-core-conversation | Handoff |
| US7.1 | Monitorar KPIs | U7 | u7-dashboard | DashAPI + Dashboard |
| US9.1 | Detectar Comportamento Anômalo | U5 | u5-anomaly | AnomalyDetector |
| US11.1 | Sincronizar Lead Qualificado | U3 | u3-async-crm | CRMAdapter |
| US10.1 | Roleta de Distribuição | U1 | u1-core-conversation | LeadRouter |
| US8.1 | Follow-up Automático | U6 | u6-followup | Followup |
| US12.1 | Ingestão de E-mail | U4 | u4-async-ingest | ContactIngest |
| US13.1 | Meta de Qualificação da POC (≥ 60%) | U1 | u1-core-conversation | LeadQualifier |
| US14.1 | Segurança de Dados e Privacidade | U1 | u1-core-conversation | SecurityLayer |
| US15.1 | Observabilidade e Monitoramento | U7 | u7-dashboard | DashAPI |
| US16.1 | Escalabilidade, Custo e IaC | Deferred | infrastructure-design | Deferred para estágio posterior |

---

## Story Implementation Order Within Each Unit

### U1: Core Conversation

**Ordem sugerida (walking skeleton primeiro):**
1. US1.1 — Iniciar Conversa (sessão básica)
2. US1.2 — Conversar com Texto Livre (SDR Agent básico)
3. US2.1 — Detectar Intenção (LeadQualifier básico)
4. US2.2 — Coletar Informações Básicas (questionário)
5. US3.1 — Calcular Score de Prontidão (scoring)
6. US3.2 — Qualificar para Handoff (threshold)
7. US4.1 — Buscar Imóveis Relevantes (RAG básico)
8. US4.2 — Apresentar Opções (recomendação)
9. US5.1 — Agendar Visita (scheduler básico)
10. US6.1 — Gerar Handoff (resumo básico)
11. US10.1 — Roleta de Distribuição (regras)
12. US13.1 — Meta de Qualificação (≥ 60%)
13. US14.1 — Segurança de Dados (PII masking)

### U2: Async Voice

**Ordem sugerida:**
1. US1.3 — Enviar Mensagem de Voz (transcrição básica)

### U3: Async CRM

**Ordem sugerida:**
1. US11.1 — Sincronizar Lead Qualificado (MCP básico)

### U4: Async Ingest

**Ordem sugerida:**
1. US12.1 — Ingestão de E-mail (extração básica)

### U5: Anomaly

**Ordem sugerida:**
1. US9.1 — Detectar Comportamento Anômalo (Isolation Forest básico)

### U6: Followup

**Ordem sugerida:**
1. US8.1 — Follow-up Automático (cadência básica)

### U7: Dashboard

**Ordem sugerida:**
1. US15.1 — Observabilidade e Monitoramento (CloudWatch)
2. US7.1 — Monitorar KPIs (UI básica)

---

## Cross-Cutting Concerns

**Segurança (US14.1):**
- Principalmente U1 (SecurityLayer)
- Aplica a todas as unidades que acessam dados sensíveis

**Observabilidade (US15.1):**
- Principalmente U7 (DashAPI)
- Todas as unidades devem emitir logs estruturados (CloudWatch)

**Escalabilidade/Custo (US16.1):**
- Deferred para infrastructure-design
- Afeta todas as unidades (Lambda configuration, Terraform)

---

## Coverage Verification

**Todas as stories atribuídas:**
- 21 stories mapeadas para 7 unidades
- 1 story (US16.1) deferred para infrastructure-design
- Todas as unidades têm stories atribuídas

**U1 tem 13 stories** (maior complexidade — walking skeleton)
**U2 tem 1 story** (foco específico)
**U3 tem 1 story** (foco específico)
**U4 tem 1 story** (foco específico)
**U5 tem 1 story** (foco específico)
**U6 tem 1 story** (foco específico)
**U7 tem 2 stories** (UI + API)