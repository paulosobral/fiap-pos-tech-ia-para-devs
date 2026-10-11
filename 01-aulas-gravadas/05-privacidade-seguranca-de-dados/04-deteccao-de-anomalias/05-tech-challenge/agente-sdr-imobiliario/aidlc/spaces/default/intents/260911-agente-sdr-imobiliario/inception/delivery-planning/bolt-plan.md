# Bolt Plan — Agente SDR Imobiliário B2B

> Estágio Delivery Planning (Inception). Fonte: requirements.md, stories.md, components.md, unit-of-work.md, unit-of-work-dependency.md, delivery-planning-questions (Q1-A, Q2-A, Q3-A, Q4-A, Q5-A), team-practices (Walking Skeleton).

---

## Bolt Sequence (WSJF Score)

| Bolt ID | Bolt Name | Units | Stories | Confidence Hypothesis | WSJF Score | Notes |
|---------|-----------|-------|--------|----------------------|-----------|-------|
| B1 | Walking Skeleton | U1 | US1.1, US1.2, US2.1, US2.2 (básico) | Telegram → Lambda → LLM funciona end-to-end | High / 1 / High | Prioridade alta (prática Walking Skeleton) |
| B2 | Core Conversation Complete | U1 | US1.1-US1.3, US2.1-US2.2, US3.1-US3.2, US4.1-US4.2, US5.1, US6.1, US10.1, US13.1, US14.1 | Conversação completa com RAG, qualificação, agendamento, handoff, roleta | LLM + RAG + SecurityLayer funcionam juntos | High / High / Medium | Depende de B1 |
| B3 | Async Voice | U2 | US1.3 (completo) | Transcrição de voz funciona assincronamente | Medium / Low / Low | Depende de B1 (SQS ready) |
| B4 | Async CRM | U3 | US11.1 (completo) | CRM integration funciona assincronamente | Medium / Low / Low | Depende de B1 (SQS ready) |
| B5 | Async Ingest | U4 | US12.1 (completo) | Ingestão de e-mail funciona assincronamente | Low / Low / Low | Depende de B1 (DynamoDB ready) |
| B6 | Anomaly | U5 | US9.1 (completo) | Job diário Isolation Forest + PCA + Autoencoder funciona | Medium / Medium / Medium | Depende de B1 (DynamoDB ready) |
| B7 | Followup | U6 | US8.1 (completo) | Follow-up automático com cadências funciona | Low / Low / Low | Depende de B1 (DynamoDB ready) |
| B8 | Dashboard | U7 | US7.1, US15.1 | Dashboard Streamlit + DashAPI funcionam | High / High / Low | Depende de B1 (DynamoDB ready) |

---

## Bolt Details

### B1: Walking Skeleton

**Units**: U1 (Core Conversation)  
**Stories**: US1.1 (Iniciar Conversa), US1.2 (Conversar com Texto Livre), US2.1 (Detectar Intenção - básico), US2.2 (Coletar Informações Básicas - básico)  
**Definition of Done**:
- POST /webhook recebe update do Telegram e responde 200
- Sessão é criada no DynamoDB
- LLM (mockado) responde com mensagem básica
- Intenção é detectada (mock/simples regras)
- Informações básicas são coletadas (mock)

**Confidence Hypothesis**: Telegram → Lambda → LLM funciona end-to-end (mock)

**Estimated Effort**: 2-3 dias

---

### B2: Core Conversation Complete

**Units**: U1 (Core Conversation)  
**Stories**: US1.1-US1.3, US2.1-US2.2, US3.1-US3.2, US4.1-US4.2, US5.1, US6.1, US10.1, US13.1, US14.1  
**Definition of Done**:
- Telegram webhook integrado (real)
- OpenRouter LLM integrado (real)
- SecurityLayer (PII masking) implementado
- RAG (FAISS local) implementado
- Qualificação e scoring implementados
- Agendamento (calendário simulado) implementado
- Handoff inteligente implementado
- Roleta de distribuição implementada
- Meta de qualificação ≥ 60% atingida
- Segurança de dados (LGPD) implementada

**Confidence Hypothesis**: LLM + RAG + SecurityLayer funcionam juntos

**Estimated Effort**: 7-10 dias

---

### B3: Async Voice

**Units**: U2 (Async Voice)  
**Stories**: US1.3 (completo com transcrição faster-whisper)  
**Definition of Done**:
- SQS voice configurada
- VoiceAdapter Lambda implementada
- faster-whisper layer implementado
- Transcrição de áudio funciona
- Texto transcrito entra no fluxo conversacional

**Confidence Hypothesis**: Transcrição de voz funciona assincronamente

**Estimated Effort**: 2-3 dias

---

### B4: Async CRM

**Units**: U3 (Async CRM)  
**Stories**: US11.1 (completo com MCP HubSpot)  
**Definition of Done**:
- SQS CRM configurada
- CRMAdapter Lambda implementada
- MCP HubSpot integration implementada (demo única ao vivo)
- CRM simulado (CSV/Excel) funciona
- Lead qualificado sincronizado com CRM

**Confidence Hypothesis**: CRM integration funciona assincronamente

**Estimated Effort**: 2-3 dias

---

### B5: Async Ingest

**Units**: U4 (Async Ingest)  
**Stories**: US12.1 (completo)  
**Definition of Done**:
- SES configurado para receber e-mails
- ContactIngest Lambda implementada
- Extração de dados de e-mail funciona
- Sessão aberta automaticamente no bot
- Primeira mensagem enviada como o lead

**Confidence Hypothesis**: Ingestão de e-mail funciona assincronamente

**Estimated Effort**: 1-2 dias

---

### B6: Anomaly

**Units**: U5 (Anomaly)  
**Stories**: US9.1 (completo)  
**Definition of Done**:
- EventBridge job diário configurado
- AnomalyDetector Lambda implementada
- Feature extraction por conversa implementada
- Isolation Forest + PCA + Autoencoder implementados
- Alerta emitido no dashboard
- Agendamento restrito para leads suspeitos

**Confidence Hypothesis**: Job diário Isolation Forest + PCA + Autoencoder funciona

**Estimated Effort**: 3-4 dias

---

### B7: Followup

**Units**: U6 (Followup)  
**Stories**: US8.1 (completo)  
**Definition of Done**:
- EventBridge cadências configuradas (dia 2, 5, 9)
- Step Functions wait states implementados
- Followup Lambda implementada
- Context da última conversa mantido
- Janela de silêncio respeitada

**Confidence Hypothesis**: Follow-up automático com cadências funciona

**Estimated Effort**: 2-3 dias

---

### B8: Dashboard

**Units**: U7 (Dashboard)  
**Stories**: US7.1, US15.1  
**Definition of Done**:
- DashAPI Lambda implementada
- KPIs agregados do DynamoDB e CloudWatch
- Dashboard Streamlit implementado
- Login via Cognito implementado
- KPIs, esteira Kanban, alertas exibidos
- Observabilidade (CloudWatch) implementada

**Confidence Hypothesis**: Dashboard Streamlit + DashAPI funcionam

**Estimated Effort**: 3-4 dias

---

## Parallel Development Opportunities

**Após B1 (Walking Skeleton) completo:**
- B3, B4, B5, B6, B7, B8 podem ser desenvolvidos em paralelo (todas dependem apenas de B1)
- B2 deve ser sequencial após B1 (B2 é U1 completo, que é walking skeleton expandido)

**Ordem sugerida (WSJF score):**
1. B1 (Walking Skeleton) — prioridade máxima
2. B2 (Core Conversation Complete) — expandido walking skeleton
3. B3, B4, B5, B6, B7, B8 (paralelo) — dependem apenas de B1
4. B8 (Dashboard) — alta prioridade para visibilidade

---

## Risk and Sequencing Rationale

**Risks Identified:**
- **Risk 1**: OpenRouter LLM pode ter downtime ou mudança de pricing — mitigado por mock em walking skeleton (Q5-A)
- **Risk 2**: FAISS pode não caber em memória da Lambda — mitigado por POC size (100-200 docs)
- **Risk 3**: faster-whisper layer pode ser pesado para Lambda — mitigado por Lambda com mais memória
- **Risk 4**: Prazo hackathon (12 de outubro) — mitigado por walking skeleton primeiro e paralelismo

**Sequencing Rationale:**
- Walking skeleton primeiro (B1) valida arquitectura básica com mocks, reduzindo risco técnico
- Core Conversation Complete (B2) expandido walking skeleton com reais (Telegram + OpenRouter)
- Paralelismo de B3-B8 reduz tempo total
- Dashboard (B8) priorizado após B2 para visibilidade rápida