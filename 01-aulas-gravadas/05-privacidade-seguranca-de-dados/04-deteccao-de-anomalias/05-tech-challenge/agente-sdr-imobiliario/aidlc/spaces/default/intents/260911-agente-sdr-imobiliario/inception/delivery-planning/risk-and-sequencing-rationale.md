# Risk and Sequencing Rationale — Agente SDR Imobiliário B2B

> Estágio Delivery Planning (Inception). Fonte: bolt-plan.md, team-allocation.md, requirements.md, components.md.

---

## Risk Assessment

### Risk 1: OpenRouter LLM Downtime or Pricing Change

**Description**: OpenRouter pode ter downtime ou mudança de pricing que impacta a POC.

**Mitigation**:
- Mock OpenRouter em walking skeleton (B1) — Q5-A (Mockar tudo)
- Implementar fallback para outro provedor (Bedrock) em B2
- Monitorar custo e performance em B8 (Dashboard)

**Residual Risk**: Low — mock inicial, fallback em B2

---

### Risk 2: FAISS Index Memory Overflow

**Description**: Índice FAISS pode não caber em memória da Lambda se base de imóveis crescer.

**Mitigation**:
- POC limitada a 100-200 docs (conforme PRD)
- Lambda configurada com memória adequada (1024MB+)
- ADR-003 define troca por Bedrock Knowledge Bases + OpenSearch Serverless se necessário

**Residual Risk**: Low — POC size limitada, ADR define caminho de migração

---

### Risk 3: faster-whisper Layer Heavy for Lambda

**Description**: faster-whisper layer pode ser pesado para Lambda (B3).

**Mitigation**:
- Lambda configurada com mais memória (2048MB+)
- Timeout adequado (30s+)
- Fallback para transcrição manual se falhar

**Residual Risk**: Medium — performance pode ser issue, mas há fallback

---

### Risk 4: Hackathon Deadline (12 de outubro)

**Description**: Prazo apertado pode impedir entrega completa.

**Mitigation**:
- Walking skeleton primeiro (B1) — reduz risco técnico
- Paralelismo de B3-B8 após B1 — reduz tempo total
- Priorização de B8 (Dashboard) para visibilidade rápida
- Priorização de B2 (Core Conversation Complete) — maior valor

**Residual Risk**: Medium — depende de paralelismo efetivo

---

### Risk 5: PII Masking Complexity

**Description**: SecurityLayer (PII masking) pode ser complexo de implementar corretamente.

**Mitigation**:
- ADR-008 define módulo interno de ConversationRouter
- Implementar validação de saída contra vazamento de PII (regex)
- Testar extensivamente com dados sintéticos

**Residual Risk**: Medium — complexidade, mas ADR define caminho claro

---

### Risk 6: CRM Integration MCP Complexity

**Description**: MCP integration com HubSpot/Kenlo/Facilita pode ser complexo.

**Mitigation**:
- ADR-006 define camada MCP genérica
- POC roda contra CRM simulado (CSV/Excel)
- Demo única ao vivo com HubSpot se tempo permitir

**Residual Risk**: Low — CRM simulado para POC

---

## Sequencing Rationale

### Why Walking Skeleton First (B1)?

**Alignment with Team Practice**: Prática aprovada em Practices Discovery define "Walking Skeleton" como prioridade.

**Technical Justification**:
- Valida arquitectura básica (Telegram → Lambda → LLM) com mocks
- Reduz risco técnico cedo antes de integrar dependências reais
- Prove que a arquitetura funciona end-to-end antes de expansão
- 2-3 dias apenas — baixo custo, alto valor de informação

**Value Hypothesis**: Se walking skeleton falhar, o problema é arquitetural, não de integração de dependências externas.

---

### Why Core Conversation Complete Second (B2)?

**Value-First Approach**:
- B2 expande walking skeleton com reais (Telegram + OpenRouter)
- Entrega maior valor (conversação completa)
- Valida LLM + RAG + SecurityLayer juntos — risco crítico

**Technical Justification**:
- Depende de B1 (DynamoDB, Telegram webhook ready)
- 7-10 dias — maior esforço, mas maior valor
- Desbloqueia B3-B8 (todos dependem de U1)

---

### Why Parallel B3-B8 After B1?

**Efficiency**:
- B3-B8 dependem apenas de B1 (DynamoDB ready, SQS ready)
- Paralelismo reduz tempo total de 22-35 dias para 14-19 dias
- Viável com 1 pessoa (B2 sequencial, B3-B8 paralelo)

**Risk Mitigation**:
- B1 prova que DynamoDB e SQS funcionam
- B3-B8 são independentes — falha em um não bloqueia outros

---

### Why Dashboard (B8) Priority in Parallel Batch?

**Visibility**:
- Dashboard fornece visibilidade rápida para stakeholders
- Alta prioridade (Q1-A: partes mais valiosas)
- 3-4 dias — esforço moderado, alto valor

**Technical Justification**:
- Depende de B1 (DynamoDB ready)
- Depende de B5 (Anomaly alerts ready) — mas pode ser parcial

---

## WSJF Scoring Model

**WSJF Formula**: (User Business Value + Time Criticality + Risk Reduction Value) ÷ Job Size

| Bolt | User Business Value | Time Criticality | Risk Reduction Value | Job Size | WSJF Score |
|------|---------------------|-----------------|---------------------|----------|------------|
| B1 | High (validates architecture) | High (blocks everything) | High (reduces technical risk) | Small (2-3d) | Very High |
| B2 | Very High (complete conversation) | High (blocks most value) | Medium (validates LLM+RAG) | Large (7-10d) | High |
| B3 | Medium (voice support) | Low (nice-to-have) | Low (low risk) | Small (2-3d) | Medium |
| B4 | Medium (CRM integration) | Low (nice-to-have) | Low (low risk) | Small (2-3d) | Medium |
| B5 | Low (anomaly detection) | Low (nice-to-have) | Medium (ML risk) | Medium (3-4d) | Medium |
| B6 | Low (follow-up) | Low (nice-to-have) | Low (low risk) | Small (2-3d) | Low |
| B7 | High (dashboard visibility) | Medium (nice-to-have) | Low (low risk) | Medium (3-4d) | High |

**Sequencing by WSJF**: B1 → B2 → B7 → B3/B4/B5/B6

---

## Alternative Sequencing Considered

### Alternative A: Risk-First (SecurityLayer → RAG → Anomaly)

**Pros**: Valida riscos técnicos cedo
**Cons**: Não entrega valor inicial ao usuário, walking skeleton validado mas sem funcionalidade real
**Rejected**: Prática Walking Skeleton define value-first approach

### Alternative B: Value-First (Dashboard First)

**Pros**: Visibilidade imediata
**Cons**: Dashboard depende de DynamoDB + CloudWatch, que não existem sem B1
**Rejected**: Dependência técnica invalida sequenciamento

### Alternative C: All Parallel (B1-B8 together)

**Pros**: Máximo paralelismo
**Cons**: Alta coordenação, B1 bloqueia todos, não viável com 1 pessoa
**Rejected**: Complexidade e dependências invalidam paralelismo total