# Unit of Work Dependency — Agente SDR Imobiliário B2B

> Estágio Units Generation (Inception). Fonte: components.md, decisions.md, unit-of-work.md.

---

## Dependency DAG

**Dependências entre unidades:**
- U1 (Core Conversation) é a unidade principal — não depende de nenhuma outra
- U2 (Async Voice) depende de U1 (ConversationRouter enfileira áudio para SQS)
- U3 (Async CRM) depende de U1 (ConversationRouter enfileira leads qualificados para SQS)
- U4 (Async Ingest) depende de U1 (ContactIngest abre sessão via ConversationRouter)
- U5 (Anomaly) depende de U1 (lê conversas do DynamoDB criadas por ConversationRouter)
- U6 (Followup) depende de U1 (lê contexto do DynamoDB criado por ConversationRouter)
- U7 (Dashboard) depende de U1 (lê métricas de negócio do DynamoDB) e U5 (lê alertas de anomalias)

**Integração entre unidades:**
- U1 → U2: SQS (fila de áudio)
- U1 → U3: SQS (fila CRM)
- U4 → U1: DynamoDB (sessão criada)
- U1 → U5: DynamoDB (conversas lidas)
- U1 → U6: DynamoDB (contexto lido)
- U1 → U7: DynamoDB (métricas lidas)
- U5 → U7: DynamoDB (alertas lidos)

**Paralelismo (Q3-A - Estrita topológica):**
- Unidades só podem começar quando todas as dependências estão completas
- U1 deve ser implementada primeiro
- U2, U3, U4, U5, U6 podem ser implementadas em paralelo após U1
- U7 pode ser implementada após U1 e U5

---

## Integration Points

| Unit | Integração | Tipo | Descrição |
|------|------------|------|-----------|
| U1 → U2 | SQS | Async | Enfileira mensagens de áudio para transcrição |
| U1 → U3 | SQS | Async | Enfileira leads qualificados para CRM |
| U4 → U1 | DynamoDB | Shared Data | Cria sessão no DynamoDB |
| U1 → U5 | DynamoDB | Shared Data | Lê conversas para análise de anomalias |
| U1 → U6 | DynamoDB | Shared Data | Lê contexto para follow-up |
| U1 → U7 | DynamoDB | Shared Data | Lê métricas de negócio |
| U5 → U7 | DynamoDB | Shared Data | Lê alertas de anomalias |
| U7 → DashAPI | HTTP/REST | Sync | Consome GET /api/kpis |

---

## Parallel Development Opportunities

**Após U1 completa:**
- U2, U3, U4, U5, U6 podem ser desenvolvidas em paralelo (todas dependem apenas de U1)
- U7 pode ser desenvolvida após U5

**Ordem topológica sugerida:**
1. U1 (Core Conversation) — walking skeleton
2. U2, U3, U4, U5, U6 (paralelo)
3. U7 (após U5)

---

## Machine-Readable Edge Block

```yaml
units:
  - name: u1-core-conversation
    kind: service
    depends_on: []
  - name: u2-async-voice
    kind: service
    depends_on: [u1-core-conversation]
  - name: u3-async-crm
    kind: service
    depends_on: [u1-core-conversation]
  - name: u4-async-ingest
    kind: service
    depends_on: [u1-core-conversation]
  - name: u5-anomaly
    kind: service
    depends_on: [u1-core-conversation]
  - name: u6-followup
    kind: service
    depends_on: [u1-core-conversation]
  - name: u7-dashboard
    kind: ui
    depends_on: [u1-core-conversation, u5-anomaly]
```