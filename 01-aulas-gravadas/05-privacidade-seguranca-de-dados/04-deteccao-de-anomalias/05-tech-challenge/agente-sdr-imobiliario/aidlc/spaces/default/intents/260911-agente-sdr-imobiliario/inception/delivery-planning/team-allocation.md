# Team Allocation — Agente SDR Imobiliário B2B

> Estágio Delivery Planning (Inception). Fonte: delivery-planning-questions (Q4-A: 1 pessoa sequencial).

---

## Team Composition

**Single Developer** (Q4-A)

- **Role**: Full-stack Developer
- **Responsibilities**: Desenvolver todos os Bolts sequencialmente
- **Skills**: Python, AWS (Lambda, DynamoDB, SQS, EventBridge, Step Functions), LLM (OpenRouter), RAG (FAISS), Streamlit
- **Ideal para**: Hackathon com prazo limitado

---

## Bolt-to-Developer Mapping

| Bolt ID | Bolt Name | Developer | Notes |
|---------|-----------|-----------|-------|
| B1 | Walking Skeleton | Developer 1 | Prioridade máxima |
| B2 | Core Conversation Complete | Developer 1 | Expandido walking skeleton |
| B3 | Async Voice | Developer 1 | Paralelo após B1 |
| B4 | Async CRM | Developer 1 | Paralelo após B1 |
| B5 | Async Ingest | Developer 1 | Paralelo após B1 |
| B6 | Anomaly | Developer 1 | Paralelo após B1 |
| B7 | Followup | Developer 1 | Paralelo após B1 |
| B8 | Dashboard | Developer 1 | Paralelo após B1 |

---

## Capacity Planning

**Assumptions:**
- 1 developer
- Prazo: 12 de outubro (aprox. 2-3 semanas)
- 40h/semana (hackathon pace)

**Estimated Effort:**
- B1: 2-3 dias
- B2: 7-10 dias
- B3-B8: 12-19 dias (paralelo)

**Total Estimated:**
- Sequencial: 22-35 dias (3-5 semanas) — **pode não bater prazo**
- Paralelo após B1: 14-19 dias (2-3 semanas) — **viável com paralelismo**

**Recommendation:**
- Foco em B1 (Walking Skeleton) primeiro
- B2 (Core Conversation Complete) depois
- B3-B8 em paralelo após B2 (ou priorizar B8 para visibilidade)

---

## Collaboration Model

**Sequential Development** (Q4-A)
- Single developer trabalha em um Bolt por vez
- Branch per feature conforme prática aprovada
- PR-based integration
- Review self ou peer (se disponível)

**Benefits:**
- Simplicidade máxima
- Sem overhead de coordenação
- Foco total em um Bolt

**Drawbacks:**
- Maior tempo total se não houver paralelismo
- Single point of failure