# Phase Check — Ideation

> Verificação de fronteira Ideation → Inception (Approval & Handoff, Step 4). Conversa: pt-BR.

## 1. Consistência Intent → Scope → Intent Backlog

- **Intent statement** (intent-capture) → define problema, cliente-alvo, métricas de sucesso e sinal de escopo.
- **Scope document** (scope-definition) → traduz o intent em in-scope/out-of-scope e boundary decisions.
- **Intent backlog** (scope-definition) → proto-units priorizadas (SC-01..SC-12) com dependências.

**Resultado: CONSISTENTE.** Cada proto-unit do backlog rastreia a um item de escopo; o escopo rastreia ao intent. Métricas de sucesso (1ª resposta <10s, qualificados ≥60%, intenção ≥85%) presentes no intent e citadas no escopo.

## 2. Todos os itens de escopo têm backing de viabilidade

| Item de escopo | Backing de viabilidade |
|---|---|
| Núcleo conversacional (SC-01) | feasibility-assessment (conversation-router Excelente; RAG Excelente) |
| Detecção de anomalias (SC-02) | feasibility-assessment (anomaly-detector OK c/ restrições — treinar offline) |
| CRM HubSpot (SC-03) | feasibility-assessment (crm-adapter Bom; MCP separado do adapter) |
| Voice (SC-04) | feasibility-assessment (faster-whisper Possível, frágil) |
| Roleta + esteira (SC-05) | feasibility-assessment (lead-router Total) |

**Resultado: CONSISTENTE.** Todos os itens in-scope têm parecer de viabilidade técnica na feasibility-assessment; riscos com mitigação no constraint-register (CT-01..11, CO-01..07, CR-C1..C8).

## 3. Decisões e riscos herdados para Inception

- **24 decisões** registradas no decision-log (D-01..D-24).
- **5 riscos de refinamento** (R-01..R-05) herdados do review de rough-mockups para refined-mockups.

**Resultado: CONSISTENTE.** Nenhuma contradição entre intent, escopo e backlog; nenhum item de escopo sem backing de viabilidade.

## Veredito

**PASSOU.** A fase Ideation está coerente e pronta para avançar a Inception.