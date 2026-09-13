# Wireframes (Low-Fidelity) — Agente SDR Imobiliário (W Levitt)

> Estágio Rough Mockups & Concept Visualization (Ideation). Fonte: intent-statement, scope-document, intent-backlog, rough-mockups-questions (Q1–Q6). UI principal: dashboard Streamlit (desktop). Telegram é app existente — sem UI a desenvolver (Q5). MCP Inspector é ferramenta de demo, não UI do produto (Q1-C). Padrão limpo/neutro Streamlit default (Q4-A). Boas práticas básicas de acessibilidade (Q6-A).

---

## W1. Dashboard — Visão geral (KPIs agregados)

```
+------------------------------------------------------------------+
| [W Levitt · Agente SDR]  Filtros: [Período v] [Canal v] [Status v]|
+------------------------------------------------------------------+
|  KPI: Leads    | KPI: Qualificados | KPI: Intenção    | KPI: 1ª   |
|  1.248         | 62%               | 87%              | resp. 4s  |
+------------------------------------------------------------------+
|  Esteira Kanban (status dos leads)                               |
|  +-----------+ +-----------+ +-----------+ +-----------+        |
|  | Novo      | | Qualif.   | | Em negoc. | | Anomalia  |        |
|  | 312       | | 180       | | 96        | | 7         |        |
|  +-----------+ +-----------+ +-----------+ +-----------+        |
+------------------------------------------------------------------+
|  [Timeline do lead selecionado]   |  [Alertas de anomalia]       |
|  entrada → qualificação → rota →  |  • Lead #4821 suspeito      |
|  status → anomalia                |  • Pico de intenção falsa   |
+------------------------------------------------------------------+
```

**Acessibilidade (W1):** h1 no título; landmarks header/main; entrada por teclado nos filtros e cards.

---

## W2. Dashboard — Detalhe do lead (timeline + ação)

```
+------------------------------------------------------------------+
| ← Voltar   Lead #4821 · Empresa XYZ · laje corporativa 800m²     |
+------------------------------------------------------------------+
|  Status: [Qualificado v]   Intenção: Locação   Score: 82          |
+------------------------------------------------------------------+
|  Timeline:                                                       |
|  • 14:02 entrada (Telegram)                                      |
|  • 14:03 qualificação (score 82)                                 |
|  • 14:04 rota roleta → corretor A (regra: score≥70)              |
|  • 14:05 handoff enviado ao corretor                             |
+------------------------------------------------------------------+
|  [Importar via MCP HubSpot]  [Agendar]  [Atualizar status]       |
+------------------------------------------------------------------+
```

**Acessibilidade (W2):** h2 nos blocos; landmarks main; foco visível nos botões de ação.

---

## W3. Conversa Telegram (cliente final) — sem UI a desenvolver

```
[Bot Agente SDR]  (app Telegram existente — não é UI da POC)
┌──────────────────────────────┐
│ Olá! Sou o assistente da    │
│ W Levitt. Posso ajudar com  │
│ espaço corporativo?         │
│ [Aviso de privacidade]      │
└──────────────────────────────┘
```

**Acessibilidade (W3):** não aplicável — app de terceiros; o bot deve usar texto claro e sem depender de cor.

---

## W4. MCP Inspector (demo CRM HubSpot) — ferramenta, não UI do produto

```
[MCP Inspector]  (demo única ao vivo — não é UI da POC)
┌──────────────────────────────┐
│ OAuth/PKCE (redirect local)  │
│ → importa lead no HubSpot    │
└──────────────────────────────┘
```

**Acessibilidade (W4):** não aplicável — ferramenta de demonstração.

---

## Nota

- Telegram (Q5): app existente, **sem desenvolvimento de UI** — apenas o comportamento conversacional do bot.
- Dashboard (Q1-A): única UI a desenvolver (Streamlit, desktop).
- MCP Inspector (Q1-C): demo única, fora do escopo de UI do produto.