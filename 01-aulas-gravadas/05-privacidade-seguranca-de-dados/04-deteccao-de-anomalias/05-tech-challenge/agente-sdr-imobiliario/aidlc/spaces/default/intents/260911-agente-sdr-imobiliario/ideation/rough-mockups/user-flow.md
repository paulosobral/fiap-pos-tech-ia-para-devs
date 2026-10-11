# User Flow — Agente SDR Imobiliário (W Levitt)

> Estágio Rough Mockups & Concept Visualization (Ideation). Fonte: rough-mockups-questions (Q2: fluxos A, B e C). Diagramas ASCII.

---

## F1. Cliente final (happy path) — Telegram

```
[Cliente] --msg (texto ou voz)--> [Telegram bot] --> [conversation-router]
                                                       |
                              (se voz)                v
                              [faster-whisper] --> [transcrição] -->+
                                                       |            |
                                                       v            |
                                   [security-layer]  (masking PII, anti-injection)
                                                       |
                                                       v
                                   [RAG: imóveis]  (base simulada)
                                                       |
                                                       v
                                   [qualificação]  (score, intenção)
                                                       |
                                                       v
                                   [roleta]  (rota → corretor)
                                                       |
                                                       v
                                   [handoff ao corretor]  (resumo qualificado)
                                                       |
                                                       v
                              [dashboard: timeline do lead]
```

> Voice é um caminho conversacional no Telegram (não URA rígida): o cliente pode responder por voz a qualquer momento, e o bot segue a conversa normalmente após a transcrição.

## F2. Corretor/SDR — dashboard

```
[Corretor] --login--> [dashboard Streamlit]
                          |
                          +--> [filtra leads] --> [vê timeline] --> [age: agenda / atualiza status]
                          |
                          +--> [importa lead via MCP HubSpot]  (demo única)
                          |
                          +--> [anomalia detectada] --> [corretor/SDR analisa]
                                                          |
                                                          +--> [reclassifica como lead]  (se falso positivo)
                                                          +--> [mantém como suspeito]  (sem agendamento)
```

## F3. Gestor — dashboard

```
[Gestor] --login--> [dashboard Streamlit]
                          |
                          +--> [KPIs agregados]  (leads, qualificados, intenção, tempo de resposta)
                          |
                          +--> [alertas de anomalia]  (job diário)
```

---

## Nota

- Telegram (F1): app existente, sem UI a desenvolver — fluxo conversacional do bot.
- Dashboard (F2/F3): única UI a desenvolver (Streamlit, desktop).
- MCP HubSpot (F2): demo única ao vivo, fora do escopo de UI do produto.