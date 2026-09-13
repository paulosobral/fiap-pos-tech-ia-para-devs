# Decision Log — Agente SDR Imobiliário B2B (W Levitt)

> Registro consolidado das decisões tomadas durante a fase Ideation. Conversa: pt-BR.

## Decisões de escopo e produto

| # | Decisão | Fonte | Status |
|---|---|---|---|
| D-01 | Canal de atendimento externo = Telegram (única via); WhatsApp fora do escopo | intent-capture Q2-A/Q8 | Decidido |
| D-02 | Canal interno do time = dashboard | intent-capture Q8 | Decidido |
| D-03 | Detecção de anomalias é objetivo da POC (plus/diferencial), não exigida pelo enunciado | intent-capture (assumption confirmada) | Decidido |
| D-04 | Integração CRM = HubSpot real via MCP numa demo única ao vivo (MCP Inspector) + crm-adapter simulado default / Private App REST | intent-capture (confirmado 2026-09-12) | Decidido |
| D-05 | Trilha de auditoria da roleta/esteira = timeline do lead no dashboard (consulta simples do DynamoDB), sem auditoria formal imutável | intent-capture (confirmado) | Decidido |
| D-06 | Reporte formal periódico (semanal/mensal) fora da POC; dashboard em tempo real cobre o agregado | intent-capture Q7 | Decidido |
| D-07 | Follow-up com contexto + calendário (ICS) fora da POC (dor central, mas excluída por decisão do usuário) | scope-definition | Decidido |
| D-08 | Ingestão de e-mail (SES, cenário C7) fora da POC; apenas voice no item voice | scope-definition | Decidido |
| D-09 | Voice = faster-whisper em Lambda layer, transcrição desacoplada via SQS | scope-definition | Decidido |
| D-10 | Roleta + esteira Kanban = nice-to-have (P2), incluída no escopo mínimo | scope-definition | Decidido |
| D-11 | Posicionamento competitivo = nicho B2B corporativo + fully serverless baixo custo + qualificação proprietária com score explicável | market-research Q3 | Decidido |
| D-12 | POC 100% em dados sintéticos end-to-end (imóveis + leads + CRM) | feasibility (compliance) | Decidido |
| D-13 | Custo ~R$15/mês com modelo econômico (DeepSeek V3); Claude Haiku = upgrade documentado | feasibility | Decidido |
| D-14 | SSM Parameter Store no lugar de Secrets Manager (R$0) | feasibility | Decidido |
| D-15 | EventBridge Scheduler (não Step Functions) para follow-up futuro | feasibility | Decidido |
| D-16 | Região us-east-1 na POC; sa-east-1 no roadmap | feasibility | Decidido |
| D-17 | MoSCoW sequencial: núcleo primeiro; congelar infra ≥1 semana antes da gravação | feasibility/scope | Decidido |
| D-18 | Limite rígido = data da gravação do vídeo (não 12/10) | scope-definition Q5-D | Decidido |
| D-19 | Dashboard é a única UI a desenvolver (Streamlit, desktop); Telegram e MCP Inspector não são UI do produto | rough-mockups Q1/Q5 | Decidido |
| D-20 | Padrão limpo/neutro Streamlit default; boas práticas básicas de acessibilidade | rough-mockups Q4/Q6 | Decidido |
| D-21 | Anomalia = decisão do corretor/SDR no dashboard (reclassificar como lead ou manter como suspeito), registrada na timeline | rough-mockups | Decidido |

## Decisões de processo (workflow)

| # | Decisão | Fonte | Status |
|---|---|---|---|
| D-22 | Scope `feature` (workflow-selected) | intent-capture | Decidido |
| D-23 | Team-formation pulado — dev solo | workflow | Decidido |
| D-24 | Conversation language: pt-BR | conversa | Decidido |

## Riscos abertos (para Inception)

- R-01: botão `[Importar via MCP HubSpot]` no wireframe W2 contradiz a decisão locked (MCP = demo única; produção usa adapter REST) — tratar em refined-mockups.
- R-02: botões de reclassificar/manter suspeito devem ser condicionais à existência de anomalia — tratar em refined-mockups.
- R-03: `[Agendar]` deve estar desabilitado para leads suspeitos (SC-02) — tratar em refined-mockups.
- R-04: esteira Kanban do wireframe W1 sem coluna "Suspeito" e soma (595) ≠ total de leads (1.248) — tratar em refined-mockups.
- R-05: login presente nos fluxos mas POC sem auth — confirmar tratamento em refined-mockups.